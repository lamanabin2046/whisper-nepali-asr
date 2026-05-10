import os, json, torch
import pandas as pd
import torchaudio
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup, Adafactor
from tqdm import tqdm
import evaluate

MODEL_ID      = "openai/whisper-base"
CLEANED_DIR   = os.path.expanduser("~/whisper_distil_nepali/data/cleaned")
AUDIO_DIR     = os.path.expanduser("~/whisper_distil_nepali/data/raw/asr_nepali/data")
OUTPUT_DIR    = os.path.expanduser("~/whisper_distil_nepali/models/exp2_whisper_base_cleaned")
SAMPLE_RATE   = 16000
BATCH_SIZE    = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS    = 5
WARMUP_STEPS  = 500
LANGUAGE      = "nepali"
TASK          = "transcribe"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class NepaliASRDataset(Dataset):
    def __init__(self, tsv_path, audio_dir, processor):
        self.df        = pd.read_csv(tsv_path, sep="\t")
        self.audio_dir = audio_dir
        self.processor = processor
    def __len__(self):
        return len(self.df)
    def get_audio_path(self, utt_id):
        return os.path.join(self.audio_dir, utt_id[:2], f"{utt_id}.flac")
    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        path     = self.get_audio_path(row["utt_id"])
        waveform, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        waveform = waveform.mean(dim=0)
        inputs = self.processor(waveform.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        labels = self.processor.tokenizer(row["transcription"], return_tensors="pt").input_ids
        return {"input_features": inputs.input_features.squeeze(0), "labels": labels.squeeze(0)}

@dataclass
class DataCollator:
    processor: WhisperProcessor
    def __call__(self, features):
        input_features = torch.stack([f["input_features"] for f in features])
        label_list     = [f["labels"] for f in features]
        max_len        = max(l.shape[0] for l in label_list)
        padded         = torch.full((len(label_list), max_len), -100, dtype=torch.long)
        for i, lab in enumerate(label_list):
            padded[i, :lab.shape[0]] = lab
        return {"input_features": input_features, "labels": padded}

def main():
    print("=" * 60)
    print("Exp 2 - Fine-tune Whisper Base (Cleaned Data)")
    print(f"Device: {DEVICE} | Batch: {BATCH_SIZE} | Epochs: {NUM_EPOCHS}")
    print("=" * 60)

    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
    model     = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
    model.config.suppress_tokens    = []
    model = model.to(DEVICE)

    collator     = DataCollator(processor=processor)
    train_ds     = NepaliASRDataset(os.path.join(CLEANED_DIR, "train.tsv"), AUDIO_DIR, processor)
    val_ds       = NepaliASRDataset(os.path.join(CLEANED_DIR, "val.tsv"),   AUDIO_DIR, processor)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collator, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator, num_workers=4)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    optimizer   = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=LEARNING_RATE)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler   = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

    wer_metric  = evaluate.load("wer")
    best_wer    = float("inf")
    train_stats = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for batch in progress:
            input_features = batch["input_features"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)
            outputs = model(input_features=input_features, labels=labels)
            loss    = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        model.eval()
        all_preds, all_refs = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_features = batch["input_features"].to(DEVICE)
                labels         = batch["labels"]
                generated = model.generate(input_features, max_new_tokens=225,
                    forced_decoder_ids=processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK))
                preds = processor.batch_decode(generated, skip_special_tokens=True)
                refs  = processor.batch_decode(torch.where(labels == -100,
                    torch.tensor(processor.tokenizer.pad_token_id), labels), skip_special_tokens=True)
                all_preds.extend(preds)
                all_refs.extend(refs)

        wer = wer_metric.compute(predictions=all_preds, references=all_refs)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | WER: {wer*100:.2f}%")

        if wer < best_wer:
            best_wer = wer
            model.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
            processor.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
            print(f"Best model saved! WER: {wer*100:.2f}%")

        train_stats.append({"epoch": epoch, "loss": round(avg_loss, 4), "wer": round(wer * 100, 2)})

    with open(os.path.join(OUTPUT_DIR, "train_stats.json"), "w") as f:
        json.dump(train_stats, f, indent=2)
    print(f"Done! Best WER: {best_wer*100:.2f}%")

if __name__ == "__main__":
    main()
