import os, json, time, torch
import torch.nn.functional as F
import pandas as pd
import torchaudio
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_linear_schedule_with_warmup,
    Adafactor
)
from tqdm import tqdm
import evaluate
import wandb

TEACHER_PATH  = os.path.expanduser("~/whisper_distil_nepali/models/exp4_whisper_medium_cleaned/best_model")
STUDENT_PATH  = os.path.expanduser("~/whisper_distil_nepali/models/exp3_whisper_small_cleaned/best_model")
CLEANED_DIR   = os.path.expanduser("~/whisper_distil_nepali/data/cleaned")
AUDIO_DIR     = os.path.expanduser("~/whisper_distil_nepali/data/raw/asr_nepali/data")
OUTPUT_DIR    = os.path.expanduser("~/whisper_distil_nepali/models/exp_distill_medium_to_small")
SAMPLE_RATE   = 16000
BATCH_SIZE    = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS    = 20
WARMUP_STEPS  = 200
LANGUAGE      = "nepali"
TASK          = "transcribe"
ALPHA         = 0.5
BETA          = 0.5
TEMPERATURE   = 2.0
SEED          = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
n_gpus = torch.cuda.device_count()

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)

class NepaliASRDataset(Dataset):
    def __init__(self, tsv_path, audio_dir, processor):
        self.df = pd.read_csv(tsv_path, sep="\t")
        self.audio_dir = audio_dir
        self.processor = processor
    def __len__(self): return len(self.df)
    def get_audio_path(self, utt_id):
        return os.path.join(self.audio_dir, utt_id[:2], f"{utt_id}.flac")
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = torchaudio.load(self.get_audio_path(row["utt_id"]))
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
        label_list = [f["labels"] for f in features]
        max_len = max(l.shape[0] for l in label_list)
        padded = torch.full((len(label_list), max_len), -100, dtype=torch.long)
        for i, lab in enumerate(label_list): padded[i, :lab.shape[0]] = lab
        return {"input_features": input_features, "labels": padded}

def distillation_loss(student_logits, teacher_logits, labels):
    vocab_size = student_logits.shape[-1]
    hard_loss = F.cross_entropy(student_logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=-100)
    mask = labels != -100
    if mask.sum() == 0: return hard_loss, hard_loss.item(), 0.0
    student_soft = F.log_softmax(student_logits[mask] / TEMPERATURE, dim=-1)
    teacher_soft = F.softmax(teacher_logits[mask] / TEMPERATURE, dim=-1)
    soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (TEMPERATURE ** 2)
    return ALPHA * hard_loss + BETA * soft_loss, hard_loss.item(), soft_loss.item()

def main():
    run = wandb.init(project="nepali-asr-whisper", name=f"distill-medium-to-small-{NUM_EPOCHS}ep",
        config={"teacher": "whisper-medium-finetuned", "student": "whisper-small-finetuned",
                "student_baseline": 13.53, "teacher_wer": 9.77, "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE, "epochs": NUM_EPOCHS,
                "alpha": ALPHA, "beta": BETA, "temperature": TEMPERATURE})
    print("="*60)
    print("Distillation: Medium FT → Fine-tuned Small")
    print(f"Teacher: Whisper Medium FT (9.77% WER)")
    print(f"Student: Whisper Tiny FT   (13.53% WER)")
    print(f"Device:  {DEVICE} | WandB: {run.url}")
    print("="*60)

    teacher_proc = WhisperProcessor.from_pretrained(TEACHER_PATH, language=LANGUAGE, task=TASK)
    teacher = WhisperForConditionalGeneration.from_pretrained(TEACHER_PATH, torch_dtype=torch.float16).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    print(f"Teacher: {sum(p.numel() for p in teacher.parameters())/1e6:.1f}M (frozen)")

    student_proc = WhisperProcessor.from_pretrained(STUDENT_PATH, language=LANGUAGE, task=TASK)
    student = WhisperForConditionalGeneration.from_pretrained(STUDENT_PATH).to(DEVICE)
    student.config.forced_decoder_ids = student_proc.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
    student.config.suppress_tokens = []
    print(f"Student: {sum(p.numel() for p in student.parameters())/1e6:.1f}M (trainable)")

    collator = DataCollator(processor=student_proc)
    train_ds = NepaliASRDataset(os.path.join(CLEANED_DIR, "train.tsv"), AUDIO_DIR, student_proc)
    val_ds   = NepaliASRDataset(os.path.join(CLEANED_DIR, "val.tsv"),   AUDIO_DIR, student_proc)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collator, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator, num_workers=4, pin_memory=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    optimizer = Adafactor(student.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=len(train_loader)*NUM_EPOCHS)
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    best_wer = float("inf")
    train_stats = []
    global_step = 0
    training_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        student.train()
        total_loss = total_hard = total_soft = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for batch in progress:
            input_features = batch["input_features"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            with torch.no_grad():
                t_logits = teacher(input_features=input_features.half(), labels=labels).logits.float()
            s_logits = student(input_features=input_features, labels=labels).logits
            min_len = min(s_logits.shape[1], t_logits.shape[1])
            loss, hard_loss, soft_loss = distillation_loss(s_logits[:,:min_len,:], t_logits[:,:min_len,:], labels[:,:min_len])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item(); total_hard += hard_loss; total_soft += soft_loss
            global_step += 1
            wandb.log({"train/step_loss": loss.item(), "train/step_hard_loss": hard_loss,
                       "train/step_soft_loss": soft_loss, "train/lr": scheduler.get_last_lr()[0], "global_step": global_step})
            progress.set_postfix({"loss": f"{loss.item():.4f}", "hard": f"{hard_loss:.4f}", "soft": f"{soft_loss:.4f}"})

        avg_loss = total_loss/len(train_loader)
        avg_hard = total_hard/len(train_loader)
        avg_soft = total_soft/len(train_loader)

        student.eval()
        total_val_loss = 0; all_preds = []; all_refs = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_features = batch["input_features"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                total_val_loss += student(input_features=input_features, labels=labels).loss.item()
                generated = student.generate(input_features, max_new_tokens=225,
                    forced_decoder_ids=student_proc.get_decoder_prompt_ids(language=LANGUAGE, task=TASK))
                all_preds.extend(student_proc.batch_decode(generated, skip_special_tokens=True))
                all_refs.extend(student_proc.batch_decode(
                    torch.where(labels==-100, torch.tensor(student_proc.tokenizer.pad_token_id).to(DEVICE), labels),
                    skip_special_tokens=True))

        avg_val_loss = total_val_loss/len(val_loader)
        wer = wer_metric.compute(predictions=all_preds, references=all_refs)
        cer = cer_metric.compute(predictions=all_preds, references=all_refs)
        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch:02d} | Train: {avg_loss:.4f} (hard:{avg_hard:.4f} soft:{avg_soft:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f} | WER: {wer*100:.2f}% | CER: {cer*100:.2f}% | Time: {epoch_time/60:.1f}min")

        wandb.log({"epoch": epoch, "train/epoch_loss": avg_loss, "train/hard_loss": avg_hard,
                   "train/soft_loss": avg_soft, "val/loss": avg_val_loss, "val/wer": wer*100,
                   "val/cer": cer*100, "train_val_loss_gap": avg_loss-avg_val_loss,
                   "wer_improvement_over_baseline": 34.47-(wer*100)})

        if wer < best_wer:
            best_wer = wer
            student.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
            student_proc.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
            print(f"  → Best saved! WER: {wer*100:.2f}% | CER: {cer*100:.2f}%")
            wandb.run.summary.update({"best_wer": wer*100, "best_cer": cer*100, "best_epoch": epoch})

        train_stats.append({"epoch": epoch, "train_loss": round(avg_loss,4), "hard_loss": round(avg_hard,4),
                            "soft_loss": round(avg_soft,4), "val_loss": round(avg_val_loss,4),
                            "wer": round(wer*100,2), "cer": round(cer*100,2), "improvement": round(34.47-(wer*100),2)})
        with open(os.path.join(OUTPUT_DIR, "train_stats.json"), "w") as f:
            json.dump(train_stats, f, indent=2)

    wandb.run.summary["total_train_hrs"] = round((time.time()-training_start)/3600, 2)
    wandb.finish()
    print(f"\nDONE! Tiny FT before: 34.47% | After distill: {best_wer*100:.2f}% | Teacher: 9.77%")

if __name__ == "__main__":
    main()
