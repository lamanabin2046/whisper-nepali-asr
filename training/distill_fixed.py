import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
import torchaudio
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Adafactor,
    get_linear_schedule_with_warmup
)
from accelerate import Accelerator
from tqdm import tqdm
import evaluate

# =========================================================
# CONFIG
# =========================================================

TEACHER_MODEL_ID  = "openai/whisper-large-v2"
STUDENT_MODEL_ID  = "openai/whisper-small"

CLEANED_DIR = "/workspace/whisper_nepali/data/cleaned"
AUDIO_DIR   = "/workspace/whisper_nepali/data/raw/asr_nepali/data"
OUTPUT_DIR  = "/workspace/whisper_nepali/models/distill_small_from_largev2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE   = 16000
BATCH_SIZE    = 8
LEARNING_RATE = 1e-5
NUM_EPOCHS    = 10
WARMUP_STEPS  = 500
TEMPERATURE   = 4.0
ALPHA         = 0.3
LANGUAGE      = "nepali"
TASK          = "transcribe"

# =========================================================
# DATASET
# =========================================================

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
        waveform = waveform.mean(dim=0)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )
        labels = self.processor.tokenizer(
            row["transcription"],
            return_tensors="pt"
        ).input_ids.squeeze(0)
        return {
            "input_features": inputs.input_features.squeeze(0),
            "labels": labels
        }

# =========================================================
# COLLATOR
# =========================================================

@dataclass
class DataCollator:
    processor: WhisperProcessor

    def __call__(self, features):
        input_features = torch.stack([f["input_features"] for f in features])
        labels         = [f["labels"] for f in features]
        max_len        = max(l.shape[0] for l in labels)
        padded         = torch.full((len(labels), max_len), -100, dtype=torch.long)
        for i, label in enumerate(labels):
            padded[i, :label.shape[0]] = label
        return {"input_features": input_features, "labels": padded}

# =========================================================
# DISTILLATION LOSS (with token masking)
# =========================================================

def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    # Only compute loss on valid tokens (ignore padding -100)
    mask           = labels != -100
    student_logits = student_logits[mask]
    teacher_logits = teacher_logits[mask]
    labels         = labels[mask]

    # KL divergence loss
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    log_student  = F.log_softmax(student_logits / temperature, dim=-1)
    kl_loss      = F.kl_div(
        log_student, soft_teacher, reduction="batchmean"
    ) * (temperature ** 2)

    # Cross entropy loss
    ce_loss = F.cross_entropy(student_logits, labels)

    # Combined loss
    total_loss = alpha * kl_loss + (1 - alpha) * ce_loss
    return total_loss, kl_loss, ce_loss

# =========================================================
# MAIN
# =========================================================

def main():

    accelerator = Accelerator(mixed_precision="fp16")

    accelerator.print("=" * 60)
    accelerator.print("Distillation: Whisper Large v2 -> Whisper Small")
    accelerator.print(f"Batch     : {BATCH_SIZE} | Epochs: {NUM_EPOCHS}")
    accelerator.print(f"Temp      : {TEMPERATURE} | Alpha : {ALPHA}")
    accelerator.print(f"Teacher   : {TEACHER_MODEL_ID}")
    accelerator.print(f"Student   : {STUDENT_MODEL_ID}")
    accelerator.print("=" * 60)

    # -----------------------------------------------------
    # Processor from teacher
    # -----------------------------------------------------
    processor = WhisperProcessor.from_pretrained(TEACHER_MODEL_ID)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)

    # -----------------------------------------------------
    # Teacher — fp16, frozen, on GPU
    # -----------------------------------------------------
    accelerator.print("Loading teacher (Large v2)...")
    teacher = WhisperForConditionalGeneration.from_pretrained(
        TEACHER_MODEL_ID,
        torch_dtype=torch.float16
    )
    teacher.eval()
    teacher.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task=TASK
    )
    for p in teacher.parameters():
        p.requires_grad = False

    # -----------------------------------------------------
    # Student — fp32, trainable
    # -----------------------------------------------------
    accelerator.print("Loading student (Small)...")
    student = WhisperForConditionalGeneration.from_pretrained(STUDENT_MODEL_ID)
    student.gradient_checkpointing_enable()
    student.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task=TASK
    )

    # -----------------------------------------------------
    # Datasets
    # -----------------------------------------------------
    collator     = DataCollator(processor)
    train_ds     = NepaliASRDataset(
        os.path.join(CLEANED_DIR, "train.tsv"), AUDIO_DIR, processor
    )
    val_ds       = NepaliASRDataset(
        os.path.join(CLEANED_DIR, "val.tsv"), AUDIO_DIR, processor
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collator, num_workers=4, pin_memory=True
    )
    val_loader   = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collator, num_workers=4, pin_memory=True
    )
    accelerator.print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # -----------------------------------------------------
    # Optimizer + Scheduler
    # -----------------------------------------------------
    optimizer   = Adafactor(
        student.parameters(),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=LEARNING_RATE
    )
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )

    # Prepare with accelerator (student only)
    student, optimizer, train_loader, val_loader, scheduler = \
        accelerator.prepare(student, optimizer, train_loader, val_loader, scheduler)

    # Teacher to GPU separately
    teacher = teacher.to(accelerator.device)

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    wer_metric = evaluate.load("wer")
    best_wer   = 999
    stats      = []

    for epoch in range(1, NUM_EPOCHS + 1):

        student.train()
        running_loss = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{NUM_EPOCHS}",
            disable=not accelerator.is_local_main_process
        )

        for batch in progress:
            input_features = batch["input_features"]
            labels         = batch["labels"]

            # Teacher forward (no grad, fp16)
            with torch.no_grad():
                teacher_out    = teacher(
                    input_features=input_features.half(),
                    labels=labels
                )
                teacher_logits = teacher_out.logits

            # Student forward
            student_out    = student(input_features=input_features, labels=labels)
            student_logits = student_out.logits

            # Distillation loss with masking
            loss, kl_loss, ce_loss = distillation_loss(
                student_logits, teacher_logits, labels, TEMPERATURE, ALPHA
            )

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "kl"  : f"{kl_loss.item():.4f}",
                "ce"  : f"{ce_loss.item():.4f}"
            })

        avg_loss = running_loss / len(train_loader)

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------
        student.eval()
        all_preds, all_refs = [], []

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc="Validation",
                disable=not accelerator.is_local_main_process
            ):
                generated = accelerator.unwrap_model(student).generate(
                    input_features=batch["input_features"],
                    max_new_tokens=225,
                    forced_decoder_ids=processor.get_decoder_prompt_ids(
                        language=LANGUAGE, task=TASK
                    )
                )
                preds = processor.batch_decode(generated, skip_special_tokens=True)
                refs  = processor.batch_decode(
                    torch.where(
                        batch["labels"] == -100,
                        torch.tensor(processor.tokenizer.pad_token_id),
                        batch["labels"]
                    ),
                    skip_special_tokens=True
                )
                all_preds.extend(preds)
                all_refs.extend(refs)

        wer = wer_metric.compute(predictions=all_preds, references=all_refs)
        accelerator.print(
            f"Epoch {epoch} | Loss: {avg_loss:.4f} | WER: {wer*100:.2f}%"
        )

        if wer < best_wer and accelerator.is_local_main_process:
            best_wer  = wer
            save_path = os.path.join(OUTPUT_DIR, "best_model")
            accelerator.unwrap_model(student).save_pretrained(save_path)
            processor.save_pretrained(save_path)
            accelerator.print(f"Best model saved! WER: {wer*100:.2f}%")

        stats.append({
            "epoch": epoch,
            "loss" : round(avg_loss, 4),
            "wer"  : round(wer * 100, 2)
        })

        with open(os.path.join(OUTPUT_DIR, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

    accelerator.print("=" * 60)
    accelerator.print(f"Done! Best WER: {best_wer*100:.2f}%")
    accelerator.print("=" * 60)

if __name__ == "__main__":
    main()