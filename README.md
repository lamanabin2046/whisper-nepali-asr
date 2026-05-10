# Nepali ASR with Whisper

Whisper adaptation for low-resource Nepali ASR: fine-tuning, LoRA, and knowledge distillation across model sizes.

## Results

| Model | Method | Trainable Params | WER |
|-------|--------|-----------------|-----|
| Whisper Tiny | Fine-tune (uncleaned) | 39M | 31.63% |
| Whisper Base | Fine-tune (uncleaned) | 74M | 21.26% |
| Whisper Small | Fine-tune (uncleaned) | 244M | 13.46% |
| Whisper Tiny | Fine-tune (cleaned) | 39M | 35.30% |
| Whisper Small | Fine-tune (cleaned) | 244M | 14.52% |
| Whisper Small | LoRA | 3.5M | 22.54% |
| Whisper Large v3 | LoRA | 15.7M | TBD |
