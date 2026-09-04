# QLoRA SFT Pipeline

End-to-end supervised fine-tuning (SFT) pipeline that adapts a 4-bit quantized
chat model to a custom instruction dataset using **QLoRA** — LoRA adapters
trained on top of a frozen, 4-bit quantized base model. Built with
[Unsloth](https://github.com/unslothai/unsloth), Hugging Face
[TRL](https://github.com/huggingface/trl), and
[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes).

## What it does

- Loads `Qwen2.5-0.5B-Instruct` in 4-bit NF4 quantization via Unsloth's
  `FastLanguageModel`, verifying quantization by inspecting the model for
  `bitsandbytes` 4-bit linear layers.
- Attaches LoRA adapters (rank 8, alpha 16) to the attention projections
  (`q_proj`, `k_proj`, `v_proj`, `o_proj`), training well under 1% of the
  model's parameters while the quantized base stays frozen.
- Builds an instruction/response dataset, formats it with explicit role
  markers, and wraps it in a Hugging Face `Dataset`.
- Runs memory-efficient SFT with TRL's `SFTTrainer` (8-bit AdamW,
  bf16/fp16 autodetection based on GPU support).
- Switches the tuned model into Unsloth's fast inference mode and generates
  replies through the tokenizer's chat template.

## Pipeline

```
load 4-bit base model ──► attach LoRA adapters ──► build & format dataset
        ──► SFT training (TRL) ──► fast inference & generation
```

## Quickstart

Requires an NVIDIA GPU (CUDA).

```bash
pip install -r requirements.txt
python train.py
```

The script logs each stage — parameter counts, trainable fraction,
tokenization stats, training loss — and ends with a sanity check that the
run produced a finite loss and a non-empty generation, printing `PASS` on
success.

## Project structure

| File | Purpose |
|---|---|
| `model.py` | Core pipeline: model loading, quantization checks, LoRA attachment, dataset construction, training, and inference utilities |
| `train.py` | Orchestrates the full run end to end |

## Key techniques

- **4-bit quantization (NF4)** cuts base-model memory roughly 4x compared to
  fp16, making fine-tuning feasible on consumer GPUs.
- **LoRA adapters** train small low-rank matrices injected into the attention
  layers instead of the full weights — combined with the 4-bit quantized base,
  this is QLoRA. The trainable fraction is logged at runtime.
- **8-bit AdamW** further reduces optimizer-state memory during training.
