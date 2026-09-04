"""QLoRA supervised fine-tuning pipeline.

Fine-tunes a 4-bit quantized chat model with LoRA adapters using Unsloth,
Hugging Face TRL, and bitsandbytes.
"""

# Unsloth must be imported before transformers/trl so its patches apply.
from unsloth import FastLanguageModel

import bitsandbytes as bnb
import datasets
import torch
from trl import SFTConfig, SFTTrainer


def load_base_model_and_tokenizer(model_name="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit", max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,  # auto-detect bfloat16/float16 based on GPU support
    )
    return model, tokenizer


def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    return sum(p.numel() for p in model.parameters())


def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    return any(isinstance(m, bnb.nn.Linear4bit) for m in model.modules())


def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_lora_target_modules():
    """Return the attention projection module name suffixes to adapt with LoRA."""
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    if target_modules is None:
        target_modules = get_lora_target_modules()
    return FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
    )


def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trainable_fraction(trainable_count, total_count):
    """Return the fraction of parameters that are trainable."""
    return float(trainable_count / total_count)


def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    return [
        {
            "instruction": "When do new hires get their laptop?",
            "response": "Your laptop is handed out during day-one orientation.",
        },
        {
            "instruction": "Who should I contact if I have a payroll question?",
            "response": "Reach out to the HR helpdesk for anything payroll related.",
        },
        {
            "instruction": "How long is the onboarding program?",
            "response": "Onboarding runs for the first two weeks after you join.",
        },
        {
            "instruction": "Where can I find the employee handbook?",
            "response": "The employee handbook is available on the company intranet.",
        },
    ]


def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    return "### Instruction:\n{}\n\n### Response:\n{}".format(
        example["instruction"], example["response"]
    )


def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    return [format_instruction_example(example) for example in examples]


def build_text_dataset(texts):
    """Wrap a list of training strings in a Hugging Face Dataset with a 'text' column."""
    return datasets.Dataset.from_dict({"text": texts})


def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""
    return tokenizer.encode(text)


def count_tokens(input_ids):
    """Return the number of tokens in a tokenized example."""
    return len(input_ids)


def build_sft_config(output_dir="./sft_out", max_steps=5, learning_rate=2e-4, max_seq_length=256):
    """Return a lightweight SFTConfig for a short, memory-efficient SFT run."""
    bf16 = torch.cuda.is_bf16_supported()
    return SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        learning_rate=learning_rate,
        bf16=bf16,
        fp16=not bf16,
        logging_steps=1,
        optim="adamw_8bit",
        output_dir=output_dir,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
    )


def build_sft_trainer(model, tokenizer, dataset, training_args):
    """Construct a trl SFTTrainer over dataset['text'], ready to .train()."""
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )


def run_sft_training(trainer):
    """Run the SFT loop and return the final training loss as a float."""
    output = trainer.train()
    return float(output.metrics["train_loss"])


def switch_to_inference_mode(model):
    """Switch the QLoRA-tuned model into Unsloth's fast inference mode and return it."""
    FastLanguageModel.for_inference(model)
    return model


def build_chat_prompt(tokenizer, instruction):
    """Return a chat-template prompt string ready for assistant generation."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_reply(model, tokenizer, prompt, max_new_tokens=32):
    """Greedy-generate a reply for `prompt` and return the decoded text."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    new_token_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True)
