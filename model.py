"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
from unsloth import FastLanguageModel
def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    # TODO: call FastLanguageModel.from_pretrained with 4-bit loading and return (model, tokenizer)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,  # auto-detect (bfloat16/float16) based on GPU
    )
    return model, tokenizer

# Step 2 - count_total_parameters
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    
    count = sum([p.numel() for p in model.parameters()])
    return count

# Step 3 - is_model_4bit_quantized
import bitsandbytes as bnb
def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    
    return any(isinstance(m, bnb.nn.Linear4bit) for m in model.modules())

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""

    target_submodules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    return target_submodules

# Step 6 - attach_lora_adapters
def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    
    if target_modules is None:
        target_modules = get_lora_target_modules()
    
    return FastLanguageModel.get_peft_model(model, r=r, target_modules=target_modules, lora_alpha=lora_alpha, lora_dropout=0, bias='none')

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    
    return float(trainable_count/total_count)

# Step 9 - build_instruction_examples
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

# Step 10 - format_instruction_example
def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    
    return "### Instruction:\n{}\n\n### Response:\n{}".format(example['instruction'], example['response'])

# Step 11 - format_all_examples
def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    
    training_strings = [format_instruction_example(example) for example in examples]
    return training_strings

# Step 12 - build_text_dataset
import datasets
def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    
    return datasets.Dataset.from_dict({
    "text": texts
})

# Step 13 - tokenize_text (not yet solved)
# TODO: implement

# Step 14 - count_tokens (not yet solved)
# TODO: implement

# Step 15 - build_training_arguments (not yet solved)
# TODO: implement

# Step 16 - build_sft_trainer (not yet solved)
# TODO: implement

# Step 17 - run_sft_training (not yet solved)
# TODO: implement

# Step 18 - switch_to_inference_mode (not yet solved)
# TODO: implement

# Step 19 - build_chat_prompt (not yet solved)
# TODO: implement

# Step 20 - generate_reply (not yet solved)
# TODO: implement

