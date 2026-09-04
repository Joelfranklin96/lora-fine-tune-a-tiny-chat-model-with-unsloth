"""End-to-end QLoRA SFT run: load, adapt, train, and generate.

Run with: python train.py
"""

import torch

from model import (
    load_base_model_and_tokenizer,
    count_total_parameters,
    is_model_4bit_quantized,
    ensure_pad_token,
    get_lora_target_modules,
    attach_lora_adapters,
    count_trainable_parameters,
    trainable_fraction,
    build_instruction_examples,
    format_instruction_example,
    format_all_examples,
    build_text_dataset,
    tokenize_text,
    count_tokens,
    build_sft_config,
    build_sft_trainer,
    run_sft_training,
    switch_to_inference_mode,
    build_chat_prompt,
    generate_reply,
)


def main():
    torch.manual_seed(0)

    # Load the 4-bit base model and tokenizer.
    model, tokenizer = load_base_model_and_tokenizer(
        model_name="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        max_seq_length=256,
    )
    total_params = count_total_parameters(model)
    quantized = is_model_4bit_quantized(model)
    print(f"[base] total_params={total_params:,} 4bit={quantized}")

    # Make sure the tokenizer has a pad token.
    tokenizer = ensure_pad_token(tokenizer)
    print(f"[tokenizer] pad_token={tokenizer.pad_token!r}")

    # Attach LoRA adapters to the attention projections.
    target_modules = get_lora_target_modules()
    print(f"[lora] target_modules={target_modules}")
    model = attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=target_modules)
    trainable = count_trainable_parameters(model)
    frac = trainable_fraction(trainable, total_params)
    print(f"[lora] trainable={trainable:,} fraction={frac:.6f}")

    # Build the instruction dataset.
    examples = build_instruction_examples()
    print(f"[data] num_examples={len(examples)}")
    first_text = format_instruction_example(examples[0])
    print(f"[data] formatted[0]={first_text!r}")
    texts = format_all_examples(examples)
    dataset = build_text_dataset(texts)
    print(f"[data] dataset_columns={dataset.column_names} size={len(dataset)}")

    # Sanity-check tokenization of one example.
    ids = tokenize_text(tokenizer, texts[0])
    print(f"[data] tokens[0]={count_tokens(ids)}")

    # Run a short, memory-efficient SFT loop.
    training_args = build_sft_config(
        output_dir="./sft_out", max_steps=5, learning_rate=2e-4, max_seq_length=256
    )
    trainer = build_sft_trainer(model, tokenizer, dataset, training_args)
    final_loss = run_sft_training(trainer)
    print(f"[train] final_loss={final_loss:.4f}")

    # Switch to fast inference and generate a reply.
    switch_to_inference_mode(model)
    prompt = build_chat_prompt(tokenizer, "Say hello in one short sentence.")
    reply = generate_reply(model, tokenizer, prompt, max_new_tokens=32)
    print(f"[gen] reply={reply!r}")

    # Final sanity checks on the run.
    passed = (
        total_params > 0
        and trainable > 0
        and trainable < total_params
        and 0.0 < frac < 0.1
        and isinstance(reply, str)
        and len(reply) > 0
        and final_loss == final_loss  # finite (not NaN)
    )
    print({"passed": bool(passed)})
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
