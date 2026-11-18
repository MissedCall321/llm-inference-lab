"""Run one short Qwen conversation on the GPU to validate the development environment."""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="请用中文简短介绍你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Run scripts/check_gpu.py first.")

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_id}; the first run downloads the model into the cache volume.")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda").eval()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = output[0, inputs["input_ids"].shape[1]:]
    print("\nModel response:")
    print(tokenizer.decode(generated, skip_special_tokens=True))
    print(f"\nGenerated tokens: {generated.numel()}")
    print(f"Peak PyTorch allocated VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
