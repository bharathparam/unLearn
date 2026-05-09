"""
Simple test to verify model loads and generates text.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading Qwen2.5-1.5B-Instruct...")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with minimal memory settings
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Simple generation test
prompt = "The capital of France is"
print(f"\nPrompt: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Result: {result}")

print("\nTest completed successfully!")
