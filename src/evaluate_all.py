import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
ADAPTERS_DIR = os.path.join(ROOT_DIR, "adapters")

TEST_CASES = [
    {
        "domain": "us_law",
        "system": "You are an expert in US Law.",
        "query": "What are the essential elements required to prove a breach of contract under US law?"
    },
    {
        "domain": "agriculture",
        "system": "You are an agricultural and botany expert.",
        "query": "How do you identify and treat powdery mildew on tomato leaves?"
    },
    {
        "domain": "fitness",
        "system": "You are a professional fitness coach specializing in heavy compound lifts and cardio routines.",
        "query": "What are the key technical cues for performing a heavy barbell back squat safely?"
    },
    {
        "domain": "sql",
        "system": "You are a database engineer.",
        "query": "Write an SQL query to find the top 5 highest-paid employees from an 'employees' table."
    },
    {
        "domain": "python",
        "system": "You are a Python software engineer.",
        "query": "Write a Python function that takes a list of integers and returns only the prime numbers."
    },
    {
        "domain": "cpp",
        "system": "You are a C++ programmer. Provide standard library syntax, clear OOP code, and robust logic.",
        "query": "Write a simple C++ class named 'BankAccount' with methods to deposit, withdraw, and check balance."
    }
]

def format_prompt(system_prompt, user_query):
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def clean_output(full_text, prompt_text):
    # Strip prompt prefix
    response = full_text[len(prompt_text):]
    # Stop cleanly at ChatML end token or EOS
    for stop_seq in ["<|im_end|>", "<|endoftext|>"]:
        if stop_seq in response:
            response = response.split(stop_seq)[0]
    return response.strip()

def run_full_evaluation():
    print("\n" + "="*90)
    print("        MULTI-AGENT LORA BENCHMARK EVALUATION (CLEAN CHATML CUTOFF)")
    print("="*90)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"[+] Loading Base Model: {MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to("cuda")

    for test in TEST_CASES:
        domain = test["domain"]
        system_prompt = test["system"]
        query = test["query"]
        adapter_path = os.path.join(ADAPTERS_DIR, f"qwen-{domain}-lora")

        print(f"\n[{domain.upper()} DOMAIN TEST]")
        print(f"Query: \"{query}\"")
        print("-" * 90)

        prompt_formatted = format_prompt(system_prompt, query)
        inputs = tokenizer(prompt_formatted, return_tensors="pt").to("cuda")

        # 1. Base Model Generation
        with torch.no_grad():
            raw_out = base_model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,  # Deterministic greedy decoding for clean benchmark
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        base_text = tokenizer.decode(raw_out[0], skip_special_tokens=False)
        cleaned_base = clean_output(base_text, prompt_formatted)

        print(f"| RAW BASE MODEL (UN-TUNED) |\n{cleaned_base}")
        print("-" * 90)

        # 2. Fine-Tuned Adapter Generation
        if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            peft_model = PeftModel.from_pretrained(base_model, adapter_path)
            with torch.no_grad():
                ft_out = peft_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            ft_text = tokenizer.decode(ft_out[0], skip_special_tokens=False)
            cleaned_ft = clean_output(ft_text, prompt_formatted)
            print(f"| FINE-TUNED [{domain.upper()}] ADAPTER |\n{cleaned_ft}")
            
            # Unload adapter to keep base model clean for next loop
            del peft_model
            torch.cuda.empty_cache()
        else:
            print(f"| FINE-TUNED [{domain.upper()}] ADAPTER | [ADAPTER NOT FOUND ON DISK]")

        print("=" * 90)

if __name__ == "__main__":
    run_full_evaluation()