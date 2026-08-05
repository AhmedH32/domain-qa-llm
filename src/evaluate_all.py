import os

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "train_config.yaml")

TEST_SUITE = [
    {
        "domain": "medical",
        "system": "You are a knowledgeable and empathetic medical AI assistant.",
        "query": "What are the early warning signs and symptoms of Type 2 Diabetes?",
        "adapter_key": "medical"
    },
    {
        "domain": "us_law",
        "system": "You are an expert in US Legal statutes and case law.",
        "query": "What are the essential elements required to prove a breach of contract under US law?",
        "adapter_key": "us_law"
    },
    {
        "domain": "agriculture",
        "system": "You are an expert in agricultural science and crop health.",
        "query": "How do you identify and treat powdery mildew on tomato leaves?",
        "adapter_key": "agriculture"
    },
    {
        "domain": "fitness",
        "system": "You are a fitness coach focused on exercise science and strength training.",
        "query": "What are the key technical cues for performing a heavy barbell back squat safely?",
        "adapter_key": "fitness"
    },
    {
        "domain": "sql",
        "system": "You are a database engineer. Output clear SQL queries.",
        "query": "Write an SQL query to find the top 5 highest-paid employees from an 'employees' table.",
        "adapter_key": "sql"
    },
    {
        "domain": "python",
        "system": "You are a Python software engineer. Output clean Python code.",
        "query": "Write a Python function that takes a list of integers and returns only the prime numbers.",
        "adapter_key": "python"
    },
    {
        "domain": "cpp",
        "system": "You are a C++ programmer. Output clean, idiomatic C++ code.",
        "query": "Write a simple C++ class named 'BankAccount' with methods to deposit, withdraw, and check balance.",
        "adapter_key": "cpp"
    }
]

def generate_response(model, tokenizer, query, system_prompt, device):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    input_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

def run_full_evaluation():
    base_name = "Qwen/Qwen2.5-0.5B"
    adapter_map = {
        "medical": os.path.join(ROOT_DIR, "adapters", "qwen-medquad-lora"),
        "us_law": os.path.join(ROOT_DIR, "adapters", "qwen-us_law-lora"),
        "agriculture": os.path.join(ROOT_DIR, "adapters", "qwen-agriculture-lora"),
        "fitness": os.path.join(ROOT_DIR, "adapters", "qwen-fitness-lora"),
        "sql": os.path.join(ROOT_DIR, "adapters", "qwen-sql-lora"),
        "python": os.path.join(ROOT_DIR, "adapters", "qwen-python-lora"),
        "cpp": os.path.join(ROOT_DIR, "adapters", "qwen-cpp-lora")
    }

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
            base_name = cfg.get("model", {}).get("base_model_name", base_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Loading Base Foundation Model: '{base_name}' on [{device.upper()}]...")

    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True
    )

    # Dynamic primary initialization using the first available adapter on disk
    existing_adapters = {k: v for k, v in adapter_map.items() if os.path.exists(v)}
    if not existing_adapters:
        print("[-] Warning: No adapters found on disk to evaluate.")
        return

    first_key = list(existing_adapters.keys())[0]
    print(f"[+] Initializing PEFT model with primary adapter: '{first_key}'...")
    model = PeftModel.from_pretrained(base_model, existing_adapters[first_key], adapter_name=first_key)

    for name, path in existing_adapters.items():
        if name != first_key:
            print(f"[+] Mounting secondary adapter: '{name}'...")
            model.load_adapter(path, adapter_name=name)

    model.eval()

    print("\n" + "=" * 90)
    print("        MULTI-AGENT LORA BENCHMARK EVALUATION (RAW BASE VS. FINE-TUNED)")
    print("=" * 90)

    for test in TEST_SUITE:
        domain = test["domain"].upper()
        adapter_key = test["adapter_key"]
        
        print(f"\n[{domain} DOMAIN TEST]")
        print(f"Query: \"{test['query']}\"")
        print("-" * 90)

        with model.disable_adapter():
            raw_response = generate_response(model, tokenizer, test["query"], test["system"], device)
        print(f"| RAW BASE MODEL (UN-TUNED) |\n{raw_response}")
        print("-" * 90)

        if adapter_key in existing_adapters:
            model.set_adapter(adapter_key)
            ft_response = generate_response(model, tokenizer, test["query"], test["system"], device)
            print(f"| FINE-TUNED [{domain}] ADAPTER |\n{ft_response}")
        else:
            print(f"| FINE-TUNED [{domain}] ADAPTER | [ADAPTER NOT FOUND ON DISK]")
            
        print("=" * 90)

if __name__ == "__main__":
    run_full_evaluation()