import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are an expert customer support specialist. "
    "Provide clear, professional, and accurate solutions to user inquiries."
)

TEST_QUERIES = [
    "I want to track my order #98234, but the page is giving a 404 error.",
    "Can I get a refund for a subscription I canceled yesterday?",
    "How do I change the shipping address for an item that hasn't shipped yet?",
    "My discount code isn't applying at checkout."
]

def generate_response(model, tokenizer, query, device):
    """
    Formats query into ChatML prompt, tokenizes, and returns generated assistant response.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode only generated response tokens (exclude prompt input)
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

def run_evaluation(config_path="configs/train_config.yaml"):
    """
    Executes comparative evaluation between un-tuned base model and LoRA adapter.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_model_name = config["model"]["base_model_name"]
    adapter_dir = config["training"]["output_dir"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Base Model & Tokenizer
    print(f"[+] Loading Base Model: '{base_model_name}' on [{device}]...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        trust_remote_code=True
    )

    # 2. Run Baseline Inference
    print("\n[+] Generating Baseline Model Responses...")
    base_results = []
    for query in TEST_QUERIES:
        res = generate_response(base_model, tokenizer, query, device)
        base_results.append(res)

    # 3. Attach LoRA Adapter to Base Model
    print(f"\n[+] Injecting LoRA Adapter from: '{adapter_dir}'...")
    ft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    ft_model.eval()

    # 4. Run Fine-Tuned Inference
    print("\n[+] Generating Fine-Tuned Model Responses...")
    ft_results = []
    for query in TEST_QUERIES:
        res = generate_response(ft_model, tokenizer, query, device)
        ft_results.append(res)

    # 5. Output Comparative Delta Table
    print("\n" + "=" * 90)
    print("                     EVALUATION & DELTA ANALYSIS TABLE")
    print("=" * 90)

    for idx, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[TEST QUERY {idx}]: {query}")
        print("-" * 90)
        print(f"| UN-TUNED BASELINE   | {base_results[idx-1]}")
        print("-" * 90)
        print(f"| FINE-TUNED ADAPTER  | {ft_results[idx-1]}")
        print("=" * 90)

if __name__ == "__main__":
    run_evaluation()