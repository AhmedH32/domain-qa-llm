import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are a knowledgeable and empathetic medical AI assistant. "
    "Provide accurate, clear, and professional information regarding health, diseases, and medical conditions."
)

TEST_QUERIES = [
    "What are the early warning signs and symptoms of Type 2 Diabetes?",
    "How is hypertension diagnosed and managed lifestyle-wise?",
    "What are the common side effects of amoxicillin?",
    "What treatments are available for acute migraine headaches?"
]

def generate_response(model, tokenizer, query, device):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
    
    # Format ChatML structure
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    input_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

def run_evaluation(config_path="configs/train_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_model_name = config["model"]["base_model_name"]
    adapter_dir = config["training"]["output_dir"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[+] Loading Raw Foundation Model: '{base_model_name}' on [{device}]...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        trust_remote_code=True
    )

    print("\n[+] Generating Un-Tuned Base Model Responses (Raw Continuation)...")
    base_results = [generate_response(base_model, tokenizer, q, device) for q in TEST_QUERIES]

    print(f"\n[+] Injecting Medical LoRA Adapter from: '{adapter_dir}'...")
    ft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    ft_model.eval()

    print("\n[+] Generating Fine-Tuned Adapter Responses...")
    ft_results = [generate_response(ft_model, tokenizer, q, device) for q in TEST_QUERIES]

    print("\n" + "=" * 90)
    print("           FOUNDATION BASE vs. FINE-TUNED ADAPTER DELTA TABLE")
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