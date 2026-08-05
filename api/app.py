import os
import threading

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Multi-Domain QA LLM Engine")

# Global thread lock to ensure atomic GPU PEFT operations and prevent race conditions
gpu_lock = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
ADAPTERS_DIR = os.path.join(ROOT_DIR, "adapters")

ALL_POSSIBLE_DOMAINS = ["us_law", "agriculture", "fitness", "sql", "python", "cpp", "medical"]

DOMAIN_DESCRIPTIONS = {
    "us_law": "US legal principles and contract law",
    "agriculture": "agriculture, botany, and soil management",
    "fitness": "fitness, exercise form, and strength training",
    "sql": "SQL database querying and database design",
    "python": "Python programming and software engineering",
    "cpp": "C++ programming, memory management, and OOP",
    "medical": "medical diagnostics, clinical symptoms, and healthcare guidance"
}

DOMAIN_EXEMPLARS = {
    "us_law": [
        "What are the legal requirements for a valid binding contract?",
        "What is the statute of limitations for personal injury claims?",
        "Can a landlord evict a tenant without prior written notice?",
        "What constitutes copyright infringement under federal law?"
    ],
    "agriculture": [
        "How do I identify and treat fungal leaf spot on tomato plants?",
        "What is the optimal soil pH range for growing blueberries?",
        "How to apply nitrogen fertilizer to maize during early growth?",
        "What are the best crop rotation strategies to prevent soil degradation?"
    ],
    "fitness": [
        "How to maintain proper bar path and form during heavy conventional deadlifts?",
        "What is the optimal rep range for muscle hypertrophy versus strength?",
        "How much daily cardio should I perform while maintaining muscle mass?",
        "How to structure a Push-Pull-Legs workout split for beginners?"
    ],
    "sql": [
        "How to write an SQL query using LEFT JOIN and GROUP BY with aggregation?",
        "What is the difference between WHERE and HAVING clauses in SQL?",
        "How to create an index to optimize slow database queries?",
        "What are database normalization rules from 1NF to 3NF?"
    ],
    "python": [
        "How to write a Python script using list comprehensions and generators?",
        "What is the difference between deepcopy and shallow copy in Python?",
        "How to handle exceptions using try except blocks properly?",
        "How to parse JSON data using the standard library in Python?"
    ],
    "cpp": [
        "How to implement an Object-Oriented C++ class using RAII and smart pointers?",
        "What is the difference between std::move and std::forward in C++11?",
        "How to avoid memory leaks when using dynamic allocation with new and delete?",
        "How to use STL containers like std::vector and std::unordered_map?"
    ],
    "medical": [
        "What are the early warning signs and symptoms of Type 2 Diabetes?",
        "How is high blood pressure diagnosed and managed clinically?",
        "What are the common side effects and risks of long-term NSAID use?",
        "What causes acute chest pain and when is it a medical emergency?"
    ]
}

print("\n" + "="*60)
print("[+] FAILSAFE INSPECTION: CHECKING LOCAL ADAPTER DIRECTORIES")
print("="*60)

AVAILABLE_DOMAINS = []
for domain in ALL_POSSIBLE_DOMAINS:
    adapter_path = os.path.join(ADAPTERS_DIR, f"qwen-{domain}-lora")
    config_file = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(config_file):
        AVAILABLE_DOMAINS.append(domain)
        print(f"  ✓ ACTIVE ADAPTER: [{domain.upper()}] -> '{adapter_path}'")
    else:
        print(f"  ✗ MISSING ADAPTER: [{domain.upper()}] (Skipped by router)")

print(f"\n[+] Total Available Domain Adapters: {len(AVAILABLE_DOMAINS)} / {len(ALL_POSSIBLE_DOMAINS)}")
print("="*60 + "\n")

print("[+] Initializing CPU SentenceTransformer router ('all-MiniLM-L6-v2')...")
router_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

AVAILABLE_CENTROIDS = []
if AVAILABLE_DOMAINS:
    for domain in AVAILABLE_DOMAINS:
        exemplar_vectors = router_model.encode(DOMAIN_EXEMPLARS[domain], convert_to_tensor=True)
        centroid = torch.mean(exemplar_vectors, dim=0, keepdim=True)
        AVAILABLE_CENTROIDS.append(centroid)

    AVAILABLE_CENTROIDS_TENSOR = torch.cat(AVAILABLE_CENTROIDS, dim=0)

def classify_and_weight_intent(prompt: str, blend_threshold: float = 0.20):
    if not AVAILABLE_DOMAINS:
        return [], [], {}

    prompt_vector = router_model.encode(prompt, convert_to_tensor=True)
    similarities = util.cos_sim(prompt_vector, AVAILABLE_CENTROIDS_TENSOR)[0]
    
    raw_scores = {domain: round(float(sim.item()), 4) for domain, sim in zip(AVAILABLE_DOMAINS, similarities)}
    
    temperature = 0.15
    probs = torch.softmax(similarities / temperature, dim=0)
    
    detected_pairs = []
    for idx, prob in enumerate(probs):
        w = float(prob.item())
        if w >= blend_threshold:
            detected_pairs.append((AVAILABLE_DOMAINS[idx], w))
            
    if not detected_pairs:
        top_idx = int(torch.argmax(similarities).item())
        detected_pairs = [(AVAILABLE_DOMAINS[top_idx], 1.0)]
        
    detected_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Cap at TOP-2 max to prevent low-rank weight collision
    top_pairs = detected_pairs[:2]
    
    active_adapters = [p[0] for p in top_pairs]
    raw_weights = [p[1] for p in top_pairs]
    
    tot = sum(raw_weights)
    norm_weights = [w / tot for w in raw_weights]
    
    return active_adapters, norm_weights, raw_scores

print(f"[+] Loading Tokenizer & Foundation Model ({BASE_MODEL_ID}) onto CUDA...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

IM_END_ID = tokenizer.convert_tokens_to_ids("<|im_end|>")
EOS_ID = tokenizer.eos_token_id
STOP_IDS = list(set(filter(None, [EOS_ID, IM_END_ID])))

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to("cuda")

peft_model = None
if AVAILABLE_DOMAINS:
    print("[+] Registering PeftModel adapters in VRAM...")
    first_domain = AVAILABLE_DOMAINS[0]
    first_path = os.path.join(ADAPTERS_DIR, f"qwen-{first_domain}-lora")
    
    peft_model = PeftModel.from_pretrained(
        base_model,
        first_path,
        adapter_name=first_domain
    )
    
    for domain in AVAILABLE_DOMAINS[1:]:
        adapter_path = os.path.join(ADAPTERS_DIR, f"qwen-{domain}-lora")
        peft_model.load_adapter(adapter_path, adapter_name=domain)
        print(f"  ✓ Registered '{domain}' in PEFT container.")

print("[+] System initialization finished successfully.\n")

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.4

@app.get("/api/status")
def status_endpoint():
    return {
        "status": "ready",
        "available_domains": AVAILABLE_DOMAINS,
        "total_available": len(AVAILABLE_DOMAINS)
    }

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    try:
        with gpu_lock:
            if not AVAILABLE_DOMAINS or peft_model is None:
                routing_mode = "Base Model (Zero Adapters Loaded)"
                active_adapters = []
                weights_dict = {}
                raw_scores = {}
                active_model = base_model
                system_instruction = "You are a helpful multi-domain expert AI assistant."
            else:
                active_adapters, weights, raw_scores = classify_and_weight_intent(request.prompt)
                weights_dict = {domain: round(w, 4) for domain, w in zip(active_adapters, weights)}
                
                domain_specs = [DOMAIN_DESCRIPTIONS[d] for d in active_adapters]
                system_instruction = f"You are an expert AI assistant specializing in {', '.join(domain_specs)}. Provide a clear, structured, and complete answer."
                
                if len(active_adapters) == 1:
                    selected_domain = active_adapters[0]
                    peft_model.set_adapter(selected_domain)
                    routing_mode = f"Hard Switch -> [{selected_domain.upper()}]"
                else:
                    scaled_weights = [w * 0.75 for w in weights]
                    peft_model.add_weighted_adapter(
                        adapters=active_adapters,
                        weights=scaled_weights,
                        adapter_name="dynamic_blend",
                        combination_type="linear"
                    )
                    peft_model.set_adapter("dynamic_blend")
                    blend_summary = " + ".join([f"{d.upper()} ({int(w*100)}%)" for d, w in zip(active_adapters, weights)])
                    routing_mode = f"Dynamic Blend -> {blend_summary}"
                
                active_model = peft_model

            formatted_prompt = (
                f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
                f"<|im_start|>user\n{request.prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = active_model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=0.85,
                    repetition_penalty=1.15,
                    do_sample=True if request.temperature > 0 else False,
                    eos_token_id=STOP_IDS,
                    pad_token_id=tokenizer.pad_token_id,
                    stop_strings=["<|im_end|>"],
                    tokenizer=tokenizer
                )
                
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            raw_response = tokenizer.decode(generated_tokens, skip_special_tokens=False)
            
            clean_response = raw_response.split("<|im_end|>")[0].split("<|endoftext|>")[0]
            if ".WARNING:" in clean_response:
                clean_response = clean_response.split(".WARNING:")[0]
            clean_response = clean_response.strip()
            
            if len(active_adapters) > 1 and peft_model is not None:
                peft_model.delete_adapter("dynamic_blend")

            return {
                "status": "success",
                "routing_mode": routing_mode,
                "active_adapters": active_adapters,
                "adapter_weights": weights_dict,
                "domain_scores": raw_scores,
                "response": clean_response
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)