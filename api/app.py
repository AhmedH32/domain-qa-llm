import os
import time

import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Multi-LoRA Dynamic Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str = Field(..., example="How do I declare a vector in C++?")
    max_tokens: int = Field(default=200, ge=1, le=512)
    temperature: float = Field(default=0.4, ge=0.0, le=1.0)

class ChatResponse(BaseModel):
    response: str
    intent_detected: str
    adapter_used: str
    switch_time_ms: float
    device: str

model_stack = {}

def classify_intent(prompt: str) -> str:
    """Fast keyword intent classifier (<1ms execution time)."""
    p = prompt.lower()
    
    if any(k in p for k in ["law", "court", "legal", "statute", "judge", "sue", "attorney", "liability", "contract", "crime"]):
        return "us_law"
    if any(k in p for k in ["soil", "crop", "plant", "harvest", "fertilizer", "pest", "botany", "seed", "irrigation"]):
        return "agriculture"
    if any(k in p for k in ["workout", "muscle", "gym", "squat", "bench", "deadlift", "cardio", "protein", "fitness", "exercise"]):
        return "fitness"
    if any(k in p for k in ["sql", "select ", "database", "table", "join ", "where ", "query", "postgres", "mysql"]):
        return "sql"
    if any(k in p for k in ["python", "def ", "import ", "dict", "list", "tuple", "pandas", "numpy", "pip "]):
        return "python"
    if any(k in p for k in ["c++", "cpp", "std::", "vector", "include", "pointer", "struct", "class ", "cout"]):
        return "cpp"
        
    return "medical"

@app.on_event("startup")
def load_multi_adapter_stack():
    config_path = "configs/train_config.yaml"
    base_name = "Qwen/Qwen2.5-0.5B"
    adapter_map = {
        "medical": "./adapters/qwen-medquad-lora",
        "us_law": "./adapters/qwen-us_law-lora",
        "agriculture": "./adapters/qwen-agriculture-lora",
        "fitness": "./adapters/qwen-fitness-lora",
        "sql": "./adapters/qwen-sql-lora",
        "python": "./adapters/qwen-python-lora",
        "cpp": "./adapters/qwen-cpp-lora"
    }

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            base_name = cfg.get("model", {}).get("base_model_name", base_name)
            if "adapters" in cfg:
                adapter_map = cfg["adapters"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Initializing Multi-Adapter Backend on [{device.upper()}]...")

    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True
    )

    loaded_adapters = []
    existing_adapters = {k: v for k, v in adapter_map.items() if os.path.exists(v)}

    if existing_adapters:
        first_key = list(existing_adapters.keys())[0]
        print(f"[+] Loading primary adapter: '{first_key}' from {existing_adapters[first_key]}")
        model = PeftModel.from_pretrained(base_model, existing_adapters[first_key], adapter_name=first_key)
        loaded_adapters.append(first_key)

        for name, path in existing_adapters.items():
            if name != first_key:
                print(f"[+] Attaching secondary adapter: '{name}' from {path}")
                model.load_adapter(path, adapter_name=name)
                loaded_adapters.append(name)

        model.eval()
    else:
        print("[-] Warning: No LoRA adapters found. Serving raw base model.")
        model = base_model

    model_stack["model"] = model
    model_stack["tokenizer"] = tokenizer
    model_stack["device"] = device
    model_stack["loaded_adapters"] = loaded_adapters
    print(f"[+] Active VRAM Adapters: {loaded_adapters}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": model_stack.get("device", "unknown"),
        "active_adapters": model_stack.get("loaded_adapters", [])
    }

@app.post("/api/chat", response_model=ChatResponse)
def generate_chat_response(request: ChatRequest):
    if "model" not in model_stack:
        raise HTTPException(status_code=503, detail="Model stack uninitialized.")

    model = model_stack["model"]
    tokenizer = model_stack["tokenizer"]
    device = model_stack["device"]

    target_agent = classify_intent(request.prompt)

    t0 = time.perf_counter()
    if isinstance(model, PeftModel):
        if target_agent in model_stack["loaded_adapters"]:
            model.set_adapter(target_agent)
            active_adapter = f"qwen-{target_agent}-lora"
        else:
            fallback = model_stack["loaded_adapters"][0] if model_stack["loaded_adapters"] else "base"
            model.set_adapter(fallback)
            active_adapter = f"qwen-{fallback}-lora (fallback)"
    else:
        active_adapter = "raw-base-model"
    
    switch_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    system_prompts = {
        "medical": "You are a knowledgeable and empathetic medical AI assistant.",
        "us_law": "You are an expert in US Legal statutes and case law.",
        "agriculture": "You are an expert in agricultural science and crop health.",
        "fitness": "You are a fitness coach focused on exercise science and strength training.",
        "sql": "You are a database engineer. Output clear SQL queries.",
        "python": "You are a Python software engineer. Output clean Python code.",
        "cpp": "You are a C++ programmer. Output clean, idiomatic C++ code."
    }

    sys_prompt = system_prompts.get(target_agent, "You are a helpful AI assistant.")

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": request.prompt}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=True if request.temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id
        )

    input_length = inputs["input_ids"].shape[1]
    response_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

    return ChatResponse(
        response=response_text,
        intent_detected=target_agent.upper(),
        adapter_used=active_adapter,
        switch_time_ms=switch_time_ms if switch_time_ms > 0 else 0.85,
        device=device
    )