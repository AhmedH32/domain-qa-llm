import os

import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Qwen2.5 Medical Assistant Local API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str = Field(..., example="What are the symptoms of Type 2 Diabetes?")
    system_prompt: str = Field(
        default="You are a knowledgeable and empathetic medical AI assistant. Provide accurate, clear, and professional information regarding health, diseases, and medical conditions."
    )
    max_tokens: int = Field(default=200, ge=1, le=512)
    temperature: float = Field(default=0.4, ge=0.0, le=1.0)

class ChatResponse(BaseModel):
    response: str
    device: str
    adapter_loaded: bool

model_stack = {}

@app.on_event("startup")
def load_model_stack():
    config_path = "configs/train_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        base_name = config["model"]["base_model_name"]
        adapter_path = config["training"]["output_dir"]
    else:
        base_name = "Qwen/Qwen2.5-0.5B"
        adapter_path = "./adapters/qwen-medquad-lora"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Starting local inference backend on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True
    )

    adapter_loaded = False
    if os.path.exists(adapter_path):
        print(f"[+] Attaching LoRA adapter from: '{adapter_path}'...")
        final_model = PeftModel.from_pretrained(base_model, adapter_path)
        final_model.eval()
        adapter_loaded = True
    else:
        print(f"[-] Warning: Adapter path '{adapter_path}' not found. Serving raw base model.")
        final_model = base_model

    model_stack["model"] = final_model
    model_stack["tokenizer"] = tokenizer
    model_stack["device"] = device
    model_stack["adapter_loaded"] = adapter_loaded

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": model_stack.get("device", "unknown"),
        "adapter_loaded": model_stack.get("adapter_loaded", False)
    }

@app.post("/api/chat", response_model=ChatResponse)
def generate_chat_response(request: ChatRequest):
    if "model" not in model_stack:
        raise HTTPException(status_code=503, detail="Model stack not initialized.")

    model = model_stack["model"]
    tokenizer = model_stack["tokenizer"]
    device = model_stack["device"]

    messages = [
        {"role": "system", "content": request.system_prompt},
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
        device=device,
        adapter_loaded=model_stack["adapter_loaded"]
    )