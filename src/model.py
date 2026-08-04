import torch
import yaml
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_model_and_tokenizer(config_path="configs/train_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["base_model_name"]
    lora_cfg = config["lora"]

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Lock single GPU mapping
    device_map = {"": 0} if torch.cuda.is_available() else "cpu"

    # Load Foundation Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if config["training"].get("fp16", True) else torch.float32,
        device_map=device_map,
        trust_remote_code=True
    )

    # Configure LoRA Adapters
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM
    )

    return model, tokenizer, peft_config

if __name__ == "__main__":
    model, tokenizer, peft_cfg = setup_model_and_tokenizer()
    print(f"[+] Loaded base foundation model: {model.config._name_or_path}")