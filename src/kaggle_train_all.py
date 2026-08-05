import os

# Pin execution to single GPU to prevent PyTorch DataParallel tensor scatter errors
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
import shutil

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset, load_dataset_builder
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# Dynamically resolve project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
ADAPTERS_DIR = os.path.join(ROOT_DIR, "adapters")
ZIP_OUTPUT_PATH = os.path.join(ROOT_DIR, "all_adapters")

MODEL_ID = "Qwen/Qwen2.5-0.5B"

def save_loss_plot(log_history, output_dir, agent_name):
    """Generates and saves cross-entropy SFT loss curve for the domain adapter."""
    steps, losses = [], []
    for log in log_history:
        if "loss" in log and "step" in log:
            steps.append(log["step"])
            losses.append(log["loss"])

    if not steps:
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, marker="o", color="#0284c7", linewidth=2, label=f"{agent_name.upper()} SFT Loss")
    plt.title(f"Qwen2.5-0.5B LoRA Convergence - {agent_name.upper()}")
    plt.xlabel("Training Steps")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

def format_chatml(system_prompt, user_msg, assistant_msg):
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
    )

AGENTS = [
    {
        "name": "us_law",
        "repo": "dzunggg/legal-qa-v1",
        "split": "train",
        "map_fn": lambda x: {"text": format_chatml("You are an expert in US Law.", x["question"], x["answer"])}
    },
    {
        "name": "agriculture",
        "repo": "YuvrajSingh9886/Agriculture-Plan-Diseases-QA-Pairs-Dataset",
        "split": "train",
        "map_fn": lambda x: {"text": format_chatml("You are an agricultural and botany expert.", x["QUESTION.question"], x["ANSWER"])}
    },
    {
        "name": "fitness",
        "repo": "hammamwahab/fitness-qa",
        "split": "train",
        "map_fn": lambda x: {"text": format_chatml("You are a professional fitness coach specializing in heavy compound lifts and cardio routines.", x["question"], x["answer"])}
    },
    {
        "name": "sql",
        "repo": "knowrohit07/know_sql",
        "split": "validation",
        "map_fn": lambda x: {"text": format_chatml("You are a database engineer.", x["question"], x["answer"])}
    },
    {
        "name": "python",
        "repo": "iamtarun/python_code_instructions_18k_alpaca",
        "split": "train",
        "map_fn": lambda x: {"text": format_chatml("You are a Python software engineer.", x["instruction"], x["output"])}
    },
    {
        "name": "cpp",
        "repo": "sahil2801/CodeAlpaca-20k",
        "split": "train",
        "map_fn": lambda x: {"text": format_chatml("You are a C++ programmer. Provide standard library syntax, clear OOP code, and robust logic.", x["instruction"], x["output"])}
    }
]

def verify_all_datasets():
    """Validates Hugging Face Hub dataset availability prior to model loading."""
    print("=" * 60)
    print("[+] PRE-FLIGHT CHECK: VERIFYING HUGGINGFACE DATASETS")
    print("=" * 60)
    failed = []
    for agent in AGENTS:
        try:
            load_dataset_builder(agent["repo"])
            print(f"  ✓ [{agent['name'].upper()}] '{agent['repo']}' verified.")
        except Exception as e:
            print(f"  ✗ [{agent['name'].upper()}] '{agent['repo']}' FAILED: {e}")
            failed.append(agent["name"])
    
    if failed:
        raise RuntimeError(f"[-] Training pipeline aborted. Unavailable datasets: {failed}")
    print("[+] All 6 datasets verified successfully! Proceeding to training.\n")

def train_all():
    verify_all_datasets()

    print(f"[+] Root directory set to: {ROOT_DIR}")
    print(f"[+] Adapters output directory: {ADAPTERS_DIR}")
    print(f"[+] Initializing Tokenizer: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    os.makedirs(ADAPTERS_DIR, exist_ok=True)

    for agent in AGENTS:
        print(f"\n" + "="*60)
        print(f"[+] STARTING TRAINING: {agent['name'].upper()}")
        print("="*60)

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to("cuda")

        output_dir = os.path.join(ADAPTERS_DIR, f"qwen-{agent['name']}-lora")

        raw_ds = load_dataset(agent["repo"], split=agent["split"])
        max_rows = min(2000, len(raw_ds))
        train_ds = raw_ds.select(range(max_rows)).map(agent["map_fn"])

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        sft_config = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=1,
            fp16=True,
            logging_steps=25,
            max_length=512,
            dataset_text_field="text",
            report_to="none"
        )

        trainer = SFTTrainer(
            model=base_model,
            args=sft_config,
            train_dataset=train_ds,
            processing_class=tokenizer,
            peft_config=peft_config
        )

        trainer.train()

        print(f"[+] Saving loss graph and adapter to: {output_dir}")
        save_loss_plot(trainer.state.log_history, output_dir, agent["name"])
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        del trainer, base_model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n[+] Compressing all adapters into '{ZIP_OUTPUT_PATH}.zip'...")
    shutil.make_archive(ZIP_OUTPUT_PATH, 'zip', ADAPTERS_DIR)
    print("[+] Training pipeline finished successfully.")

if __name__ == "__main__":
    train_all()
    
    try:
        from src.evaluate_all import run_full_evaluation
        run_full_evaluation()
    except Exception as e:
        print(f"[-] Evaluation warning: {e}")