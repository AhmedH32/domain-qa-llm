import matplotlib.pyplot as plt
import torch
import yaml
from trl import SFTConfig, SFTTrainer

from src.data import load_and_prepare_dataset
from src.model import setup_model_and_tokenizer


def save_loss_plot(log_history, output_dir):
    steps, losses = [], []
    for log in log_history:
        if "loss" in log and "step" in log:
            steps.append(log["step"])
            losses.append(log["loss"])
            
    if not steps:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, marker="o", color="#0284c7", linewidth=2, label="SFT Loss")
    plt.title("Qwen2.5 Base Model LoRA Domain Fine-Tuning")
    plt.xlabel("Training Steps")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(f"{output_dir}/loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

def run_training(config_path="configs/train_config.yaml"):
    train_ds, eval_ds, config = load_and_prepare_dataset(config_path)
    model, tokenizer, peft_config = setup_model_and_tokenizer(config_path)

    train_cfg = config["training"]
    max_seq_len = config["model"]["max_seq_length"]

    sft_config = SFTConfig(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        logging_steps=train_cfg["logging_steps"],
        fp16=train_cfg.get("fp16", torch.cuda.is_available()),
        optim=train_cfg.get("optim", "adamw_torch"),
        max_length=max_seq_len,
        packing=False,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config
    )

    print("\n[+] Initiating SFT Training Phase on Base Foundation Model...")
    trainer.train()

    save_loss_plot(trainer.state.log_history, train_cfg["output_dir"])

    print(f"\n[+] Saving trained LoRA adapter to: {train_cfg['output_dir']}")
    trainer.model.save_pretrained(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])

if __name__ == "__main__":
    run_training()