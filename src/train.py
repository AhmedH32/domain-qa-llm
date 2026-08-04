import torch
import yaml
from trl import SFTConfig, SFTTrainer

from src.data import load_and_prepare_dataset
from src.model import setup_model_and_tokenizer


def run_training(config_path="configs/train_config.yaml"):
    """
    Executes supervised fine-tuning loop and saves adapter artifacts.
    """
    # 1. Load Data & Prepared Model Stack
    train_ds, eval_ds, config = load_and_prepare_dataset(config_path)
    model, tokenizer, peft_config = setup_model_and_tokenizer(config_path)

    train_cfg = config["training"]
    max_seq_len = config["model"]["max_seq_length"]

    # 2. Configure SFT Parameters via SFTConfig
    sft_config = SFTConfig(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        logging_steps=train_cfg["logging_steps"],
        fp16=train_cfg.get("fp16", torch.cuda.is_available()),
        optim=train_cfg.get("optim", "adamw_torch"),
        max_length=max_seq_len,  # Correct parameter name in TRL SFTConfig
        packing=False,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none"
    )

    # 3. Instantiate SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config
    )

    # 4. Train and Persist Weights
    print("\n[+] Initiating SFT Training Phase...")
    trainer.train()

    print(f"\n[+] Saving trained LoRA adapter to: {train_cfg['output_dir']}")
    trainer.model.save_pretrained(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])
    print("[+] Training complete successfully.")

if __name__ == "__main__":
    run_training()