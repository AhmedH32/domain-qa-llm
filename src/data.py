import yaml
from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a knowledgeable and empathetic medical AI assistant. "
    "Provide accurate, clear, and professional information regarding health, diseases, and medical conditions."
)

def format_to_chatml(example):
    """
    Transforms MedQuad Question/Answer rows into standard ChatML messages.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["Question"]},
        {"role": "assistant", "content": example["Answer"]}
    ]
    return {"messages": messages}

def load_and_prepare_dataset(config_path="configs/train_config.yaml"):
    """
    Loads MedQuad dataset and formats splits for Hugging Face SFTTrainer.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    train_split = config["dataset"]["train_split"]
    eval_split = config["dataset"]["eval_split"]

    raw_train = load_dataset(dataset_name, split=train_split)
    raw_eval = load_dataset(dataset_name, split=eval_split)

    formatted_train = raw_train.map(
        format_to_chatml, 
        remove_columns=raw_train.column_names
    )
    formatted_eval = raw_eval.map(
        format_to_chatml, 
        remove_columns=raw_eval.column_names
    )

    return formatted_train, formatted_eval, config

if __name__ == "__main__":
    train_ds, eval_ds, cfg = load_and_prepare_dataset()
    print(f"[+] Loaded {len(train_ds)} train records and {len(eval_ds)} eval records.")
    print("[+] Prepared Medical ChatML sample:\n", train_ds[0]["messages"])