from huggingface_hub import upload_folder

upload_folder(
    repo_id="meancodi/opssim-qwen25-1p5b-grpo-lora",
    folder_path="clean_outputs_folder",
    ignore_patterns=["train.log"]
)