import os
from huggingface_hub import snapshot_download

# Define models to download
models = [
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct"
]

# Define local storage directory
base_model_dir = "models"
os.makedirs(base_model_dir, exist_ok=True)

for model_id in models:
    print(f"Downloading {model_id}...")
    try:
        # Create a local directory name based on the model ID
        local_dir = os.path.join(base_model_dir, model_id.replace("/", "_"))

        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Ensure actual files are downloaded
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # Optional: ignore non-pytorch weights if desired, but safer to get all
        )
        print(f"Successfully downloaded {model_id} to {local_dir}")
    except Exception as e:
        print(f"Failed to download {model_id}: {e}")
