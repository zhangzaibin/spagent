from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor

cache_dir = "./orient_anything_cache"

print("Downloading Orient-Anything large checkpoint...")

ckpt = hf_hub_download(
    repo_id="Viglong/Orient-Anything",
    filename="croplargeEX2/dino_weight.pt",
    repo_type="model",
    cache_dir=cache_dir,
)

print("checkpoint:", ckpt)

print("Downloading dinov2-large processor/config...")

AutoImageProcessor.from_pretrained(
    "facebook/dinov2-large",
    cache_dir=cache_dir,
)

print("dinov2-large processor cached")