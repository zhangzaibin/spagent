from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="moondream/moondream3-preview",
    resume_download=True
)

print("downloaded to:", path)
