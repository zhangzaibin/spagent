#!/usr/bin/env bash
# Download Orient Anything V2 checkpoint from Hugging Face.
# Saves to: checkpoints/orient_anything_v2/rotmod_realrotaug_best.pt

set -euo pipefail

DEST="checkpoints/orient_anything_v2"
mkdir -p "$DEST"

huggingface-cli download Viglong/OriAnyV2_ckpt \
    demo_ckpts/rotmod_realrotaug_best.pt \
    --local-dir "$DEST" \
    --local-dir-use-symlinks False

echo "Checkpoint saved to $DEST/demo_ckpts/rotmod_realrotaug_best.pt"
echo "Move or symlink to $DEST/rotmod_realrotaug_best.pt if needed."
