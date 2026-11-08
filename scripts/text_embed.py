"""Embed every clean ad description once per sentence-transformer (cache/text_emb_<model>.npy).

Usage: python scripts/text_embed.py [model keys...]   (default: rubert-tiny2 user-base e5-base)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from krisha_ml import text  # noqa: E402

for key in sys.argv[1:] or list(text.MODELS):
    t0 = time.time()
    emb = text.build_embeddings(key)
    print(f"{key}: {emb.shape} in {time.time() - t0:.0f} s", flush=True)
