"""Upload the exported models to the Hugging Face Hub.

Expects the layout written by ``brainssl export all``:

    exports/segmentation/   README.md (model card), model.safetensors, config.json,
                            segmentation.onnx, segmentation.json
    exports/connectome/     README.md, connectome.json, weights_<atlas>.safetensors

and creates or updates one model repository per folder. Needs a token with
write access in HF_TOKEN, or a prior ``hf auth login``.

    python scripts/push_to_hub.py --exports exports --user mahdisabetkish
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

REPOS = {
    "segmentation": "brainssl-tumour-segmentation",
    "connectome": "brainssl-abide-connectome",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--exports", type=Path, default=Path("exports"))
    parser.add_argument("--user", default="mahdisabetkish")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for folder, name in REPOS.items():
        local = args.exports / folder
        if not (local / "README.md").exists():
            print(f"skipping {folder}: {local / 'README.md'} not found")
            continue
        repo_id = f"{args.user}/{name}"
        files = sorted(p.relative_to(local).as_posix() for p in local.rglob("*") if p.is_file())
        print(f"{repo_id}: {len(files)} files")
        if args.dry_run:
            for f in files:
                print("   ", f)
            continue
        api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=str(local), repo_type="model",
                          commit_message="Upload model and model card")
        print(f"    https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
