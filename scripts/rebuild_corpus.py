import shutil
from pathlib import Path
import subprocess
import sys

DOCS_DIR = Path("storage/processed/docs")
CHUNKS_PATH = Path("storage/processed/chunks.jsonl")
INDEX_DIR = Path("storage/faiss_index")

SCRIPTS = [
    "scripts/build_canonical_docs.py",
    "scripts/build_chunks.py",
    "scripts/build_index.py",
]

def clean_outputs():
    if DOCS_DIR.exists():
        shutil.rmtree(DOCS_DIR)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    if CHUNKS_PATH.exists():
        CHUNKS_PATH.unlink()

    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

def main():
    clean_outputs()

    for script in SCRIPTS:
        print(f"\nRunning {script} ...")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            raise SystemExit(f"Failed: {script}")

    print("\nCorpus rebuild complete.")

if __name__ == "__main__":
    main()