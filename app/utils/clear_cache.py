import shutil
import os
from pathlib import Path


def clear_cache():
    """
    Clear the __pycache__ directories in the current directory and its subdirectories.
    """

    path = Path(__file__).parent.parent
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            if name == "__pycache__":
                full_path = os.path.join(root, name)
                print(f"Removing {full_path}")
                shutil.rmtree(full_path)


if __name__ == "__main__":
    clear_cache()
