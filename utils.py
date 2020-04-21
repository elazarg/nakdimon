from typing import List
import os


def iterate_files(base_paths: List[str]) -> List[str]:
    for name in base_paths:
        if not os.path.isdir(name):
            yield name
            continue
        for root, dirs, files in os.walk(name):
            for fname in files:
                path = os.path.join(root, fname)
                yield path
