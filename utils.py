from typing import List

import sys
import contextlib
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


# from: https://stackoverflow.com/a/45735618/2289509
@contextlib.contextmanager
def smart_open(filename: str, mode: str = 'r', *args, **kwargs):
    """Open files and i/o streams transparently."""
    if filename == '-':
        if 'r' in mode:
            stream = sys.stdin
        else:
            stream = sys.stdout
        if 'b' in mode:
            fh = stream.buffer
        else:
            fh = stream
        close = False
    else:
        fh = open(filename, mode, *args, **kwargs)
        close = True

    try:
        yield fh
    finally:
        if close:
            try:
                fh.close()
            except AttributeError:
                pass
