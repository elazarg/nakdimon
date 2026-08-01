"""Command-line entry point: `diacritize input.txt -o output.txt`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nakdimon import diacritize
from nakdimon.hebrew import remove_niqqud


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="diacritize", description="Add Hebrew niqqud to plain text.")
    parser.add_argument("input_path", help="Input file path, or '-' for stdin.")
    parser.add_argument("-o", "--output", default="-", help="Output file path, or '-' for stdout (default: -).")
    args = parser.parse_args(argv)

    text = sys.stdin.read() if args.input_path == "-" else Path(args.input_path).read_text(encoding="utf-8")

    result = diacritize(remove_niqqud(text))

    if args.output == "-":
        sys.stdout.write(result)
    else:
        Path(args.output).write_text(result, encoding="utf-8")
