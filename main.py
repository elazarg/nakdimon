import argparse
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser(description="""Diacritize Hebrew text.""")
    parser.add_argument('input_path', help='input file')
    parser.add_argument('-o', help='output file', default="-")
    args = parser.parse_args()

    if not pathlib.Path(args.input_path).exists():
        print(f"File not found: '{args.input_path}'", file=sys.stderr)
        sys.exit(1)

    import nakdimon.predict
    nakdimon.predict.main(args.input_path, args.o)
    sys.exit(0)


if __name__ == '__main__':
    main()
