import argparse
import logging
import os
import sys
from importlib.metadata import PackageNotFoundError, version

from nakdimon.config import MAIN_MODEL

try:
    __version__ = version("nakdimon")
except PackageNotFoundError:  # editable install or unreleased checkout
    __version__ = "0.0.0+local"


def _discover_test_sets(base: str = "tests") -> list[str]:
    """Find tests/<x>/ subdirs that contain an expected/ folder. Empty when
    invoked outside the source checkout, which is the common case for the
    installed CLI — that's fine, the run_test / results commands fall back to
    treating --test_set as a free-form path argument."""
    try:
        entries = os.listdir(base)
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return []
    return [f"{base}/{f}" for f in entries
            if os.path.isdir(f"{base}/{f}/expected")]


def do_train(**kwargs) -> None:
    from nakdimon import train
    train.main(**kwargs)


def do_run_test(**kwargs) -> None:
    from nakdimon import run_test
    run_test.main(**kwargs)


def do_metrics(**kwargs) -> None:
    from nakdimon import metrics
    metrics.main(**kwargs)


def do_predict(**kwargs) -> None:
    from nakdimon import predict
    predict.main(**kwargs)


def do_server(**kwargs):
    import importlib.util
    import os
    import sys
    spec = importlib.util.find_spec("server")
    assert spec is not None and spec.origin is not None
    logging.info("Executing flask server...")
    os.execv(sys.executable, [sys.executable, spec.origin])
    exit(1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="""Train and evaluate Nakdimon and other diacritizers. Reproduce the Nakdimon paper.""",
    )
    parser.add_argument('-q', '--quiet', action='store_true', help='suppress info logging.', default=False)
    parser.add_argument('-V', '--version', action='version', version=f'nakdimon {__version__}')

    subparsers = parser.add_subparsers(help='sub-command help', dest="command", required=True)

    parser_train = subparsers.add_parser('train', help='train Nakdimon')
    parser_train.add_argument('--wandb', action='store_true', help='use wandb.', default=False)
    parser_train.add_argument('--model', help='path to output model (.keras file)', default='models/Nakdimon.keras', dest='model_path')
    parser_train.add_argument('--ablation', help='ablation test', default=None, dest='ablation_name')
    parser_train.set_defaults(func=do_train)

    test_systems = ['Snopi', 'Morfix', 'Dicta', 'Nakdimon', 'MajMod', 'MajAllWithDicta', 'MajAllWithoutDicta']
    available_tests = _discover_test_sets()

    parser_test = subparsers.add_parser('run_test', help='diacritize a test set')
    parser_test.add_argument('--test_set', help='choose test set (path to a directory with expected/ subdir)',
                             default='tests/new')
    parser_test.add_argument('--system', choices=test_systems, help='diacritization system to use', default='Nakdimon')
    parser_test.add_argument('--model', help='path to model (.onnx file)', default=MAIN_MODEL, dest='model_path')
    parser_test.add_argument('--skip-existing', action='store_true', help='skip existing files')
    parser_test.set_defaults(func=do_run_test)

    parser_predict = subparsers.add_parser('predict', help='diacritize a text file')
    parser_predict.add_argument('input_path', help='input file')
    parser_predict.add_argument('output_path', help='output file')
    parser_predict.set_defaults(func=do_predict)

    # parser_daemon = subparsers.add_parser('server', help='run Nakdimon server as a daemon')
    # parser_daemon.set_defaults(func=do_server)

    parser_eval = subparsers.add_parser('results', help='evaluate the results of a test run')
    parser_eval.add_argument('--test_set', help='choose test set (path to a directory with expected/ subdir)',
                             default='tests/new')
    partial_result, _ = parser.parse_known_args()
    if partial_result.command == 'results' and os.path.isdir(partial_result.test_set):
        test_systems = [folder for folder in os.listdir(partial_result.test_set)
                        if os.path.isdir(f'{partial_result.test_set}/{folder}') and folder != 'expected']
    parser_eval.add_argument('--systems', nargs='+', help='list of systems to evaluate',
                             default=test_systems)
    parser_eval.set_defaults(func=do_metrics)

    # Reference available_tests in help text so it is not unused when populated.
    if available_tests:
        parser_test.epilog = f"discovered test sets: {', '.join(available_tests)}"
        parser_eval.epilog = parser_test.epilog

    args = parser.parse_args()

    if args.quiet:
        logging.disable(logging.INFO)
    del args.quiet

    kwargs = vars(args).copy()
    del kwargs['command']
    del kwargs['func']
    args.func(**kwargs)

    sys.exit(0)


def diacritize_main():
    import argparse
    import pathlib
    import sys
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


def diacritize(text: str, model_path: str = MAIN_MODEL) -> str:
    import nakdimon.predict
    return nakdimon.predict.predict(text, model_path)
