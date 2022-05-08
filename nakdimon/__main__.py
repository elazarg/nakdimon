import argparse
import sys
import os
import logging


def do_train(**kwargs):
    import train
    train.main(**kwargs)


def do_run_test(**kwargs):
    import run_test
    run_test.main(**kwargs)


def do_metrics(**kwargs):
    import metrics
    metrics.main(**kwargs)


def do_predict(**kwargs):
    import predict
    predict.main(**kwargs)


def do_server(**kwargs):
    import os
    import sys
    import pkgutil
    package = pkgutil.get_loader("server")
    os.execv(sys.executable, [sys.executable, package.get_filename()])


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="""Train and evaluate Nakdimon. Reproduce the Nakdimon paper.""",
    )
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress info logging.', default=False)

    subparsers = parser.add_subparsers(help='sub-command help', dest="command", required=True)

    parser_train = subparsers.add_parser('train', help='train Nakdimon')
    parser_train.add_argument('--wandb', action='store_true', help='use wandb', default=False)
    parser_train.set_defaults(func=do_train)

    test_systems = ['Snopi', 'Morfix', 'Dicta', 'Nakdimon', 'MajMod', 'MajAllWithDicta', 'MajAllWithoutDicta']
    # iterate over folders to find available options:
    available_tests = [f'tests/{folder}' for folder in os.listdir('tests/')
                       if os.path.isdir(f'tests/{folder}') and os.path.isdir(f'tests/{folder}/expected')]

    parser_test = subparsers.add_parser('run_test', help='diacritize a test set')
    parser_test.add_argument('--test_set', choices=available_tests, help='choose test set', default='tests/new')
    parser_test.add_argument('--system', choices=test_systems, help='diacritization system to use', default='Nakdimon')
    parser_test.add_argument('--model', help='path to model', default='final_model/final.h5', dest='model_path')
    parser_test.add_argument('--skip-existing', action='store_true', help='skip existing files')
    parser_test.set_defaults(func=do_run_test)

    parser_predict = subparsers.add_parser('predict', help='diacritize a text file')
    parser_predict.add_argument('input', help='input file')
    parser_predict.add_argument('output', help='output file')
    parser_predict.set_defaults(func=do_predict)

    parser_daemon = subparsers.add_parser('server', help='run Nakdimon server as a daemon')
    parser_daemon.set_defaults(func=do_server)

    parser_eval = subparsers.add_parser('results', help='evaluate the results of a test run')
    parser_eval.add_argument('--test_set', choices=available_tests, help='choose test set', default='tests/new')
    partial_result, _ = parser.parse_known_args()
    if partial_result.command == 'results':
        test_systems = [folder for folder in os.listdir(partial_result.test_set)
                        if os.path.isdir(f'{partial_result.test_set}/{folder}') and folder != 'expected']
    parser_eval.add_argument('--systems', choices=test_systems, nargs='+', help='list of systems to evaluate',
                             default=test_systems)
    parser_eval.set_defaults(func=do_metrics)

    args = parser.parse_args()
    if args.quiet:
        logging.disable(logging.INFO)
    del args.quiet

    kwargs = vars(args).copy()
    del kwargs['command']
    del kwargs['func']
    args.func(**kwargs)

    sys.exit(0)
