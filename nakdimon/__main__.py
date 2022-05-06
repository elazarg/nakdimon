import argparse
import sys
import os
import logging

import run_test
import train
import metrics
import predict
import app


if __name__ == '__main__':
    os.environ["FLASK_APP"] = "app.py"
    os.environ["FLASK_ENV"] = "development"

    logging.basicConfig(encoding='utf-8', level=logging.INFO, format='%(levelname)s - %(message)s')

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="""Train and evaluate Nakdimon. Reproduce the Nakdimon paper.""",
    )
    subparsers = parser.add_subparsers(help='sub-command help', dest="command", required=True)

    parser_train = subparsers.add_parser('train', help='train Nakdimon')
    parser_train.add_argument('--wandb', action='store_true', help='use wandb', default=False)
    parser_train.set_defaults(func=train.main)

    test_systems = ['Snopi', 'Morfix', 'Dicta', 'Nakdimon']
    # iterate over folders to find available options:
    available_tests = [f'tests/{folder}' for folder in os.listdir('tests/')
                       if os.path.isdir(f'tests/{folder}') and os.path.isdir(f'tests/{folder}/expected')]

    parser_eval = subparsers.add_parser('results', help='evaluate the results of a test run')
    parser_eval.add_argument('test_set', choices=available_tests, help='choose test set', default='new')
    partial_result, _ = parser.parse_known_args()
    if partial_result.command == 'results':
        systems = [folder for folder in os.listdir(partial_result.test_set)
                   if os.path.isdir(f'{partial_result.test_set}/{folder}') and folder != 'expected']
    parser_eval.add_argument('--systems', choices=test_systems, nargs='+', help='list of systems to evaluate',
                             default=test_systems)
    parser_eval.set_defaults(func=metrics.main)

    parser_test = subparsers.add_parser('run_test', help='diacritize a test set')
    parser_test.add_argument('test_set', choices=available_tests, help='choose test set', default='new')
    parser_test.add_argument('--systems', choices=test_systems, help='diacritization system', default='Nakdimon')
    parser_test.set_defaults(func=run_test.main)

    parser_predict = subparsers.add_parser('predict', help='diacritize a text file')
    parser_predict.add_argument('input', type=str, help='input file')
    parser_predict.add_argument('output', type=str, help='output file')
    parser_predict.set_defaults(func=predict.main)

    parser_daemon = subparsers.add_parser('daemn', help='run nakdimon as a daemon')
    parser_daemon.set_defaults(func=app.main)
    # parser.add_argument("daemon", help="Run the daemon.")

    args = parser.parse_args()
    kwargs = vars(args).copy()
    del kwargs['command']
    del kwargs['func']
    args.func(**kwargs)

    sys.exit(0)
