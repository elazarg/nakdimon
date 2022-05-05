import argparse


if __name__ == '__main__':
    import sys
    import os
    import train
    import metrics
    import predict
    import app

    os.environ["FLASK_APP"] = "app.py"
    os.environ["FLASK_ENV"] = "development"

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="""Reproduce the Nakdimon paper.""",
    )
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_train = subparsers.add_parser('train', help='train a model')
    parser_train.set_defaults(func=train.main)

    parser_eval = subparsers.add_parser('eval', help='evaluate a model')
    parser_eval.set_defaults(func=metrics.main)

    parser_predict = subparsers.add_parser('predict', help='diacritize a text file')
    parser_predict.add_argument('input', type=str, help='input file')
    parser_predict.add_argument('output', type=str, help='output file')
    parser_predict.set_defaults(func=predict.main)

    parser_daemon = subparsers.add_parser('daemon', help='run nakdimon as a daemon')
    parser_daemon.set_defaults(func=app.main)
    # parser.add_argument("daemon", help="Run the daemon.")
    args = parser.parse_args()
    args.func(args)
    sys.exit(0)
