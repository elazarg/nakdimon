from __future__ import annotations
import flask
import werkzeug
import logging

import predict

app = flask.Flask(__name__)


@app.route('/', methods=["POST"])
def diacritize():
    text = flask.request.values.get('text')
    if not text:
        text = flask.request.files.get('text').read().decode('utf-8')
        if not text:
            raise werkzeug.exceptions.BadRequest
    logging.info(f'request: {text}')
    model_name = flask.request.values.get('model_name')
    logging.info(f'model_name: {model_name}')
    actual = predict.predict(predict.load_cached_model(model_name), text)
    logging.debug(f'result: {actual}')
    response = flask.make_response(actual, 200)
    response.mimetype = "text/plain"
    return response


def main():
    logging.info("Loading models/Nakdimon.h5")
    try:
        predict.predict(predict.load_cached_model('models/Nakdimon.h5'), "שלום")
        logging.info("Done loading.")
    except OSError:
        logging.warning("Could not load default model")
    app.run(host='localhost')


if __name__ == '__main__':
    main()
