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
    logging.debug('request:', text)
    model_name = flask.request.values.get('model_name')
    logging.debug('model_name:', model_name)
    actual = predict.predict(predict.load_cached_model(model_name), text)
    logging.debug('result:', actual)
    response = flask.make_response(actual, 200)
    response.mimetype = "text/plain"
    return response


def main():
    app.run(host='localhost')


if __name__ == '__main__':
    main()
