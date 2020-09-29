import flask
import werkzeug

import nakdimon

app = flask.Flask(__name__)


@app.route('/', methods=["POST"])
def diacritize():
    text = flask.request.values.get('text')
    if not text:
        text = flask.request.files.get('text').read().decode('utf-8')
        if not text:
            raise werkzeug.exceptions.BadRequest
    print('request:', text)
    actual = nakdimon.call_nakdimon(text)
    print('result:', actual)
    response = flask.make_response(actual, 200)
    response.mimetype = "text/plain"
    return response


if __name__ == '__main__':
    app.run(host='localhost')
