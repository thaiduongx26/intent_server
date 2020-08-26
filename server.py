from flask import Flask
from flask import request
from model import handle_request

app = Flask(__name__)


@app.route('/message', methods=['POST'])
def intent_detection():
    data = request.json
    intent = handle_request(data['message'])
    return {'intent': intent}

if __name__ == '__main__':
    app.run()