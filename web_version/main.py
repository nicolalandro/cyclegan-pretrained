import io

from flask import Flask
from flask import render_template
from werkzeug import run_simple

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', show_result=False)

if __name__ == "__main__":
    run_simple('localhost', 5001, app, use_reloader=True, use_debugger=True)
