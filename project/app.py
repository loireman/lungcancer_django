from flask import Flask
from flask_frozen import Freezer
from flask_flatpages import FlatPages

UPLOAD_FOLDER = 'upload'

app = Flask(__name__)
app.config.from_pyfile('settings.py')
pages = FlatPages(app)
freezer = Freezer(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER