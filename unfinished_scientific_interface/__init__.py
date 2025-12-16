import flask
from flask import Flask, redirect, url_for

app = Flask(__name__)

app.config.from_object("interface.config")

import unfinished_scientific_interface.views # noqa: E402  pylint: disable=wrong-import-position