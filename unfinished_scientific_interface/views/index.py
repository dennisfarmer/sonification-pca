import flask
from flask import redirect, url_for

import csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pathlib

import unfinished_scientific_interface
import pickle

# flask --app interface --debug run --host 0.0.0.0 --port 8000


@unfinished_scientific_interface.app.route('/test')
def show_test():
    context = {}
    return flask.render_template("index.html", **context)


def save_file_to_disk(fileobj):
    """Save File To Disk."""
    filename = fileobj.filename
    stem = pathlib.Path(filename).stem.lower()
    suffix = pathlib.Path(filename).suffix.lower()
    basename = f"{stem}{suffix}"

    dest_path = unfinished_scientific_interface.app.config["UPLOAD_FOLDER"]/basename
    fileobj.save(dest_path)
    return dest_path

@unfinished_scientific_interface.app.route('/')
def show_index():
    context = {}
    return flask.render_template("index.html", **context)


@unfinished_scientific_interface.app.route("/set_parameters/<basename>")
def set_parameters(basename):
    with open(unfinished_scientific_interface.app.config["UPLOAD_FOLDER"]/basename, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        column_names = reader.fieldnames or []

    context = {
        "column_names": column_names,
        "basename": basename
    }
    return flask.render_template("set_parameters.html", **context)

@unfinished_scientific_interface.app.route("/render_sonification/", methods=["POST"])
def render_sonification():
    mode = flask.request.form.get("mode")
    key_sig = flask.request.form.get("key_sig")
    timestep = flask.request.form.get("timestep")
    start_date = flask.request.form.get("start_date")
    major_dates = flask.request.form.get("major_dates")
    vis_y1 = flask.request.form.get("visualize_y1")
    vis_y2 = flask.request.form.get("visualize_y2")
    basename = flask.request.form.get("basename")
    filename = unfinished_scientific_interface.app.config["UPLOAD_FOLDER"]/basename

    major_dates = [int(d) for d in major_dates.split(",")]

    df = pd.read_csv(filename)
    matplotlib.use("Agg")

    plt.figure()
    start_date = str(start_date)[:4]
    df = df[int(df["Year"]) >= int(start_date)]
    df.set_index("Year")[[vis_y1, vis_y2]].plot()
    for date in major_dates:
        plt.axvline(x=date, color='red')
    png_basepath = f"{pathlib.Path(filename).stem}_plot.png"
    plt.savefig(unfinished_scientific_interface.app.config["UPLOAD_FOLDER"]/png_basepath)
    plt.close()

    chuck_params = {
        "df": df,
        "start_date": start_date,
        "major_dates": major_dates
    }

    #pickle.dump(chuck_params, open(interface.app.config["UPLOAD_FOLDER"]/f"{pathlib.Path(filename).stem}_params.pkl", "wb"))
    pickle.dump(chuck_params, open(unfinished_scientific_interface.app.config["UPLOAD_FOLDER"]/"params.pkl", "wb"))

    return redirect(url_for("show_result", png_basepath=png_basepath))

@unfinished_scientific_interface.app.route("/result/<png_basepath>")
def show_result(png_basepath):
    context = {
        "png_path": f"/uploads/{png_basepath}"
    }
    return flask.render_template("result.html", **context)

@unfinished_scientific_interface.app.route("/uploads/<path:filename>")
def file(filename):
    try:
        return flask.send_from_directory(
            unfinished_scientific_interface.config.UPLOAD_FOLDER, filename
        )
    except (FileNotFoundError, PermissionError):
        flask.abort(404)

@unfinished_scientific_interface.app.route("/csvfile/", methods=["POST"])
def set_csvfile():
    fileobj = flask.request.files["csvfile"]
    if not fileobj or fileobj.filename == "":
        flask.abort(400)

    filename = save_file_to_disk(fileobj)
    basename = pathlib.Path(filename).name
    return redirect(url_for("set_parameters", basename=basename))




