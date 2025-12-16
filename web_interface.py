#!/usr/bin/env python

print("starting web interface (may take a while on first run)")
from dash import Dash, dcc, html, Input, Output, State, callback, exceptions
import numpy as np
import base64
from pathlib import Path
from dimensionality_reduction import create_plotly_figure, add_track, compute_pc_mapping
import time

from call_chuck import generate_sonification

added_tracks = []

app = Dash(__name__)


base_style = {
    "width": "100%",
    "height": "80px",
    "lineHeight": "80px",
    "borderWidth": "2px",
    "borderStyle": "dashed",
    "borderRadius": "6px",
    "textAlign": "center",
    "cursor": "pointer",
}

pc_mapper = lambda pc1, pc2, pc3: np.zeros(50)

def initialize_pc_mapping():
    global pc_mapper
    pc_mapper = compute_pc_mapping()

def initialize_dash_app():
    raw_added_tracks = [
        ("No Idea", "Don Toliver", "./data/NoIdea_DonToliver.mp3"),
        ("Where The Light Goes", "Hello Meteor", "./data/WhereTheLightGoes_HelloMeteor.mp3")
    ]

    initial_new_tracks = [add_track(t) for t in raw_added_tracks]

    fig = create_plotly_figure(initial_new_tracks)

    initialize_pc_mapping()


    # https://dash.plotly.com/dash-core-components/upload
    app.layout = html.Div([
        dcc.Graph(
            id = "pca-scatter",
            figure=fig,
            clear_on_unhover=True,
            style={'width': '100%', 'height': '80vh'}
        ),
        html.Div(id="click-output"),
        html.Audio(id="audio-playback", src="", controls=True, autoPlay=True),
        html.Div(
            [
            dcc.Upload(
                id="upload-mp3",
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                children=html.Div(
                    id="upload-status",
                    children="Drop an mp3 file here or click to upload"
                ),
                # Allow multiple files to be uploaded
                multiple=False
            ),
            dcc.Input(
                id="upload-title",
                type="text",
                placeholder="Title"
            ),
            dcc.Input(
                id="upload-artist",
                type="text",
                placeholder="Artist"
            ),

            html.Button(id="submit-upload",
                        children="Upload track",
                        n_clicks=0,
                        style={"marginTop": "12px"},
                        disabled=True
                        ),
            ],
            style={"maxWidth": "400px"}
        ),
        html.Div(id="output-upload-mp3"),
        dcc.Store(id="added-tracks", data=initial_new_tracks)
    ]
)


@callback(
    Output("upload-status", "children"),
    Output("upload-mp3", "style"),
    Input("upload-mp3", "filename"),
)
def show_selected_file(filename):

    if filename:
        return (
            html.Div([
                html.Span(filename),
            ]),
            {
                **base_style,
                "borderColor": "#2ecc71",
                "backgroundColor": "#ecf9f1",
            },
        )

    return "Drop an mp3 file here or click to upload", base_style



@callback(
        Output("submit-upload", "disabled"),
        Input("upload-mp3", "contents"),
        Input("upload-title", "value"),
        Input("upload-artist", "value")
)
def enable_submit(contents, title, artist):
    if contents and title and artist:
        return False
    return True


@callback(
        Output("click-output", "children"),
        Output("audio-playback", "src"),
        Input("pca-scatter", "clickData"),
)
def handle_click(clickData):
    try:
        if clickData is None:
            return "Click a point on the scatterplot!\n", ""
        point = clickData["points"][0]
        pc1, pc2, pc3, track_id, track_name, track_artist = point["customdata"]
        result = f"Sonification of {track_name} {track_artist}\n"
        #print(added_tracks)

        print(f"sending to ChucK: {(pc1, pc2, pc3)}")
        generate_sonification(pc1, pc2, pc3, pc_mapper=pc_mapper, dest="assets/sonification.wav")

        timestamp = int(time.time())
        src = f"/assets/sonification.wav?t={timestamp}"
        return result, src
    except:
        return "Click a point on the scatterplot!\n", ""


@callback(
        Output("pca-scatter", "figure"),
        Output("added-tracks", "data"),
        Output("upload-mp3", "contents"),
        Output("upload-title", "value"),
        Output("upload-artist", "value"),
        Output("upload-status", "children", allow_duplicate=True),
        Output("upload-mp3", "style", allow_duplicate=True),
        Input("submit-upload", "n_clicks"),
        State("upload-mp3", "contents"),
        State("upload-mp3", "filename"),
        State("upload-title", "value"),
        State("upload-artist", "value"),
        State("added-tracks", "data"),
        prevent_initial_call=True
)
def upload_file(n_clicks, contents, filename, title, artist, tracks):
    if contents is None or title is None or artist is None:
        raise exceptions.PreventUpdate
    print("contents:")
    #print(contents)
    content_type, encoded_string = contents.split(",")
    decoded_data = base64.b64decode(encoded_string)
    filename = str(Path("./data/")/filename)
    with open(filename, "wb") as f:
        f.write(decoded_data)

    print(title)
    print(artist)
    print(filename)
    print("--------")
    
    tracks.append(add_track((title, artist, filename)))

    return create_plotly_figure(tracks), tracks, None, "", "", "Drop an mp3 file here or click to upload", base_style

if __name__ == "__main__":
    initialize_dash_app()
    app.run()