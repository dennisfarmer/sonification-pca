from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import pickle
import ast

import essentia
import essentia.standard as es
import numpy as np
import os
import subprocess
import logging

interpretations = [
    "PC1: (-) acoustic / rock -> electronic / digital (+)",
    "PC2: (-) fast / happy -> slow / sad (+)",
    "PC3: (-) many notes ->  few notes (+)"
]

musicnn_tags = [
    "rock", 
    "pop", 
    "alternative", 
    "indie", 
    "electronic", 
    "female vocalists", 
    "dance", 
    "00s", 
    "alternative rock", 
    "jazz", 
    "beautiful", 
    "metal", 
    "chillout", 
    "male vocalists", 
    "classic rock", 
    "soul", 
    "indie rock", 
    "mellow", 
    "electronica", 
    "80s", 
    "folk", 
    "90s", 
    "chill", 
    "instrumental", 
    "punk", 
    "oldies", 
    "blues", 
    "hard rock", 
    "ambient", 
    "acoustic", 
    "experimental", 
    "female vocalist", 
    "guitar", 
    "hip-hop", 
    "70s", 
    "party", 
    "country", 
    "easy listening", 
    "sexy", 
    "catchy", 
    "funk", 
    "electro", 
    "heavy metal", 
    "progressive rock", 
    "60s", 
    "rnb", 
    "indie pop", 
    "sad", 
    "house", 
    "happy"
]


def add_track(new_track):
    return {
        "track_name": new_track[0],
        "track_artist": new_track[1],
        "embedding": compute_audio_embedding(new_track[2]),
        }

color_discrete_map={
            "average": "rgba(127, 0, 255, 1)",
            "added": "rgba(0, 128, 255, 1)",
            "hybrid_recommended": "rgba(144, 238, 144, 1)",
            "recommended": "rgba(255, 0, 0, 1)",
            "dataset": "rgba(0, 255, 255, 0.05)"
            }

def map_sizes(category: str) -> float:
    if category == "added":
        return 3.0
    elif category == "average":
        return 3.0
    elif category == "recommended":
        return 3.0
    elif category == "dataset":
        return 1.0
    else:
        return 1.0

def compute_audio_embedding(audio_path: str):
    MUSICNN_SR = 16000
    try:
        audio = es.MonoLoader(filename=audio_path, sampleRate=MUSICNN_SR)()
        musicnn_emb = es.TensorflowPredictMusiCNN(graphFilename='./musicnn_feature_extractor/msd-musicnn-1.pb', output='model/dense_1/BiasAdd')(audio)
        mean_emb = np.mean(musicnn_emb, axis=0)
        mean_emb = mean_emb[np.newaxis, :]
    except:
        return None
    return mean_emb[0]

def load_dataset():
    """
    ```
    metadata_df, feature_df, X = load_dataset()
    ```
    """
    metadata_df = pd.read_pickle("./data/music_metadata_df.pkl")
    feature_df = pd.read_pickle("./data/music_embeddings_df.pkl")

    # compute PCA with embeddings (50 -> 3)
    X = np.array(feature_df["embedding"].tolist())
    
    # compute PCA with features (9 -> 3)
    #X = feature_df[features]
    return metadata_df, feature_df, X

def load_principal_components() -> tuple[list, PCA]:
    """
    returns the list of initial datapoints, and the fitted PCA matrix

    tranform future music embeddings using `pca.transform([track_embedding_1, track_embedding_2, ...])`
    """
    metadata_df, feature_df, X = load_dataset()
    pca, scaler, X_reduced = fit_pca(X)

    data = []
    for (track_id, row), pcs in zip(metadata_df.iterrows(), X_reduced):
        data.append({
            "pc1": pcs[0],
            "pc2": pcs[1],
            "pc3": pcs[2],
            "track_id": track_id,
            "track_name": row["name"],
            "track_artist": ", ".join(ast.literal_eval(row["artists"])),
            "category": "dataset"
        })
    return data, pca, scaler

def transform_added_tracks(added_tracks: list[dict], pca: PCA, scaler: StandardScaler) -> list[dict]:
    data = []
    X = np.array([t["embedding"] for t in added_tracks])
    X = scaler.transform(X)
    X_reduced = pca.transform(X)
    for row, pcs in zip(added_tracks, X_reduced):
        data.append({
            "pc1": pcs[0],
            "pc2": pcs[1],
            "pc3": pcs[2],
            "track_id": "",
            "track_name": row["track_name"],
            "track_artist": row["track_artist"],
            "category": "added"
        })
    return data


def create_plotly_figure(added_tracks: list[dict]):
    data, pca, scaler = load_principal_components()

    added_tracks_transformed = transform_added_tracks(added_tracks, pca, scaler)

    avg_pca_coord = stats.trim_mean([[x["pc1"], x["pc2"], x["pc3"]] for x in added_tracks_transformed], proportiontocut=0)

    for t in added_tracks_transformed:
        print(f"pca point for {t['track_name']} by {t['track_artist']}: {[t['pc1'], t['pc2'], t['pc3']]}")
    print(f"avg pca point: {avg_pca_coord}")

    # make recommendations
    k = 5
    pcs_data = [[d["pc1"], d["pc2"], d["pc3"]] for d in data]
    distances = euclidean_distances([avg_pca_coord], pcs_data)[0]
    closest_indices = np.argsort(distances, kind="quicksort")[:k]
    recommended_track_ids = set([data[i]["track_id"] for i in closest_indices])


    # add user-added tracks and the average point
    data.extend(added_tracks_transformed)
    data.append({
        "pc1": avg_pca_coord[0],
        "pc2": avg_pca_coord[1],
        "pc3": avg_pca_coord[2],
        "track_id": "",
        "track_name": "Added Tracks Average",
        "track_artist": "",
        "category": "average"
    })

    df = pd.DataFrame(data)

    df.loc[df["track_id"].isin(recommended_track_ids), "category"] = "recommended"

    df["size"] = df["category"].apply(map_sizes)
    df["track_artist"] = df["track_artist"].apply(lambda x: f"by {x}" if x != "" else "")

    fig = go.Figure()

    for category, subdf in df.groupby("category"):
        fig.add_trace(
            go.Scatter3d(
                x=subdf["pc1"],
                y=subdf["pc2"],
                z=subdf["pc3"],
                mode="markers",
                name=category,
                marker=dict(
                    size=subdf["size"],
                    color=color_discrete_map[category],
                    opacity=0.8
                ),
                customdata=subdf[[
                    "pc1",
                    "pc2",
                    "pc3",
                    "track_id",
                    "track_name",
                    "track_artist",
                ]].values,
            )
    )


    fig.update_layout(
        title = "Sonification of Principal Component Analysis - for the enhanced interpretability of a content-based music recommender system",
        scene = {
            "xaxis_title": interpretations[0],
            "yaxis_title": interpretations[1],
            "zaxis_title": interpretations[2],
            },
        hoverlabel_align = 'left',
        template="plotly_dark",
        margin=dict(l=0, r=30, t=100, b=0),
        #showlegend=False,
    )
    #print("plotly express hovertemplate:", fig.data[0].hovertemplate)
    fig.update_traces(hovertemplate='<b>%{customdata[4]}</b><br><b>%{customdata[5]}<extra></extra>')

    # the average of all added tracks
    mrpe = df[df["category"] == "average"][["pc1", "pc2", "pc3"]].values[0]

    lines = [
        {"start": mrpe, "end": p} 
        for p in df[df["category"] == "added"][["pc1", "pc2", "pc3"]].values
    ]

    for line in lines:
        fig.add_trace(
            go.Scatter3d(
                x=[line['start'][0], line['end'][0]],
                y=[line['start'][1], line['end'][1]],
                z=[line['start'][2], line['end'][2]],
                mode='lines',
                line=dict(color='rgba(0,128,255,1)', width=3),
                showlegend=False,
                hoverinfo='none',
            )
        )

    return fig

def fit_pca(X, n_components=3):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_reduced = pca.transform(X)
    return pca, scaler, X_reduced

def facet_plt(X_reduced, feature_df, features):
    """
    uses matplotlib, simplier than plotly interface
    """
    #fig = plt.figure
    fig = plt.figure(figsize=(15, 15))
    axes = [fig.add_subplot(3, 3, i + 1, projection='3d') for i in range(len(features))]

    for i, feature in enumerate(features):
        ax = axes[i]
        scatter = ax.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            X_reduced[:, 2],
            c=feature_df[feature],
            s=40,
            cmap='viridis'
        )
        ax.set(
            title=f"PCA colored by {feature}",
            xlabel="1st Eigenvector",
            ylabel="2nd Eigenvector",
            zlabel="3rd Eigenvector"
        )
        #ax.xaxis.set_ticklabels([])
        #ax.yaxis.set_ticklabels([])
        #ax.zaxis.set_ticklabels([])
        fig.colorbar(scatter, ax=ax)

    plt.tight_layout()
    plt.show()

def compute_pc_mapping():
    """
    X_reduced: shape [n_observations, 3]
    corr_loadings_matrix: shape [50, 3], correlations between each tag and each PC
                        (tags are "rock", "pop", "metal", ...)
    pca_scaler: MinMaxScaler to map each PC to a range between 0 and 1
    """
    metadata_df, feature_df, X = load_dataset()
    pca, scaler, X_reduced = fit_pca(X, n_components=3)
    # eigenvalues of the covariance matrix: shape [3, 50]
    # array containing the variance explained by each
    # principal component (descending order of importance)
    eigenvalues = pca.explained_variance_

    # eigenvectors of the covariance matrix: shape [3,]
    # the principal components: represent the directions
    # of maximum variance in the data, with corresponding
    # eigenvalues indicating the amount of variance
    # explained by each component (ordered from highest to lowest variance)
    eigenvectors = pca.components_

    # correlation loadings: shape [50, 3]
    # the correlations between original (standardized) variables 
    # and each of the three principal components
    # used to interpret the principal components
    corr_loadings_matrix = eigenvectors.T * np.sqrt(eigenvalues)

    # parameter mapping: shape [n, 50]
    # transform each coordinate in PC space to a value in the range [0, 1]
    # representing how much of each of the 50 tags to incorperate in the
    # sonification
    param_mapped = X_reduced.dot(corr_loadings_matrix.T)
    param_map_scaler = MinMaxScaler((0, 1))
    param_map_scaler.fit(param_mapped)
    return lambda pc1, pc2, pc3: param_map_scaler.transform(np.array([pc1, pc2, pc3]).reshape(1,3).dot(corr_loadings_matrix.T))[0]

    #scaled_param_mapped = param_map_scaler.transform(param_mapped)
    #scaled_map_df = pd.DataFrame(scaled_param_mapped, columns=embedding_tags)
    #print(scaled_map_df.head())

    #for col in scaled_map_df.columns:
        #plt.figure()
        #plt.hist(scaled_map_df[col], bins=30, edgecolor='black')
        #plt.title(f"Distribution of {col}")
        #plt.xlabel(col)
        #plt.ylabel("Frequency")
        #plt.show()







