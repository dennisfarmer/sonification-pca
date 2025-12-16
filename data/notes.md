finished creating embeddings function

Features to implement:
- clickable 3D PCA plot that triggers ChucK stuff when clicked
- on hover, show track name
- add ability to upload audio, refreshes plot and makes newly added points and makes them larger
- toggle the nine different colorbars with clickable buttons

ChucK features:
- decide whether to have infinite loop, or 5 seconds with decay on click
- create parameter mapping super simple and intuitive, based on analysis of colors
    - mapping principal components to sonification parameters



todo: 







---

speaking dates, based on specified 

interface to specify:
- timestep
- start date
- variables to visualize
- variables to sonify
- sonification parameter mapping

chuck -s renders quickly
(silent disables real-time audio)


Two forms of functionality:

1: scientific sonification:
- UI that allows uploading a csv, selecting params
    to visualize and sonify, and then exports a config
    in the form of a pickle file to then sonify using ChucK
- Issues:
    - Best way to structure the intermediate between web
        interface and ChucK? Would be nice to be pure ChucK?

2: creative musicification of Music Recommender System PCA
- map ChucK parameters to represent different attributes
    based on how PCA and paramset represent
- key idea: provide explainability to PCA
- requires that I recreate the pipeline that converts music tracks to coordinates in PCA space. This will require a massive music dataset that aligns with a Spotify dataset
    - core issue I ran into last time: metric computation
        is not reproducable, since Spotify keeps their methods
        secret.
    - original idea was to try to create a model that predicts
        metrics based on a training set; that seems really
        difficult to do reliably
    - better idea: 
        1) compute PCA of spotify data, 
            based on musicnn embeddings
        2) analyze correlations between PCA dimensions and
            Spotify metrics
        3) use analysis to create fixed parameter mapping 
            between PCA coordinates and sonification parameters
            - especially bpm; perform data analysis on song tempo range as a function of PCA dimension that most strongly captures bpm, check for homoskedasticity
            - limit to three principal components, but try to
                generalize to more than three if it is possible
        4) use this fixed parameter mapping to analyze 
            new songs that are uploaded


The idea is then to provide a toolkit that can be used to perform sonification of principal component analysis in the future, being as interpretable as possible. Each PC is some repeating sound or aspect of the sound, and as the respective PC is altered, the sound changes. Each dimension can affect the other dimensions, especially with bpm (seems like a good starting point)


