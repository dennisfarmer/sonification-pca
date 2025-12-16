# Converting Spectrograms to Embeddings - MusiCNN

see `musicnn_paper.pdf`

```python
#!/usr/bin/env python

# pip install essentia-tensorflow==2.1b6.dev1389
import essentia.standard as es

def compute_audio_embedding(audio_path: str):
    MUSICNN_SR = 16000
    try:
        audio = es.MonoLoader(filename=audio_path, sampleRate=MUSICNN_SR)()
        musicnn_emb = es.TensorflowPredictMusiCNN(graphFilename='msd-musicnn-1.pb', output='model/dense_1/BiasAdd')(audio)
        mean_emb = np.mean(musicnn_emb, axis=0)
        mean_emb = mean_emb[np.newaxis, :]
    except:
        return None
    return mean_emb[0]
```