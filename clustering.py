from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from sklearn.metrics import roc_curve

from keras.models import Model
from keras.callbacks import Callback

import numpy as np
import time


class ClusterCallback(Callback):
    def __init__(self, config, model, data_generator):
        self.config = config
        self.model = model
        self.data_generator = data_generator

    def on_epoch_end(self, epoch, logs):
        if epoch % 1 == 0:
            start = time.time()
            embeddings = {}
            model = Model(inputs=self.model.input, outputs=self.model.layers[-3].output)
            gen = self.data_generator()
            try:
                while True:
                    audio_name, samples = gen.__next__()
                    embedding = np.asarray(model.predict(np.array(samples)))
                    embeddings[audio_name] = np.mean(embedding, axis=0)
            except:
                pass
            print(time.time() - start)


def cluster_embeddings(embeddings, metric='cosine', method='complete'):
    embeddings_distance = cdist(embeddings, embeddings, 'cosine')
    embeddings_linkage = linkage(embeddings_distance, 'complete', 'cosine')

    thresholds = embeddings_linkage[:, 2]
    print(thresholds)
    predicted_clusters = []

    for threshold in thresholds:
        predicted_cluster = fcluster(embeddings_linkage, threshold, 'distance')
        predicted_clusters.append(predicted_cluster)
    
    print(predicted_clusters)
