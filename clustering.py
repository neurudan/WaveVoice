from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from sklearn.metrics import roc_curve

from keras.models import Model
from keras.callbacks import Callback

import numpy as np


class ClusterCallback(Callback):
    def __init__(self, config, model, data_generator):
        self.config = config
        self.model = model
        self.data_generator = data_generator

    def on_epoch_end(self, epoch, logs):
        print('finito')
        if epoch % 1 == 0:
            print('finito')
            y = []
            y_score = []

            model = Model(inputs=self.model.input, outputs=self.model.layers[-3].output)
            
            print('finito')
            try:
                while True:
                    x1, x2, y = self.data_generator.__next__()
                    print(y)
                    print(x1)
                    embeddings = np.asarray(model.predict(x1))
                    print(embeddings)
                    print('ending')
                    speakers.extend(y)
                    print()
            except:
                pass
            fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
            eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            thresh = interp1d(fpr, thresholds)(eer)


def cluster_embeddings(embeddings, set_of_true_clusters, metric='cosine', method='complete'):
    embeddings_distance = cdist(embeddings, embeddings, 'cosine')
    embeddings_linkage = linkage(embeddings_distance, 'complete', 'cosine')

    thresholds = embeddings_linkage[:, 2]
    print(thresholds)
    predicted_clusters = []

    for threshold in thresholds:
        predicted_cluster = fcluster(embeddings_linkage, threshold, 'distance')
        predicted_clusters.append(predicted_cluster)
    
    print(predicted_clusters)
