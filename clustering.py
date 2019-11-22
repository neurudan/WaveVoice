from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from sklearn.metrics import roc_curve

from keras.models import Model
from keras.callbacks import Callback

import numpy as np

import time
import wandb


class ClusterCallback(Callback):
    def __init__(self, config, model, data_handler):
        self.config = config
        self.model = model
        self.data_handler = data_handler

    def on_epoch_end(self, epoch, logs):
        if epoch % 1 == 0:
            start = time.time()
            embeddings = {}
            model = Model(inputs=self.model.input, outputs=self.model.layers[-3].output)
            gen = self.data_handler.test_generator()
            try:
                while True:
                    audio_name, samples = gen.__next__()
                    embedding = np.asarray(model.predict(np.array(samples)))
                    embeddings[audio_name] = np.mean(embedding, axis=0)
            except:
                pass
            
            scores = []
            true_scores = []
            for (label, file1, file2) in self.data_handler.test_data:
                true_scores.append(int(label))
                a = np.array([embeddings[file1], embeddings[file2]])
                print(a)
                print(a.shape, flush=True)
                scores.append(cluster_embeddings(np.array([embeddings[file1], embeddings[file2]])))

            fpr, tpr, thresholds = roc_curve(true_scores, scores, pos_label=1)
            eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            thresh = interp1d(fpr, thresholds)(eer)
            wandb.log({'EER': eer, 'thresh': thresh}, step=epoch)
            print(time.time() - start)


def cluster_embeddings(embeddings, metric='cosine', method='complete'):
    embeddings_distance = cdist(embeddings, embeddings, 'cosine')
    embeddings_linkage = linkage(embeddings_distance, 'complete', 'cosine')

    thresholds = embeddings_linkage[:, 2]
    predicted_clusters = []
    print(thresholds)
    print(len(thresholds))
    predicted_cluster = fcluster(embeddings_linkage, threshold[0], 'distance')
    print(predicted_cluster)
    if predicted_cluster[0] == predicted_cluster[1]:
        return 1
    return 0
