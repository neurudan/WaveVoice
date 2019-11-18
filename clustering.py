from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist
from keras.models import Model

import numpy as np

import keras.callbacks.callbacks.Callback


class ClusterCallback(keras.callbacks.callbacks.Callback):
    def __init__(self, config, model, data_generator):
        self.config = config
        self.model = model
        self.data_generator = data_generator

    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            embeddings = []
            speakers = []
            model = Model(inputs=self.model.input, outputs=self.model.layers[-2].output)
            while not self.data_generator.finished:
                x, y = self.data_generator.__next__()
                embeddings = np.asarray(model.predict(x))
                speakers.extend(y)


def cluster_embeddings(set_of_embeddings, set_of_true_clusters, metric='cosine', method='complete'):
    set_predicted_clusters = []

    for embeddings, true_clusters in zip(set_of_embeddings, set_of_true_clusters):
        embeddings_distance = cdist(embeddings, embeddings, metric)
        embeddings_linkage = linkage(embeddings_distance, method, metric)

        thresholds = embeddings_linkage[:, 2]
        predicted_clusters = []

        for threshold in thresholds:
            predicted_cluster = fcluster(embeddings_linkage, threshold, 'distance')
            predicted_clusters.append(predicted_cluster)

        set_predicted_clusters.append(predicted_clusters)

    num_speakers = len(set(speakers_inputs[0]))

def generate_embeddings(outputs, speakers_inputs, vector_size):
    num_speakers = len(set(speakers_inputs[0]))
    num_embeddings = len(outputs) * num_speakers

    all_embeddings = []
    all_speakers = []

    for output, speakers_input in zip(outputs, speakers_inputs):
        embeddings = np.zeros((num_speakers, vector_size))
        speakers = set(speakers_input)

        for i in range(0, num_speakers):
            utterance = embeddings[i]

            indices = np.where(speakers_input == i)[0]
            outputs = np.take(output, indices, axis=0)
            for value in outputs:
                utterance = np.add(utterance, value)

            embeddings[i] = np.divide(utterance, len(outputs))

        all_embeddings.extend(embeddings)
        all_speakers.extend(speakers)

    return all_embeddings, all_speakers, num_embeddings