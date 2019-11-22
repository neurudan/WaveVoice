from scipy.optimize import brentq
from scipy.interpolate import interp1d

from sklearn.metrics import roc_curve

from keras.models import Model
from keras.callbacks import Callback

import numpy as np

import time
import wandb


def calculate_eer(full_model, test_data_handler):
    embeddings = {}
    model = Model(inputs=full_model.input,
                  outputs=full_model.layers[-2].output)
                  
    gen = test_data_handler.test_generator()
    try:
        while True:
            audio_name, samples = gen.__next__()
            embedding = np.asarray(model.predict(np.array(samples)))
            embeddings[audio_name] = np.linalg.norm(np.mean(embedding, axis=0))
    except:
        pass

    scores = []
    true_scores = []
    for (label, file1, file2) in test_data_handler.test_data:
        true_scores.append(int(label))
        scores.append(np.sum(embeddings[file1]*embeddings[file2]))

    fpr, tpr, thresholds = roc_curve(true_scores, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


class ClusterCallback(Callback):
    def __init__(self, config, model, test_data_handler):
        self.config = config
        self.model = model
        self.test_data_handler = test_data_handler

    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            eer, thresh = calculate_eer(self.model, self.test_data_handler)
            wandb.log({'EER': eer,
                       'thresh': thresh},
                      step=epoch + 1)