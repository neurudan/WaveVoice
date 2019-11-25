from scipy.optimize import brentq
from scipy.interpolate import interp1d

from sklearn.metrics import roc_curve

from keras.models import Model
from keras.callbacks import Callback

from tqdm import tqdm


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

    scores1 = []
    scores2 = []
    scores3 = []
    true_scores = []
    for (label, file1, file2) in tqdm(test_data_handler.test_data, ncols=100, ascii=True, desc='compare embeddings'):
        true_scores.append(int(label))
        scores1.append(np.sum(embeddings[file1]*embeddings[file2]))
        e = np.abs(embeddings[file1] - embeddings[file2])
        scores2.append(1 - np.sum(e))
        scores3.append(1 - np.sum(e) ** 2)

    print('calculate EER')
    fpr, tpr, _ = roc_curve(true_scores, scores1, pos_label=1)
    eer1 = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    fpr, tpr, _ = roc_curve(true_scores, scores2, pos_label=1)
    eer2 = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    fpr, tpr, _ = roc_curve(true_scores, scores3, pos_label=1)
    eer3 = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer1, eer2, eer3


class ClusterCallback(Callback):
    def __init__(self, config, model, test_data_handler):
        self.config = config
        self.model = model
        self.test_data_handler = test_data_handler

    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            eer1, eer2, eer3 = calculate_eer(self.model, self.test_data_handler)
            wandb.log({'EER1': eer1, 'EER2': eer2, 'EER3': eer3},
                      step=epoch + 1)