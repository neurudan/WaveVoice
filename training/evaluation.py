from scipy.optimize import brentq
from scipy.interpolate import interp1d

from sklearn.metrics import roc_curve

from keras.models import Model
from keras.callbacks import Callback

from tqdm import tqdm


import numpy as np

import time
import wandb
import math


def cosine_similarity(a, b, sim_model):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def vgg_approach(a, b, sim_model):
    return np.sum(a*b)

def vgg_approach_norm(a, b, sim_model):
    return vgg_approach(np.linalg.norm(a), np.linalg.norm(b), sim_model)

def absolute_difference(a, b, sim_model):
    return 1 - np.sum(np.abs(a - b))

def absolute_difference_norm(a, b, sim_model):
    return absolute_difference(np.linalg.norm(a), np.linalg.norm(b), sim_model)

def sim_model_score(a, b, sim_model):
    a = a.reshape((1,) + a.shape)
    b = b.reshape((1,) + b.shape)
    score = np.asarray(sim_model.predict([a,b]))
    return score[0][0]
    

def calculate_eer(full_model, test_data_handler, sim_model=None):
    print()
    embeddings = {}
    model = Model(inputs=full_model.input,
                  outputs=full_model.layers[-2].output)
                  
    gen = test_data_handler.test_generator()
    times = []
    try:
        while True:
            audio_name, samples = gen.__next__()
            s = time.time()
            embedding = np.asarray(model.predict(np.array(samples)), dtype='float32')
            times.append(time.time() - s)
            embeddings[audio_name] = np.mean(embedding, axis=0)
    except:
        pass
    print(np.mean(times))

    test_lists = test_data_handler.test_lists
    eers = {}
    for list_name in test_lists:
        scores = {'cos_sim': {'method': cosine_similarity, 'scores': []},
                  'vgg': {'method': vgg_approach, 'scores': []},
                  'vgg_norm': {'method': vgg_approach_norm, 'scores': []},
                  'abs_diff': {'method': absolute_difference, 'scores': []},
                  'abs_diff_norm': {'method': absolute_difference_norm, 'scores': []}}
        
        if sim_model is not None:
            scores['sim_model'] = {'method': sim_model_score, 'scores': []}        
        
        true_scores = []
        for (label, file1, file2) in tqdm(test_lists[list_name], ncols=100, ascii=True, desc='compare embeddings'):
            true_scores.append(int(label))
            
            a, b = embeddings[file1], embeddings[file2]
            for k in scores:
                scores[k]['scores'].append(scores[k]['method'](a, b, sim_model))

        print('calculate EER')
        for k in scores:
            fpr, tpr, _ = roc_curve(true_scores, scores[k]['scores'], pos_label=1)
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            eers['EER_'+list_name+'_'+k] = eer
    
    return eers


class EvalCallback(Callback):
    def __init__(self, config, model, test_data_handler):
        self.config = config
        self.model = model
        self.test_data_handler = test_data_handler

    def on_epoch_end(self, epoch, logs):
        epochs_before_eer = self.config.get('TRAINING.epochs_before_eer')
        if epoch % epochs_before_eer == 0:
            eers = calculate_eer(self.model, self.test_data_handler)
            wandb.log(eers, step=epoch + 1)