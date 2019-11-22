from utils.path_handler import get_speaker_list_files, get_dataset_file, get_test_list_files

from multiprocessing import Process, Queue
from tqdm import tqdm

import numpy as np
import h5py
import time
import math


def get_test_list(dataset, test_list):
    file_path = get_test_list_files(dataset)[test_list]
    lines = []
    with open(file_path) as f:
        lines = f.readlines()
    lines = list(set(lines))
    if '\n' in lines:
        lines.remove('\n')
    data = []
    for line in lines:
        if line[-1] == '\n':
            line = line[:-1]
        parts = line.split(' ')
        data.append(parts)
    return data


class TestDataGenerator:
    def __init__(self, config):
        self.config = config
        self.test_dataset = config.get('DATASET.base_test')
        self.data_type = config.get('DATASET.data_type')
        self.receptive_field = self.config.get('MODEL.receptive_field')
        self.test_list = config.get('DATASET.test_list')
        self.test_data = get_test_list(self.test_dataset, self.test_list)
        self.test_statistics = []

        with h5py.File(get_dataset_file(self.test_dataset, self.data_type), 'r') as data:
            files = []
            for (_, file1, file2) in self.test_data:
                files.append(file1)
                files.append(file2)
            files = list(set(files))

            speaker_sorted = {}
            for f in files:
                speaker = f.split('/')[0]
                if speaker not in speaker_sorted:
                    speaker_sorted[speaker] = []
                speaker_sorted[speaker].append(f)

            for speaker in tqdm(speaker_sorted, ncols=100, ascii=True, desc='build test statistics'):
                names = list(data['audio_names/'+speaker])
                for f in speaker_sorted[speaker]:
                    file_name = f.split('/')[1] + '/' + f.split('/')[2]
                    i = names.index(file_name)
                    time = data['statistics/'+speaker][i]
                    n_chunks = math.floor(time / self.receptive_field)
                    end = n_chunks * self.receptive_field
                    if self.data_type == 'mel':
                        end = end * 128
                    self.test_statistics.append((speaker, i, f, end, n_chunks))


    def test_generator(self):
        with h5py.File(get_dataset_file(self.test_dataset, self.data_type), 'r') as data:
            for (speaker, i, audio_name, end, n_chunks) in self.test_statistics:
                samples = np.array(np.split(data['data/' + speaker][i][:end], n_chunks))
                if self.data_type == 'original':
                    samples = samples.reshape((n_chunks, self.receptive_field, 1))
                elif self.data_type == 'mel':
                    samples = samples.reshape((n_chunks, self.receptive_field, 128))
                elif self.data_type == 'ulaw':
                    samples = np.eye(256)[samples]
                yield audio_name, samples