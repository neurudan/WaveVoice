from utils.path_handler import get_speaker_list_files, get_dataset_file, get_test_list_files

from multiprocessing import Process, Queue
from tqdm import tqdm

import numpy as np
import h5py
import time
import math


def get_speaker_list(dataset, speaker_list):
    file_path = get_speaker_list_files(dataset)[speaker_list]
    lines = []
    with open(file_path) as f:
        lines = f.readlines()
    lines = list(set(lines))
    if '\n' in lines:
        lines.remove('\n')
    speakers = []
    for line in lines:
        if line[-1] == '\n':
            line = line[:-1]
        speakers.append(line)
    return speakers

class SimDataGenerator:
    def __init__(self, config, embedding_model):
        self.config = config
        self.embedding_model = embedding_model

        self.dataset = config.get('DATASET.base')
        self.data_type = config.get('DATASET.data_type')
        self.test_single = config.get('DATASET.test_single')
        self.batch_size = config.get('DATASET.batch_size')
        self.val_set = config.get('DATASET.val_set')
        self.val_part = config.get('DATASET.val_part')

        self.steps_per_epoch = config.get('TRAINING.steps_per_epoch')

        self.receptive_field = config.get('MODEL.receptive_field')


        self.train_speakers = get_speaker_list(self.dataset, config.get('DATASET.speaker_list'))
        self.num_speakers = len(self.train_speakers)


        self.statistics = {}

        with h5py.File(get_dataset_file(self.dataset, self.data_type), 'r') as data:
            for speaker in tqdm(self.train_speakers, ncols=100, ascii=True, desc='build speaker statistics'):

                train_ids = {}
                val_ids = {}
                for i, time in enumerate(data['statistics/'+speaker][:]):
                    val_time = int(time * self.val_set)
                    if val_time < self.receptive_field:
                        val_time = self.receptive_field
                    if time - val_time >= self.receptive_field:

                        if self.val_part == 'before':
                            val_ids[i] = self.__get_sample_information__(0, val_time)
                            train_ids[i] = self.__get_sample_information__(val_time, time - val_time)
                        elif self.val_part == 'after':
                            val_ids[i] = self.__get_sample_information__(time - val_time, time)
                            train_ids[i] = self.__get_sample_information__(0, time - val_time)
                sample_ids = list(train_ids.keys())
                num_samples = len(sample_ids)
                self.statistics[speaker] = {'train': train_ids, 'val': val_ids, 'sample_ids': sample_ids, 'num_samples': num_samples}
        self.close = False


    def __get_sample_information__(self, start, end):
        time = end - start
        if self.test_single:
            return [start, end, 0]
        else:
            n_chunks = math.floor(time / self.receptive_field)
        offset = n_chunks * self.receptive_field
        end = start + offset
        return [start, end, n_chunks]


    def __draw_sample__(self, data, set, speaker):
        sample_id = self.statistics[speaker]['sample_ids'][np.random.randint(self.statistics[speaker]['num_samples'])]
        [start, end, n_chunks] = self.statistics[speaker][set][sample_id]
        if n_chunks == 0:
            start = np.random.randint(end - start - self.receptive_field)
            end = start + self.receptive_field
            n_chunks = 1
        if self.data_type == 'mel':
            start *= 128
            end *= 128
        samples = np.split(data['data/' + speaker][sample_id][start:end], n_chunks)
        return samples, n_chunks
        

    def batch_generator(self, set):
        with h5py.File(get_dataset_file(self.dataset, self.data_type), 'r') as data:
            while not self.close:
                samples_1 = []
                samples_2 = []
                chunks_1 = [0]
                chunks_2 = [0]
                labels = []
                for _ in range(self.batch_size // 2):
                    speaker = self.train_speakers[np.random.randint(self.num_speakers)]
                    s, c = self.__draw_sample__(data, set, speaker)
                    samples_1.extend(s)
                    chunks_1.append(chunks_1[-1] + c)

                    s, c = self.__draw_sample__(data, set, speaker)
                    samples_2.extend(s)
                    chunks_2.append(chunks_2[-1] + c)

                    speaker_1 = np.random.randint(self.num_speakers)
                    speaker_2 = np.random.randint(self.num_speakers)
                    while speaker_1 == speaker_2:
                        speaker_2 = np.random.randint(self.num_speakers)
                    s, c = self.__draw_sample__(data, set, self.train_speakers[speaker_1])
                    samples_1.extend(s)
                    chunks_1.append(chunks_1[-1] + c)

                    s, c = self.__draw_sample__(data, set, self.train_speakers[speaker_2])
                    samples_2.extend(s)
                    chunks_2.append(chunks_2[-1] + c)

                samples_1.extend(samples_2)
                samples = np.array(samples)

                if self.data_type == 'original':
                    samples = samples.reshape((len(samples), self.receptive_field, 1))
                elif self.data_type == 'mel':
                    samples = samples.reshape((len(samples), self.receptive_field, 128))
                elif self.data_type == 'ulaw':
                    samples = np.eye(256)[samples]
                
                samples = np.asarray(self.embedding_model.predict(samples))

                samples = np.split(samples_1, chunks_1)
                samples_1 = samples[1:-1]
                samples_2 = np.split(samples[-1], chunks_2)[1:-1]

                samples_1 = [np.mean(x, axis=0) for x in samples_1]
                samples_2 = [np.mean(x, axis=0) for x in samples_2]

                yield [np.array(samples_1), np.array(samples_2)], np.array(labels)


    def get_generator(self, generator):
        gen = self.batch_generator(generator)
        gen.__next__()
        return gen

    def terminate_generator(self, generator):
        self.close = True
        try:
            while True:
                generator.__next__()
        except:
            pass
