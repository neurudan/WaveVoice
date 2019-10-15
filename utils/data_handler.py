from utils.path_handler import get_speaker_list_files, get_dataset_file

from multiprocessing import Process, Queue

import numpy as np
import h5py
import time


def get_speakers(dataset, speaker_list):
    file = get_speaker_list_files(dataset)[speaker_list]
    lines = []
    with open(file) as f:
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

class DataGenerator:
    def __init__(self, dataset, sequence_length, batch_size, speakers,
                 use_ulaw=False):
        self.speakers = speakers
        self.num_speakers = len(speakers)
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.data = h5py.File(get_dataset_file(dataset, use_ulaw), 'r')

    def calculate_speaker_statistics(self, speakers):
        statistics = []
        for speaker in speakers:
            statistics.append([speaker, np.sum(self.data['statistics/'+speaker][:])])
        return statistics

    def sample_enqueuer(self, queue):
        statistics = {}
        empty_label = np.zeros(self.num_speakers)
        for speaker in self.speakers:
            statistics[speaker] = self.data['statistics/'+speaker][:]
            statistics[speaker+'_size'] = len(self.data['statistics/'+speaker][:])
        while True:
            samples = []
            labels = []
            for i in range(self.batch_size):
                speaker = self.speakers[np.argmax(np.random.uniform(size=self.num_speakers))]
                label = empty_label.copy()
                label[self.speakers.index(speaker)] = 1
                sample_id = np.argmax(np.random.uniform(size=statistics[speaker+'_size']) * statistics[speaker])
                start_id = np.random.randint(statistics[speaker][sample_id] - self.sequence_length)
                samples.append(self.data['data/'+speaker][sample_id][start_id:start_id+self.sequence_length])
                labels.append(label)
            samples = np.array(samples)
            shape = samples.shape
            queue.put([samples.reshape(shape[0], shape[1], 1), np.array(labels)])

    def terminate_queue(self):
        self.sample_enqueuer.terminate()

    def batch_generator(self):
        sample_queue = Queue(50)
        self.sample_enqueuer = Process(target=self.sample_enqueuer, args=(sample_queue,))
        self.sample_enqueuer.start()
        while True:
            [samples, labels] = sample_queue.get()
            yield samples, labels
