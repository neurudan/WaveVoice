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
    def __init__(self, dataset, sequence_length, batch_size, speakers, queue_size,
                 use_ulaw=False, val_set=0.2):
        self.speakers = speakers
        self.num_speakers = len(speakers)
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.use_ulaw = use_ulaw
        self.statistics = {}
        with h5py.File(get_dataset_file(dataset, use_ulaw), 'r') as data:
            for speaker in speakers:
                times = data['statistics/'+speaker][:]

                id_times = list(zip(np.arange(len(times)), times))
                id_times.sort(key=lambda x: x[1])

                total_time = np.sum(times)
                train_ids = []
                val_time = 0
                val_ids = []
                for id, time in id_times:
                    if time + val_time < total_time * val_set:
                        val_ids.append((id, time))
                        val_time += time
                    else:
                        train_ids.append((id, time))
                self.statistics[speaker] = {'train':train_ids, 'val':val_ids}

        self.train_queue = Queue(queue_size)
        self.val_queue = Queue(queue_size)

        self.train_enqueuer = Process(target=self.sample_enqueuer, args=('train',))
        self.train_enqueuer.start()

        self.val_enqueuer = Process(target=self.sample_enqueuer, args=('val',))
        self.val_enqueuer.start()

    def calculate_speaker_statistics(self, speakers):
        statistics = []
        with h5py.File(get_dataset_file(dataset, use_ulaw), 'r') as data:
            for speaker in speakers:
                statistics.append([speaker, np.sum(data['statistics/'+speaker][:])])
        return statistics

    def sample_enqueuer(self, set):
        empty_label = np.zeros(self.num_speakers)
        empty_sample = np.zeros((self.sequence_length, 256))
        while True:
            samples = []
            labels = []
            for i in range(self.batch_size):
                speaker = self.speakers[np.argmax(np.random.uniform(size=self.num_speakers))]
                label = empty_label.copy()
                label[self.speakers.index(speaker)] = 1

                ids, times = zip(*self.statistics[speaker][set])
                temp_id = np.argmax(np.random.uniform(size=len(times)) * times)
                sample_id = ids[temp_id]

                start_id = np.random.randint(times[temp_id] - self.sequence_length)
                sample = None

                with h5py.File(get_dataset_file(self.dataset, self.use_ulaw), 'r') as data:
                    sample = data['data/'+speaker][sample_id][start_id:start_id+self.sequence_length]

                if self.use_ulaw:
                    sample = sample.reshape(sample.shape[0], 1)
                    new_sample = empty_sample.copy()
                    new_sample[sample] = 1
                    sample = new_sample

                samples.append(sample)
                labels.append(label)
            samples = np.array(samples)

            if not self.use_ulaw:
                shape = samples.shape
                samples = samples.reshape(shape[0], shape[1], 1)

            if set == 'val':
                self.val_queue.put([samples, np.array(labels)])
            elif set == 'train':
                self.train_queue.put([samples, np.array(labels)])

    def terminate_queue(self):
        self.train_enqueuer.terminate()
        self.val_enqueuer.terminate()

    def train_batch_generator(self):
        while True:
            [samples, labels] = self.train_queue.get()
            yield samples, labels

    def val_batch_generator(self):
        while True:
            [samples, labels] = self.val_queue.get()
            yield samples, labels
