from utils.path_handler import get_speaker_list_files, get_dataset_file

from multiprocessing import Process, Queue

import numpy as np
import h5py
import time


class DataGenerator:
    def __init__(self, config):
        self.config = config

        self.dataset = config.get('DATASET.base')
        self.use_ulaw = config.get('DATASET.use_ulaw')
        self.label = config.get('DATASET.label')
        self.condition = config.get('DATASET.condition')
        queue_size = config.get('DATASET.queue_size')
        val_set = config.get('DATASET.val_set')
        speaker_list = config.get('DATASET.speaker_list')


        file_path = get_speaker_list_files(self.dataset)[speaker_list]
        lines = []
        with open(file_path) as f:
            lines = f.readlines()
        lines = list(set(lines))
        if '\n' in lines:
            lines.remove('\n')
        self.speakers = []
        for line in lines:
            if line[-1] == '\n':
                line = line[:-1]
            self.speakers.append(line)
        self.num_speakers = len(self.speakers)


        if self.label == 'timestep':
            config.set('DATASET.output_bins', 256)
        elif self.label == 'speaker':
            config.set('DATASET.output_bins', self.num_speakers)        
        config.set('DATASET.num_speakers', self.num_speakers)


        self.statistics = {}
        with h5py.File(get_dataset_file(self.dataset, self.use_ulaw), 'r') as data:
            for speaker in self.speakers:
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


        self.enqueuer = Process(target=self.sample_enqueuer)
        self.enqueuer.start()


    def sample_enqueuer(self):
        batch_size = self.config.get('DATASET.batch_size')
        receptive_field = self.config.get('MODEL.receptive_field')

        empty_speaker_sample = np.zeros(self.num_speakers)
        empty_sample = np.zeros((receptive_field, 256))
        empty_timestep = np.zeros(256)
        
        with h5py.File(get_dataset_file(self.dataset, self.use_ulaw), 'r') as data:
            while True:
                samples = []
                speaker_samples = []
                timesteps = []
                set = 'train'
                if self.train_queue.qsize() > self.val_queue.qsize():
                    set = 'val'
                for _ in range(batch_size):
                    speaker = self.speakers[np.argmax(np.random.uniform(size=self.num_speakers))]
                    speaker_sample = empty_speaker_sample.copy()
                    speaker_sample[self.speakers.index(speaker)] = 1

                    ids, times = zip(*self.statistics[speaker][set])
                    temp_id = np.argmax(np.random.uniform(size=len(times)) * times)
                    sample_id = ids[temp_id]

                    start_id = np.random.randint(times[temp_id] - receptive_field - 1)
                    sample = data['data/'+speaker][sample_id][start_id:start_id + receptive_field + 1]

                    next_timestep = sample[-1]
                    sample = sample[:-1]

                    if self.use_ulaw:
                        sample = sample.reshape(sample.shape[0], 1)
                        temp = empty_sample.copy()
                        temp[sample] = 1
                        sample = temp

                        temp = empty_timestep.copy()
                        temp[next_timestep] = 1
                        next_timestep = temp

                    samples.append(sample)
                    speaker_samples.append(speaker_sample)
                    timesteps.append(next_timestep)

                try:
                    if set == 'val':
                        self.val_queue.put([np.array(samples), np.array(timesteps), np.array(speaker_samples)], timeout=0.5)
                    elif set == 'train':
                        self.train_queue.put([np.array(samples), np.array(timesteps), np.array(speaker_samples)], timeout=0.5)
                except:
                    pass


    def terminate_queue(self):
        self.enqueuer.terminate()


    def get_generator(self, generator):
        generators = {'train': self.train_batch_generator(),
                      'val': self.val_batch_generator()}
        generators[generator].__next__()
        return generators[generator]


    def train_batch_generator(self):
        while True:
            [samples, timesteps, speaker_samples] = self.train_queue.get()
            
            label = None
            if self.label == 'speaker':
                label = speaker_samples
            elif self.label == 'timesteps':
                label = timesteps
            
            input = None
            if self.condition == 'none':
                input = [samples]
            elif self.condition == 'speaker':
                input = [samples, speaker_samples]
            
            yield input, label


    def val_batch_generator(self):
        while True:
            [samples, timesteps, speaker_samples] = self.val_queue.get()

            label = None
            if self.label == 'speaker':
                label = speaker_samples
            elif self.label == 'timesteps':
                label = timesteps
            
            input = None
            if self.condition == 'none':
                input = [samples]
            elif self.condition == 'speaker':
                input = [samples, speaker_samples]

            yield input, label