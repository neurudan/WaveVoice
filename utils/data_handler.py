from utils.path_handler import get_speaker_list_files, get_dataset_file

from multiprocessing import Process, Queue

import numpy as np
import h5py
import time
import math


def get_speaker_list(config):
    dataset = config.get('DATASET.base')
    speaker_list = config.get('DATASET.speaker_list')
    
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

class DataGenerator:
    def __init__(self, config):
        self.config = config

        self.dataset = config.get('DATASET.base')
        self.data_type = config.get('DATASET.data_type')
        self.label = config.get('DATASET.label')
        self.condition = config.get('DATASET.condition')
        queue_size = config.get('DATASET.queue_size')
        val_set = config.get('DATASET.val_set')

        self.speakers = get_speaker_list(config)
        self.num_speakers = len(self.speakers)

        if self.label == 'timesteps':
            config.set('DATASET.output_bins', 256)
        elif self.label == 'speaker':
            config.set('DATASET.output_bins', self.num_speakers)
        config.set('DATASET.num_speakers', self.num_speakers)

        self.statistics = {}
        with h5py.File(get_dataset_file(self.dataset, self.data_type), 'r') as data:
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
                self.statistics[speaker] = {'train': train_ids, 'val': val_ids}

        print('done')
        self.train_queue = Queue(queue_size)
        self.val_queue = Queue(queue_size)

        self.enqueuer = Process(target=self.sample_enqueuer)
        self.enqueuer.start()

        self.steps_per_epoch = config.get('TRAINING.steps_per_epoch')
        if self.config.get('DATASET.batch_type') == 'overfit':
            self.steps_per_epoch = math.ceil(self.num_speakers / self.config.get('DATASET.batch_size'))


    def __draw_from_speaker__(self, speaker, receptive_field, dset, data):
        speaker_sample = self.empty_speaker_sample.copy()
        speaker_sample[self.speakers.index(speaker)] = 1

        ids, times = zip(*self.statistics[speaker][dset])
        
        temp_id = np.argmax(np.random.uniform(size=len(times)) * times)
        sample_id = ids[temp_id]

        start_id = np.random.randint(times[temp_id] - receptive_field - 1)
        offset = receptive_field + 1

        sample = None
        if self.data_type == 'mel':
            sample = data['data/' + speaker][sample_id][:,start_id * 128:(start_id + offset) * 128]
            sample = sample.reshape((offset, 128))
        else:
            sample = data['data/' + speaker][sample_id][:,start_id:start_id + offset]

        next_timestep = sample[-1]
        print(next_timestep.shape)
        sample = sample[:-1]
        print(sample.shape)

        if self.data_type == 'ulaw':
            sample = sample.reshape(sample.shape[0], 1)
            temp = self.empty_sample.copy()
            temp[sample] = 1
            sample = temp

            temp = self.empty_timestep.copy()
            temp[next_timestep] = 1
            next_timestep = temp
        return [sample, next_timestep, speaker_sample]


    def __get_batch__(self, batch_size, receptive_field, dset, data, from_all_speakers=False):
        samples = []
        if from_all_speakers:
            for i in range(self.num_speakers):
                speaker = self.speakers[i]
                samples.append(self.__draw_from_speaker__(speaker, receptive_field, dset, data))
        else:
            for _ in range(batch_size):
                speaker = self.speakers[np.argmax(np.random.uniform(size=self.num_speakers))]
                samples.append(self.__draw_from_speaker__(speaker, receptive_field, dset, data))

        samples, timesteps, speaker_samples = zip(*samples)
        samples = np.array(list(samples), dtype='float32')
        if self.data_type == 'original':
            samples = samples.reshape(samples.shape + (1,))
        return samples, np.array(list(timesteps), dtype='float32'), np.array(list(speaker_samples), dtype='float32')

    def sample_enqueuer(self):
        batch_size = self.config.get('DATASET.batch_size')
        receptive_field = self.config.get('MODEL.receptive_field')
        batch_type = self.config.get('DATASET.batch_type')

        self.empty_speaker_sample = np.zeros(self.num_speakers)
        self.empty_sample = np.zeros((receptive_field, 256))
        self.empty_timestep = np.zeros(256)
        
        if batch_type == 'real':
            with h5py.File(get_dataset_file(self.dataset, self.data_type), 'r') as data:
                while True:
                    try:
                        if self.train_queue.qsize() > self.val_queue.qsize():
                            samples, timesteps, speaker_samples = self.__get_batch__(batch_size, receptive_field, 'val', data)
                            self.val_queue.put([samples, timesteps, speaker_samples], timeout=0.5)
                        else:
                            samples, timesteps, speaker_samples = self.__get_batch__(batch_size, receptive_field, 'train', data)
                            self.train_queue.put([samples, timesteps, speaker_samples], timeout=0.5)
                    except Exception as e:
                        print(e)
        elif batch_type == 'zeros':
            # generate zero samples (inspired by karpathys blog for testing)
            empty_timesteps = np.zeros((batch_size, 256))
            empty_speaker_samples = np.zeros((batch_size, self.num_speakers))
            empty_samples = None
            if self.data_type == 'original':
                empty_samples = np.zeros((batch_size, receptive_field, 1))
            elif self.data_type == 'ulaw':
                empty_samples = np.zeros((batch_size, receptive_field, 256))
            elif self.data_type == 'mel':
                empty_samples = np.zeros((batch_size, receptive_field, 128))
            while True:
                if self.label == 'speaker':
                    speaker_samples = empty_speaker_samples.copy()
                    for i in range(batch_size):
                        speaker_samples[i, np.random.randint(self.num_speakers)] = 1
                    try:
                        if self.train_queue.qsize() > self.val_queue.qsize():
                            self.val_queue.put([empty_samples, empty_timesteps, speaker_samples], timeout=0.5)
                        else:
                            self.train_queue.put([empty_samples, empty_timesteps, speaker_samples], timeout=0.5)
                    except:
                        pass
                elif self.label == 'timesteps':
                    timesteps = empty_timesteps.copy()
                    for i in range(batch_size):
                        timesteps[i, np.random.randint(256)] = 1
                    try:
                        if self.train_queue.qsize() > self.val_queue.qsize():
                            self.val_queue.put([empty_samples, empty_timesteps, speaker_samples], timeout=0.5)
                        else:
                            self.train_queue.put([empty_samples, empty_timesteps, speaker_samples], timeout=0.5)
                    except:
                        pass
        elif batch_type == 'overfit':
            # generates a single batch and always yields this batch to overfit on it (inspired by karpathys blog for testing)
            samples_o, timesteps_o, speaker_samples_o = None, None, None
            with h5py.File(get_dataset_file(self.dataset, self.data_type), 'r') as data:
                samples_o, timesteps_o, speaker_samples_o = self.__get_batch__(batch_size, receptive_field, 'train', data, from_all_speakers=True)
            
            i = 0
            while True:
                samples, timesteps, speaker_samples = samples_o[i:i+batch_size], timesteps_o[i:i+batch_size], speaker_samples_o[i:i+batch_size]
                i += batch_size
                if i >= len(samples_o):
                    i = 0
                try:
                    if self.train_queue.qsize() > self.val_queue.qsize():
                        self.val_queue.put([samples, timesteps, speaker_samples], timeout=0.5)
                    else:
                        self.train_queue.put([samples, timesteps, speaker_samples], timeout=0.5)
                except:
                    pass


    def terminate_enqueuer(self):
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
