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

class TrainDataGenerator:
    def __init__(self, config, val_active, train_speakers=None):
        self.config = config
        self.train_speakers = train_speakers
        self.val_active = val_active

        self.dataset = config.get('DATASET.base')
        self.data_type = config.get('DATASET.data_type')
        self.label = config.get('DATASET.label')
        self.condition = config.get('DATASET.condition')

        val_set = config.get('DATASET.val_set')
        val_part = config.get('DATASET.val_part')
        speaker_list = config.get('DATASET.speaker_list')
        receptive_field = self.config.get('MODEL.receptive_field')

        if self.train_speakers is None:
            self.train_speakers = get_speaker_list(self.dataset, speaker_list)
        self.num_speakers = len(self.train_speakers)

        if self.label in ['single_timestep', 'all_timesteps']:
            config.set('MODEL.output_bins', 256)
        elif self.label == 'speaker':
            config.set('MODEL.output_bins', self.num_speakers)
        config.set('DATASET.num_speakers', self.num_speakers)

        self.statistics = {}

        with h5py.File(get_dataset_file(self.dataset, self.data_type), 'r') as data:
            if self.val_active:
                if val_part == 'overall':
                    for speaker in tqdm(self.train_speakers, ncols=100, ascii=True, desc='build speaker statistics'):
                        times = data['statistics/'+speaker][:]
                        id_times = list(zip(np.arange(len(times)), times))
                        id_times.sort(key=lambda x: x[1])
                        total_time = np.sum(times)
                        train_ids = []
                        val_time = 0
                        val_ids = []
                        for id, time in id_times:
                            if time + val_time < total_time * val_set:
                                val_ids.append((id, 0, time))
                                val_time += time
                            else:
                                train_ids.append((id, 0, time))
                        self.statistics[speaker] = {'train': train_ids, 'val': val_ids}
                else:
                    for speaker in tqdm(self.train_speakers, ncols=100, ascii=True, desc='build speaker statistics'):
                        train_ids = []
                        val_ids = []
                        for i, time in enumerate(data['statistics/'+speaker][:]):
                            if time > receptive_field:
                                val_time = int(time * val_set)
                                if val_time < receptive_field + 2:
                                    val_time = receptive_field + 2
                                if time - val_time >= receptive_field + 2:
                                    if val_part == 'before':
                                        val_ids.append((i, 0, val_time))
                                        train_ids.append((i, val_time, time - val_time))
                                    elif val_part == 'after':
                                        val_ids.append((i, time - val_time, val_time))
                                        train_ids.append((i, 0, time - val_time))
                                else:
                                    train_ids.append((i, 0, time))
                        self.statistics[speaker] = {'train': train_ids, 'val': val_ids}
            else:
                for speaker in tqdm(self.train_speakers, ncols=100, ascii=True, desc='build speaker statistics'):
                    train_ids = []
                    val_ids = []
                    for i, time in enumerate(data['statistics/'+speaker][:]):
                        train_ids.append((i, 0, time))
                    self.statistics[speaker] = {'train': train_ids, 'val': val_ids}
        
        self.start_enqueuer()

        self.steps_per_epoch = config.get('TRAINING.steps_per_epoch')
        if self.config.get('DATASET.batch_type') == 'overfit':
            self.steps_per_epoch = math.ceil(self.num_speakers / self.config.get('DATASET.batch_size'))


    def __read_sample__(self, speaker, sample_id, start_id, receptive_field, data):
        offset = receptive_field + 1 # To get next timestep aswell

        sample = None
        if self.data_type == 'mel':
            sample = data['data/' + speaker][sample_id][start_id * 128:(start_id + offset) * 128]
            sample = sample.reshape((offset, 128))
        elif self.data_type == 'original':
            sample = data['data/' + speaker][sample_id][start_id:start_id + offset]
        if self.data_type == 'ulaw':
            sample = data['data/' + speaker][sample_id][start_id:start_id + offset]
            sample = np.eye(256)[sample]

        next_timestep = None
        if self.label == 'single_timestep':
            next_timestep = sample[-1]
        elif self.label == 'all_timesteps':
            next_timestep = sample[1:]

        return sample[:-1], next_timestep


    def __draw_from_speaker__(self, speaker, receptive_field, dset, data):
        speaker_sample = self.empty_speaker_sample.copy()
        speaker_sample[self.train_speakers.index(speaker)] = 1

        ids, offsets, times = zip(*self.statistics[speaker][dset])
        
        temp_id = np.argmax(np.random.uniform(size=len(times)) * times)
        sample_id = ids[temp_id]
        
        start_id = np.random.randint(times[temp_id] - receptive_field - 1) + offsets[temp_id]
        sample, next_timestep = self.__read_sample__(speaker, sample_id, start_id, receptive_field, data)

        return [sample, next_timestep, speaker_sample]


    def __get_batch__(self, batch_size, receptive_field, dset, data, from_all_speakers=False):
        samples = []
        if from_all_speakers:
            for i in range(self.num_speakers):
                samples.append(self.__draw_from_speaker__(self.train_speakers[i], receptive_field, dset, data))
        else:
            speaker_ids = np.random.randint(self.num_speakers, size=batch_size)
            for speaker_id in speaker_ids:
                samples.append(self.__draw_from_speaker__(self.train_speakers[speaker_id], receptive_field, dset, data))
            
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
                if self.val_active:
                    while not self.exit_process:
                        try:
                            if self.train_queue.qsize() > self.val_queue.qsize():
                                samples, timesteps, speaker_samples = self.__get_batch__(batch_size, receptive_field, 'val', data)
                                self.val_queue.put([samples, timesteps, speaker_samples], timeout=0.5)
                            else:
                                samples, timesteps, speaker_samples = self.__get_batch__(batch_size, receptive_field, 'train', data)
                                self.train_queue.put([samples, timesteps, speaker_samples], timeout=0.5)
                        except:
                            pass
                else: 
                    while not self.exit_process:
                        try:
                            samples, timesteps, speaker_samples = self.__get_batch__(batch_size, receptive_field, 'train', data)
                            self.train_queue.put([samples, timesteps, speaker_samples], timeout=0.5)
                        except:
                            pass
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
            while not self.exit_process:
                result = None
                if self.label == 'speaker':
                    speaker_samples = np.eye(self.num_speakers)[np.random.randint(self.num_speakers, size=(batch_size, self.num_speakers))]
                    result = [empty_samples, empty_timesteps, speaker_samples]
                elif self.label == 'single_timestep':
                    timesteps = np.eye(256)[np.random.randint(256, size=(batch_size, 256))]
                    result = [empty_samples, timesteps, empty_speaker_samples]
                elif self.label == 'all_timesteps':
                    timesteps = np.eye(256)[np.random.randint(256, size=(batch_size, receptive_field, 256))]
                    result = [empty_samples, timesteps, empty_speaker_samples]

                try:
                    if self.train_queue.qsize() > self.val_queue.qsize():
                        self.val_queue.put(result, timeout=0.5)
                    else:
                        self.train_queue.put(result, timeout=0.5)
                except:
                    pass
        elif batch_type == 'overfit':
            # generates a single batch and always yields this batch to overfit on it (inspired by karpathys blog for testing)
            samples_o, timesteps_o, speaker_samples_o = None, None, None
            with h5py.File(get_dataset_file(self.dataset, self.data_type), 'r') as data:
                samples_o, timesteps_o, speaker_samples_o = self.__get_batch__(batch_size, receptive_field, 'train', data, from_all_speakers=True)
            
            i = 0
            while not self.exit_process:
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


    def start_enqueuer(self):
        queue_size = self.config.get('DATASET.queue_size')
        self.train_queue = Queue(queue_size)
        self.val_queue = Queue(queue_size)

        self.enqueuer = Process(target=self.sample_enqueuer)
        self.exit_process = False
        self.enqueuer.start()


    def terminate_enqueuer(self):
        print('stopping enqueuer...')
        self.exit_process = True
        time.sleep(5)
        self.enqueuer.terminate()
        print('enqueuer stopped.')


    def get_generator(self, generator):
        gen = self.batch_generator(generator)
        gen.__next__()
        return gen

    def batch_generator(self, set):
        queue = None
        if set == 'train':
            queue = self.train_queue
        elif set == 'val':
            queue = self.val_queue
        while True:
            [samples, timesteps, speaker_samples] = queue.get()

            label = None
            if self.label == 'speaker':
                label = speaker_samples
            elif self.label == 'timesteps':
                label = timesteps
            
            input = [samples]
            if self.condition == 'speaker':
                input.append(speaker_samples)
            yield input, label
