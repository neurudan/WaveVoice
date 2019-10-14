from path_handler import get_speaker_list_files, get_dataset_file
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
    def __init__(self, dataset, sequence_length, batch_size, use_ulaw=False):
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.data = h5py.File(get_dataset_file(dataset, use_ulaw), 'r')

    def calculate_speaker_statistics(self, speakers):
        statistics = []
        for speaker in speakers:
            statistics.append([speaker, np.sum(self.data['statistics/'+speaker][:])])
        return statistics

    def batch_generator(self, speaker_list):
        speakers = get_speakers(self.dataset, speaker_list)
        n_speakers = len(speakers)
        statistics = {}
        for speaker in speakers:
            statistics[speaker] = self.data['statistics/'+speaker][:]
            statistics[speaker+'_size'] = len(self.data['statistics/'+speaker][:])
        while True:
            samples = []
            for i in range(self.batch_size):
                speaker = speakers[np.argmax(np.random.uniform(size=n_speakers))]
                sample_id = np.argmax(np.random.uniform(size=statistics[speaker+'_size']) * statistics[speaker])
                start_id = np.random.randint(statistics[speaker][sample_id] - self.sequence_length)
                samples.append(self.data['data/'+speaker][sample_id][start_id:start_id+self.sequence_length])
            yield samples

dg = DataGenerator('timit', 6561, 100)
bg = dg.batch_generator('timit_speakers_470_stratified.txt')
