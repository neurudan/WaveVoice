import os
import shutil


def get_base_dir():
    path = os.path.abspath(__file__)
    last_base = 0
    for i, folder in enumerate(path.split('/')):
        if folder == 'WaveVoice':
            last_base = i
    base_path = '/'.join(path.split('/')[:last_base+1]) + '/'
    return base_path

def find_in_path(path, extension):
    files = {}
    for file in os.listdir(path):
        if extension == file[-len(extension):]:
            files[file] = path + file
    return files

def get_dataset_base_path():
    return get_base_dir() + 'dataset/'

def get_dataset_path(dataset):
    return get_dataset_base_path() + dataset + '/'

def get_dataset_file(dataset, data_type):
    if data_type == 'original':
        return get_dataset_path(dataset)+dataset+'_original.h5'
    elif data_type == 'ulaw':
        return get_dataset_path(dataset)+dataset+'_ulaw.h5'
    elif data_type == 'mel':
        return get_dataset_path(dataset)+dataset+'_mel.h5'

def get_speaker_list_path(dataset):
    return get_dataset_path(dataset) + 'speaker_lists/'

def get_test_list_path(dataset):
    return get_dataset_path(dataset) + 'test_lists/'

def get_speaker_list_files(dataset):
    return find_in_path(get_speaker_list_path(dataset), '.txt')

def get_test_list_files(dataset):
    return find_in_path(get_test_list_path(dataset), '.txt')

def get_config_path():
    return get_base_dir() + 'configs/'

def get_sweep_config_path(filename=''):
    return get_config_path() + 'sweep/' + filename

def get_base_config_path(filename=''):
    return get_config_path() + 'base/' + filename