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

def get_result_dir():
    return get_base_dir() + 'results/'

def get_dataset_base_path():
    return get_base_dir() + 'dataset/'

def get_dataset_path(dataset):
    return get_dataset_base_path() + dataset + '/'

def get_dataset_file(dataset, use_ulaw=False):
    if use_ulaw:
        return get_dataset_path(dataset)+dataset+'_ulaw.h5'
    return get_dataset_path(dataset)+dataset+'_original.h5'

def get_speaker_list_path(dataset):
    return get_dataset_path(dataset) + 'speaker_lists/'

def get_speaker_list_files(dataset):
    return find_in_path(get_speaker_list_path(dataset), '.txt')

def get_config_path(filename=''):
    return get_base_dir() + 'configs/' + filename

def create_result_dir(config):
    dest = get_result_dir() + config.get('GENERAL', 'model') + '/' + config.file_name[:-4] + '/'
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    config_path = get_config_path(config.file_name)
    shutil.copyfile(config_path, dest + config.file_name)
    return dest
