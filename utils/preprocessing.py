import h5py
import librosa
import os
import numpy as np
import pickle
import shutil

from scipy import signal
from tqdm import tqdm
from utils.path_handler import get_dataset_path


def orig(x, fs=16000):
    return x, len(x)

def ulaw(x, fs=16000):
    u = 255
    x = np.sign(x) * (np.log(1 + u * np.abs(x)) / np.log(1 + u))
    x = (x + 1.) / 2. * np.iinfo('uint8').max
    return x.astype('uint8'), len(x)

def mel_spectrogram(x, fs=16000):
    nperseg = int(10 * fs / 1000)
    mel_spectrogram = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=1024, hop_length=nperseg)
    mel_spectrogram = np.log10(1 + 10000 * mel_spectrogram).T
    length = len(mel_spectrogram)
    mel_spectrogram = mel_spectrogram.flatten()
    return mel_spectrogram, length

def vgg_spectrogram(x, fs=16000):
    linear = librosa.stft(x, n_fft=512, win_length=400, hop_length=160).T
    mag, _ = librosa.magphase(linear)
    mag_T = mag.T
    freq, time = mag_T.shape
    spec_mag = mag_T
    print(spec_mag.shape)
    return spec_mag, spec_mag.shape[1]


functions = [[vgg_spectrogram, 'vgg.h5', h5py.special_dtype(vlen=np.dtype('float32')), 257]]


def create_h5_file(h5_path, audio_dict, progress_file, name):
    print('extracting %s corpus:'%name, flush=True)
    global functions
    if not os.path.isfile(progress_file):
        for _, name, data_type, dim in functions:
            with h5py.File(h5_path + name, 'w') as f:
                data = f.create_group('data')
                audio_names = f.create_group('audio_names')
                statistics = f.create_group('statistics')
                for speaker in audio_dict:
                    shape = (len(audio_dict[speaker]),)
                    if dim is not None:
                        shape = (len(audio_dict[speaker]),dim,)
                    data.create_dataset(speaker, shape, dtype=data_type)
                    audio_names.create_dataset(speaker, shape, dtype=h5py.string_dtype(encoding='utf-8'))
                    statistics.create_dataset(speaker, shape, dtype='long')
    else:
        audio_dict = pickle.load(open(progress_file, 'rb'))
    total = 0
    total_s = 0
    for speaker in audio_dict:
        total_s += 1
        total += len(audio_dict[speaker])

    pbar_s = tqdm(total=total_s, desc='speaker extraction', ncols=100, ascii=True)
    pbar = tqdm(total=total, desc='audio extraction', ncols=100, ascii=True)

    progress_dict = audio_dict.copy()

    for speaker in audio_dict:
        for i, [audio_file, audio_name] in enumerate(audio_dict[speaker]):
            x, fs = librosa.core.load(audio_file, sr=16000)
            for function, name, _, _ in functions:
                with h5py.File(h5_path + name, 'a') as f:
                    x_new, length = function(x, fs)

                    f['data/'+speaker][i] = x_new
                    f['statistics/'+speaker][i] = length
                    f['audio_names/'+speaker][i] = audio_name
            pbar.update(1)
        progress_dict.pop(speaker, None)
        pickle.dump(progress_dict, open(progress_file, 'wb'))
        pbar_s.update(1)

    pbar.close()
    pbar_s.close()
    print('%s extraction finished!\n'%name, flush=True)

def prepare_timit_dict(timit_path):
    ignored_files = ['._.DS_Store', '.DS_Store']
    required_extension = '_RIFF.WAV'

    timit_dict = {}

    if timit_path[-1] != '/':
        timit_path += '/'

    for set in ['TRAIN/', 'TEST/']:
        for subset in os.listdir(timit_path+set):
            if subset not in ignored_files:
                for speaker in os.listdir(timit_path+set+subset):
                    if speaker not in ignored_files:
                        audio_files = []
                        for audio_file in os.listdir(timit_path+set+subset+'/'+speaker):
                            if required_extension in audio_file:
                                audio_files.append([timit_path+set+subset+'/'+speaker+'/'+audio_file, audio_file])
                        timit_dict[speaker] = audio_files
    return timit_dict

def prepare_vox2_dict(vox2_path):
    required_extension = '.m4a'

    vox2_dict = {}

    if vox2_path[-1] != '/':
        vox2_path += '/'

    for set in ['vox2_train/', 'vox2_test/']:
        for speaker in tqdm(os.listdir(vox2_path+set), ncols=100, ascii=True, desc='vox2 read '+set):
            audio_files = []
            for video in os.listdir(vox2_path+set+speaker):
                for audio in os.listdir(vox2_path+set+speaker+'/'+video):
                    if required_extension in audio:
                        audio_files.append([vox2_path+set+speaker+'/'+video+'/'+audio, video+'/'+audio])
            vox2_dict[speaker] = audio_files
    return vox2_dict


def prepare_vox1_dict(vox1_path):
    required_extension = '.wav'

    vox1_dict = {}

    if vox1_path[-1] != '/':
        vox1_path += '/'

    for set in ['wav/']:
        for speaker in tqdm(os.listdir(vox1_path+set), ncols=100, ascii=True, desc='vox1 read '+set):
            audio_files = []
            for video in os.listdir(vox1_path+set+speaker):
                for audio in os.listdir(vox1_path+set+speaker+'/'+video):
                    if required_extension in audio:
                        audio_files.append([vox1_path+set+speaker+'/'+video+'/'+audio, video+'/'+audio])
            vox1_dict[speaker] = audio_files
    return vox1_dict


def prepare_vctk_dict(vctk_path):
    required_extension = '.wav'

    vctk_dict = {}

    if vctk_path[-1] != '/':
        vctk_path += '/'

    for set in ['wav48/']:
        for speaker in tqdm(os.listdir(vctk_path+set), ncols=100, ascii=True, desc='vctk read '+set):
            audio_files = []
            for audio in os.listdir(vctk_path+set+speaker+'/'):
                if required_extension in audio:
                    audio_files.append([vctk_path+set+speaker+'/'+audio, audio])
            vctk_dict[speaker] = audio_files
    return vctk_dict

def setup_datasets():
    global functions
    bases = [[prepare_timit_dict, '/cluster/home/neurudan/datasets/TIMIT/', 'timit_'],
             [prepare_vctk_dict, '/cluster/home/neurudan/datasets/VCTK-Corpus/', 'vctk_'],
             [prepare_vox1_dict, '/cluster/home/neurudan/datasets/vox1/', 'vox1_'],
             [prepare_vox2_dict, '/cluster/home/neurudan/datasets/vox2/', 'vox2_']]

    #bases = [[prepare_timit_dict, '/data/Datasets/TIMIT/', 'timit_']]


    for [f, base, name] in bases:
        dest = base + name
        full_struct = base + 'full_structure.p'
        progress = base + 'progress.p'

        dic = None

        print('\n')
        if not os.path.isfile(full_struct):
            dic = f(base)
            pickle.dump(dic, open(full_struct, 'wb'))
        else:
            dic = pickle.load(open(full_struct, 'rb'))

        if os.path.isfile(progress):
            temp = pickle.load(open(progress, 'rb'))
            if len(temp.keys()) != 0:
                create_h5_file(dest, dic, progress, name.replace('_', ''))
                project = get_dataset_path(name.replace('_', '')) + name
                for _, suffix, _ in functions:
                    shutil.copyfile(dest + suffix, project + suffix)
        else:
            create_h5_file(dest, dic, progress, name.replace('_', ''))
            project = get_dataset_path(name.replace('_', '')) + name
            #for _, suffix, _ in functions:
                #print('copying %s to project folder...'%(name+suffix))
                #shutil.copyfile(dest + suffix, project + suffix)
            print('\n')