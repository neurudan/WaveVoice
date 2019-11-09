import h5py
import librosa
import os
import numpy as np
import pickle

from tqdm import tqdm

def ulaw(x):
    u = 255
    x = np.sign(x) * (np.log(1 + u * np.abs(x)) / np.log(1 + u))
    x = (x + 1.) / 2. * np.iinfo('uint8').max
    return x.astype('uint8')

def create_h5_file(h5_path, audio_dict, progress_file):
    if not os.path.isfile(progress):
        for use_ulaw, name in [[True, 'ulaw.h5'], [False, 'original.h5']]:
            dt = h5py.special_dtype(vlen=np.dtype('float32'))
            if use_ulaw:
                dt = h5py.special_dtype(vlen=np.dtype('uint8'))
            with h5py.File(h5_path + name, 'w') as f:
                data = f.create_group('data')
                audio_names = f.create_group('audio_names')
                statistics = f.create_group('statistics')
                for speaker in audio_dict:
                    data.create_dataset(speaker, (len(audio_dict[speaker]),), dtype=dt)
                    audio_names.create_dataset(speaker, (len(audio_dict[speaker]),), dtype=h5py.string_dtype(encoding='utf-8'))
                    statistics.create_dataset(speaker, (len(audio_dict[speaker]),), dtype='long')
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
        for i, [audio_file, name] in enumerate(audio_dict[speaker]):
            x, _ = librosa.core.load(audio_file)
            with h5py.File(h5_path + 'original.h5', 'a') as f:
                f['data/'+speaker][i] = x
                f['statistics/'+speaker][i] = len(x)
            x = ulaw(x)
            with h5py.File(h5_path + 'ulaw.h5', 'a') as f:
                f['data/'+speaker][i] = x
                f['statistics/'+speaker][i] = len(x)
            pbar.update(1)
        progress_dict.pop(speaker, None)
        pickle.dump(progress_dict, open(progress_file, 'wb'))
        pbar_s.update(1)

    pbar.close()
    pbar_s.close()

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

bases = [[prepare_timit_dict, '/cluster/home/neurudan/datasets/TIMIT/', 'timit_'],
         [prepare_vctk_dict, '/cluster/home/neurudan/datasets/VCTK-Corpus/', 'vctk_'], 
         [prepare_vox2_dict, '/cluster/home/neurudan/datasets/vox2/', 'vox2_']]

for [f, base, name] in bases:
    dest = base + name
    full_struct = base + 'full_structure.p'
    progress = base + 'progress.p'

    dic = None

    if not os.path.isfile(full_struct):
        dic = f(base)
        pickle.dump(dic, open(full_struct, 'wb'))
    else:
        dic = pickle.load(open(full_struct, 'rb'))

    create_h5_file(dest, dic, progress)
