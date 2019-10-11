import h5py
import librosa
import os
import numpy as np
from tqdm import tqdm

def ulaw(x, u=255):
    x = np.sign(x) * (np.log(1 + u * np.abs(x)) / np.log(1 + u))
    x = (x + 1.) / 2. * np.iinfo('uint8').max
    return x.astype('uint8')

def create_h5_file(h5_path, audio_dict, dtype, use_ulaw=True):
    dt = h5py.special_dtype(vlen=np.dtype(dtype))
    with h5py.File(h5_path, 'w') as f:
        for speaker in audio_dict:
            speaker_group = f.create_group(speaker)
            speaker_group.create_dataset('data', (len(audio_dict[speaker]),), dtype=dt)

    total = 0
    for speaker in audio_dict:
        total += len(audio_dict[speaker])

    pbar = tqdm(total=total, desc='audio extraction')

    for speaker in audio_dict:
        for i, audio_file in enumerate(audio_dict[speaker]):
            x, fs = librosa.core.load(audio_file)
            if use_ulaw:
                x = ulaw(x)
            with h5py.File(h5_path, 'a') as f:
                f[speaker+'/data'][i] = x
            pbar.update(1)
    pbar.close()

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
                                audio_files.append(timit_path+set+subset+'/'+speaker+'/'+audio_file)
                        timit_dict[speaker] = audio_files
    return timit_dict

def prepare_vox2_dict(vox2_path):
    required_extension = '.m4a'

    vox2_dict = {}

    if vox2_path[-1] != '/':
        vox2_path += '/'

    for set in ['train/', 'test/']:
        for speaker in tqdm(os.listdir(vox2_path+set+'aac/'), desc='read '+set):
            audio_files = []
            for video in os.listdir(vox2_path+set+'aac/'+speaker):
                for audio in os.listdir(vox2_path+set+'aac/'+speaker+'/'+video):
                    if required_extension in audio:
                        audio_files.append(vox2_path+set+'aac/'+speaker+'/'+video+'/'+audio)
            vox2_dict[speaker] = audio_files
    return vox2_dict

timit_dict = prepare_vox2_dict('/data/Datasets/VOX2/')
create_h5_file('vox2_original.h5', timit_dict, 'float32', False)
