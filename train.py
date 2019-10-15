from utils.data_handler import DataGenerator, get_speakers
from model.WaveNet import wavenet_base, final_output_conv, final_output_dense

from keras.engine import Input
from keras.engine import Model
from keras.optimizers import Adadelta, Adam
from keras.metrics import categorical_accuracy

speakers = get_speakers('timit', 'timit_speakers_100_50w_50m_not_reynolds.txt')

dilation_depth = 7
utterance_length = 3 ** dilation_depth
num_stacks = 1
num_filters = 256
num_output_bins = len(speakers)
use_ulaw = True

adam_lr=0.001
adam_beta_1=0.9
adam_beta_2=0.999

optimizer = Adadelta()
optimizer = Adam(adam_lr, adam_beta_1, adam_beta_2)

dg = DataGenerator('timit', utterance_length, 10, speakers, use_ulaw)
bg = dg.batch_generator()


input = Input(shape=(utterance_length,1), name='input')
if use_ulaw:
    input = Input(shape=(utterance_length,256), name='input')
wavenet = wavenet_base(input, num_filters, num_stacks, dilation_depth)
output = final_output_dense(wavenet, 128, num_output_bins)

loss = 'categorical_crossentropy'

bg.__next__()

model = Model(input, output)
model.summary()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.fit_generator(bg, steps_per_epoch=100, epochs=1000)
