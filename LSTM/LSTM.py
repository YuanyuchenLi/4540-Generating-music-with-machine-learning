import numpy as np
#from mido import Message, MidiFile, MidiTrack
# import torch
# from torch import nn, optim
# from torch.utils.data import DataLoader, Dataset
import os, sys, pickle5 , argparse

from tensorflow.python.client import device_lib



from sklearn.preprocessing import MinMaxScaler
import tensorflow
from tensorflow import keras
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Flatten, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.losses import SparseCategoricalCrossentropy
#from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from glob import glob
import pandas as pd

sys.path.append('./src/')
from midi_decoder import convert_events_to_midi
from build_vocab import Vocab
from chord_processor import ChordProcessor
from copy import deepcopy
import random


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
print(get_available_gpus())


tensorflow.config.list_physical_devices('GPU')

value = tensorflow.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)
print ('***If TF can access GPU: ***\n\n',value) # MUST RETURN True IF IT CAN!!






model = Sequential()
# model.add(LSTM(256, input_shape=(1024,1), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(256, return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(1))
# model.add(Activation("relu"))
# optimizer = Adam(learning_rate=0.001)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.add(LSTM(
      256,
      input_shape=(256, 1),
      recurrent_dropout=0.3,
      return_sequences=True
))
model.add(LSTM(256))
model.add(Dropout(0.3))
# model.add(BatchNorm())
# model.add(Dense(382))
# model.add(Activation('relu'))
# model.add(BatchNorm())
# model.add(Dropout(0.3))
model.add(Dense(388))
model.add(Activation('softmax'))
optimizer = RMSprop(learning_rate = 0.005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
filepath = "lstm_cp/checkpoint_model_{epoch:02d}.hdf5"
model_save_callback = ModelCheckpoint(
    filepath,
    monitor="val_acc",
    verbose=1,
    save_best_only=False,
    mode="auto",
    save_freq=500,
)
vocab = pickle5.load(open('pickles/remi_wstruct_vocab.pkl', 'rb'))
event2word, word2event = vocab.event2idx, vocab.idx2event
training_data_file = "data/training_seqs_struct_new_final.pkl"
training_seqs = pickle5.load( open(training_data_file, 'rb') )

#model = multi_gpu_model(model, 2)
all = []
for i in range(len(training_seqs)):
  for j in training_seqs[i]:
    if j not in all:
      all.append(j)
all.append(33)
all.append(34)
all.append(35)
all.append(98)
all.append(99)
all.append(100)
all = sorted(all)
entry_len = 256
def get_epoch_data( training_seqs, epoch, entry_len = 256, ep_start_pitchaug= 10, pitchaug_range=(-3, 3)):
  input_data = []
  output_data = []
  for seq in training_seqs:
    if epoch >= ep_start_pitchaug:
      seq = deepcopy(seq)
      pitch_change = random.choice( pitchaug_range )
      for i, ev in enumerate(seq):
        if 'Note-On' in word2event[ev] and ev >= 21:
          seq[i] += pitch_change
        if 'Chord-Tone' in word2event[ev]:
          seq[i] += pitch_change
          if seq[i] > event2word['Chord-Tone_B']:
            seq[i] -= 12
          elif seq[i] < event2word['Chord-Tone_C']:
            seq[i] += 12
        if 'Chord-Slash' in word2event[ev]:
          seq[i] += pitch_change
          if seq[i] > event2word['Chord-Slash_B']:
            seq[i] -= 12
          elif seq[i] < event2word['Chord-Slash_C']:
            seq[i] += 12
    # if len(seq) < entry_len + 1:
    #   padlen = entry_len - len(seq) + 1
    #   seq.append(1)
    #   seq.extend([0 for x in range(padlen)])
    # if epoch < 10:
    #   offset = random.choice([0, (len(seq) - entry_len) - 1]) # only 2 possible return value
    # else:
    #   offset = random.randint(0, (len(seq) - entry_len) - 1)  # all entries in the list are possible return value
    # x = seq[ offset : offset + entry_len ]
    # y = seq[ offset + entry_len]
    # input_data.append(x)
    # opd = np.zeros(382)
    # if y == 0:
    #   output_data.append(opd)
    # elif y == 1:
    #   opd[0] = 1
    #   output_data.append(opd)
    # else:
    #   opd[all.index(y)+1] = 1
    #   output_data.append(opd)
    for i in range(0, len(seq) - entry_len , 1):
      x = seq[ i : i + entry_len ]
      y = seq[ i + entry_len ]
      input_data.append(x)
      opd = np.zeros(388)
      if y == 0:
        output_data.append(opd)
      elif y == 1:
        opd[0] = 1
        output_data.append(opd)
      else:
        opd[all.index(y)+1] = 1
        output_data.append(opd)
  return input_data, np.array(output_data)

model.load_weights('lstm_cp/checkpoint_model_x_108.hdf5')
epoch = 140

for e in range(109,epoch):
    input_data, output_data = get_epoch_data(training_seqs, e)
    n_patterns = len(input_data)
    input_data = np.reshape(input_data, (n_patterns, entry_len, 1))
    input_data = input_data / float(451)
    if e % 4 == 0:
        print('epoch:' + str(e))
        model.fit(input_data, output_data, 1024, 1, verbose=1, callbacks=[model_save_callback])
        model.save('lstm_cp/checkpoint_model_x_'+str(e)+'.hdf5')
    else:
        print('epoch:' + str(e))
        model.fit(input_data, output_data, 1024, 1, verbose=1)