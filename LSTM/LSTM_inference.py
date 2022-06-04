import copy

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






model = Sequential()
model.add(LSTM(
      256,
      input_shape=(256, 1),
      recurrent_dropout=0.3,
      return_sequences=True
))
model.add(LSTM(256))
model.add(Dropout(0.3))
model.add(Dense(388))
model.add(Activation('softmax'))
optimizer = RMSprop(learning_rate = 0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


filepath = "lstm_cp/checkpoint_model_295.hdf5"
model_save_callback = ModelCheckpoint(
    filepath,
    monitor="val_acc",
    verbose=1,
    save_best_only=False,
    mode="auto",
    save_freq=5,
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
epoch = 300
val_data_file = "data/val_seqs_struct_new_final.pkl"
val_seqs = pickle5.load( open(val_data_file, 'rb') )

def get_random_index1(prediction):
    probs = prediction/np.sum(prediction)
    r = random.random()
    summ = 0
    index = 0
    for i in range(len(probs)):
        summ += probs[i]
        if summ >= r:
            index = i
            break
    return index
def get_random_index2(prediction):
    p = copy.deepcopy(prediction)
    index1 = np.argmax(p)
    p[index1] = -1
    index2 = np.argmax(p)
    p[index2] = -1
    index3 = np.argmax(p)
    p[index3] = -1
    index4 = np.argmax(p)
    p[index4] = -1
    index5 = np.argmax(p)
    r = random.randint(1, 2)
    print(r)
    if r == 1:
        return index1
    if r == 2:
        return index2
    # if r == 3:
    #     return index3
    # if r == 4:
    #     return index4
    # if r == 5:
    #     return index5


ss = 47
# input_seq = val_seqs[ss][:256]
# output_seq = val_seqs[ss][:256]

input_seq = training_seqs[ss][:256]
output_seq = training_seqs[ss][:256]
npbb = 0
for i in range(256):
    if 'Note-On' in word2event[input_seq[i]]:
        npbb+=1

fail_cnt = 0
event_number = 0
allowed_pos = list(set([x for x in range(event2word['Position_0/64'] + 1, event2word['Position_0/64'] + 17)]))
model.load_weights('lstm_cp/checkpoint_model_x_108.hdf5')
number_of_bar = 0
n_p = 0
while event_number <= 700:
# while number_of_bar < 8:
    print(number_of_bar)
    print('event_number: ' + str(event_number) + ' ' + word2event[output_seq[-1]])
    prediction_input = np.reshape(input_seq, (1, len(input_seq), 1))
    prediction_input = prediction_input / float(451)
    prediction = model.predict(prediction_input, verbose=0)
    eventpass = 0
    reroll_count = 0
    while eventpass != 1:
        # if reroll_count >=200:
        #     print(word2event[output_seq[-1]])
        #     continue
        #index = get_random_index2(prediction[0])
        index = np.argmax(prediction[0])
        fail_cnt = 0
        if index == 0:
            fail_cnt += 1
        if 'Bar' in word2event[output_seq[-1]] and word2event[all[index-1]] != 'Position_0/64':
            fail_cnt += 1
#        if word2event[output_seq[-1]] in beat_pos and 'Tempo-Class' not in word2event[all[index-1]]:
#            fail_cnt += 1
        if 'Tempo-Class' in word2event[output_seq[-1]] and 'Tempo_' not in word2event[all[index-1]]:
            fail_cnt += 1
        if 'Note-Velocity' in word2event[output_seq[-1]] and 'Note-On' not in word2event[all[index-1]]:
            fail_cnt += 1
        if 'Note-On' in word2event[output_seq[-1]] and 'Note-Duration' not in word2event[all[index-1]]:
            fail_cnt += 1
        if 'Note-Duration' in word2event[output_seq[-1]] and 'Position' not in word2event[all[index-1]]:
            fail_cnt += 1
        if 'Chord-Tone' in word2event[output_seq[-1]] and 'Chord-Type' not in word2event[all[index-1]]:
            fail_cnt += 1
        if 'Chord-Type' in word2event[output_seq[-1]] and 'Chord-Slash' not in word2event[all[index-1]]:
            fail_cnt += 1
        if 'Position' in word2event[all[index-1]] and all[index-1] == n_p:
            fail_cnt += 1
        # if 'Position' in word2event[all[index-1]] and all[index-1] >= event2word['Position_50/64']:
        #     fail_cnt += 1
        if 'Position' in word2event[all[index-1]] and all[index-1] not in allowed_pos:
            fail_cnt += 1
        if word2event[all[index-1]].split('_')[0] == word2event[output_seq[-1]].split('_')[0]:
            fail_cnt += 1
        if reroll_count>=452:
            eventpass = 1
            print(allowed_pos)
            this_p = random.choice(allowed_pos)
            output_seq.append(this_p)
            input_seq.pop(0)
            input_seq.append(this_p)
            event_number += 1

        if fail_cnt == 0:
            if 'Position_0/64' in word2event[all[index-1]] and 'Bar' not in word2event[output_seq[-1]]:
                output_seq.append(event2word['Bar'])
                input_seq.pop(0)
                input_seq.append(event2word['Bar'])
                event_number += 1
                number_of_bar += 1
            # if 'Bar' in word2event[all[index-1]]:
            #     number_of_bar += 1
            # if 'Note-Duration' in word2event[all[index-1]]:
            #
            if 'Position' in word2event[all[index-1]]:
                if all[index-1] > n_p:
                    n_p = all[index-1]
                else:
                    n_p = all[index-1]
                    number_of_bar += 1
            # if 'Note' in word2event[all[index-1]]:
            #     allow_pass = 1
            # if word2event[all[index - 1]] in beat_pos:
            #     if word2event[all[index - 1]] == 'Position_48/64':
            #         allowed_pos = list(set(
            #             [x for x in range(event2word['Position_49/64'], event2word['Position_49/64'] + 15)] + [
            #                 event2word['Position_0/64']]))
            #     else:
            #         allowed_pos = list(set([x for x in range(all[index - 1] + 1, all[index - 1] + 17)]))
            if 'Position' in word2event[all[index - 1]]:
                if all[index - 1] < event2word['Position_41/64']:
                    allowed_pos = list(set([x for x in range(all[index - 1] + 1, all[index - 1] + 23)]))
                    allowed_pos = allowed_pos[2:]
                else:
                    more = all[index - 1] - event2word['Position_41/64']
                    allowed_pos = [x for x in range(all[index - 1] + 1, event2word['Position_49/64'] + 15)] + [ x + event2word['Position_0/64'] for x in range(more)]
                    allowed_pos = list(set(allowed_pos[2:]))
                print([word2event[x]for x in allowed_pos])
            output_seq.append(all[index-1])
            input_seq.pop(0)
            input_seq.append(all[index-1])
            event_number +=1
            eventpass = 1
            if 'Bar' in word2event[all[index-1]]:
                output_seq.append(event2word['Position_0/64'])
                input_seq.pop(0)
                input_seq.append(event2word['Position_0/64'])
                event_number += 1
                number_of_bar += 1
        else:
            prediction[0][index] = 0
            reroll_count += 1
# output_seq.append(event2word['Note-Duration_8/64'])
# output_seq.append(event2word['Position_8/64'])
output_seq = np.array(output_seq)

def seq_to_csv(seq, word2event, out_csv):
    placeholder = np.empty( (len(seq), 2) )
    df_out = pd.DataFrame(placeholder, columns=['EVENT', 'ENCODING'])

    for i, ev in enumerate(seq):
        df_out.loc[i] = [word2event[ev], int(ev)]

    df_out.to_csv(out_csv, encoding='utf-8', index=False)

    return



events = [ word2event[w] for w in output_seq ]
out_midi_file = 'lstm_op2/output.midi'
out_struct_csv_file= 'lstm_op2/output.csv'
chord_processor = pickle5.load(open('pickles/chord_processor.pkl', 'rb'))
# try:
#   if out_struct_csv_file:
#     convert_events_to_midi(events, out_midi_file, chord_processor, use_structure=True, output_struct_csv= out_struct_csv_file)
#   else:
#     convert_events_to_midi(events, out_midi_file, chord_processor)
#   event_file = out_midi_file.replace(os.path.splitext(out_midi_file)[-1], '.csv')
#   print ('generated event sequence will be written to:', event_file)
#   seq_to_csv(output_seq, word2event, event_file)
# except Exception as e:
#   print ('error occurred when converting to', out_midi_file)
#   print (e)
convert_events_to_midi(events, out_midi_file, chord_processor, use_structure=True, output_struct_csv= out_struct_csv_file)
event_file = out_midi_file.replace(os.path.splitext(out_midi_file)[-1], '.csv')
print ('generated event sequence will be written to:', event_file)
seq_to_csv(output_seq, word2event, event_file)