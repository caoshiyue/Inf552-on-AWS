
import numpy as np
import os
from keras.utils import np_utils
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Activation

char_set = set(open('./corpus').read().lower())
char_set = sorted(list(char_set))
char_2_float = dict()
char_2_int = dict()
int_2_char = dict()
i = 0
for c in char_set:
    char_2_float[c] = i/len(char_set)
    i += 1
i = 0
for c in char_set:
    char_2_int[c] = i
    i += 1
i = 0
for c in char_2_float:
    int_2_char[i] = c
    i += 1


chars = open('./corpus').read().lower()
total_chars = len(chars)

W = 99
train_data = []
train_target = []
for i in range(0, total_chars - W):
    input_char = chars[i:i + W]
    output_char = chars[i + W]
    p = []
    for c in input_char:
        p.append(char_2_float[c])
    train_data.append(p)
    train_target.append(char_2_int[output_char])

train_data = np.reshape(train_data, (len(train_data), W, 1))
train_target = np_utils.to_categorical(train_target)


print(train_data.shape)
print(train_target.shape)

model = Sequential()
model.add(LSTM(256, input_shape=(train_data.shape[1], train_data.shape[2])))
model.add(Dense(train_target.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
print(model.summary())

checkpointer = ModelCheckpoint(
    filepath='./checkpoint/{epoch:02d}-{loss:.2f}.hdf5', monitor='loss',  save_best_only=True, mode='min', verbose=0)
model.fit(train_data, train_target, batch_size=512,
          epochs=30, verbose=1, callbacks=[checkpointer])

# result
model = Sequential()
model.add(LSTM(256, input_shape=(train_data.shape[1], train_data.shape[2])))
model.add(Dense(train_target.shape[1], activation='softmax'))
model.load_weights("./checkpoint/30-2.37.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam')

init = 'There are those who take mental phenomena naively, just as they would physical phenomena. This school of psychologists tends not to emphasize the object.'
init = init.lower()
content = init
init_float = []
for c in init:
    init_float.append(char_2_float[c])

data = init_float[-99:]
for i in range(1000):
    pred = model.predict(np.reshape(data, (1, len(data), 1)))
    char = int_2_char[np.argmax(pred)]
    content += char
    data.append(char_2_float[char])
    data.pop(0)

print(content)
