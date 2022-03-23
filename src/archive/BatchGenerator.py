from tensorflow.keras.utils import Sequence
import numpy as np
# Implementation adapted from: https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c
# https://datascience.stackexchange.com/questions/85966/in-sequence-models-is-it-possible-to-have-training-batches-with-different-times

# Generates a batch for each recording


class BatchGenerator(Sequence):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return int(np.floor(len(self.inputs) / self.batch_size))

    def __getitem__(self, index):
        max_length = 0
        start_index = index*batch_size
        end_index = start_index+batch_size
        for i in range(start_index, end_index):
            l = len(self.inputs[i])
            if l > max_length:
                max_length = l

        out_x = np.empty([self.batch_size, max_length], dtype='int32')
        out_y = np.empty([self.batch_size, 1], dtype='float32')
        for i in range(self.batch_size):
            out_y[i] = self.labels[start_index+i]
            tweet = self.inputs[start_index+i]
            l = len(tweet)
            for j in range(l):
                out_x[i][j] = tweet[j]
            for j in range(l, max_length):
                out_x[i][j] = self.padding
        return out_x, out_y


# The model.fit function can then be called like this:

training_generator = BatchGenerator(tokens_train, y_train, pad, batch_size)
model.fit(training_generator, epochs=epochs)
