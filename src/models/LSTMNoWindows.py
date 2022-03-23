from models.RainbowModel import RainbowModel
from models.LSTMModel import LSTMModel
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate, Activation, Masking  # type: ignore
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from utils.Window import Window
import numpy as np


class LSTMModelNoWindows(LSTMModel):

    def __init__(self, **kwargs):
        """
        LSTM 
        :param kwargs:
            n_max_timesteps: int
            stride_size: int
            test_percentage: float
            n_features: int
            n_outputs: int
        """

        # hyper params to instance vars
        self.n_max_timesteps = kwargs['n_max_timesteps']
        self.test_percentage = kwargs['test_percentage']

        self.verbose = 1
        self.epochs = 100
        self.batch_size = 5

        # create model
        self.model = self.__create_model(kwargs['n_features'], kwargs['n_outputs'])

    def __create_model(self, n_features, n_outputs):
        # window_size, n_features, n_outputs = X.shape[1], X.shape[2], y.shape[1]

        print(f"Building model for {self.n_max_timesteps} timesteps (max timesteps) and {n_features} features")

        ip = Input(shape=(n_features, self.n_max_timesteps))

        x = Masking()(ip)
        x = LSTM(8)(ip)
        x = Dropout(0.8)(x)

        y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = super().squeeze_excite_block(y)

        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = super().squeeze_excite_block(y)

        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])

        out = Dense(n_outputs, activation='softmax')(x)

        model = Model(ip, out)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return model

    def windowize(self, recordings):

        windows = []
        sensor_arrays = []

        # convert DFs into numpy arrays for further operations
        for recording in recordings:
            sensor_array = recording.sensor_frame.to_numpy()
            sensor_arrays.append(sensor_array)

        # get max number of timesteps over all recordings
        n_max_timesteps = max(list(map(lambda sensor_array: sensor_array.shape[0], sensor_arrays)))

        # post-pad all sensor arrays with 0's so they all have timestep size of n_max_timesteps
        for i in range(0, len(sensor_arrays)):
            n_to_pad = n_max_timesteps - sensor_arrays[i].shape[0]
            sensor_arrays[i] = np.pad(sensor_arrays[i], [(0, n_to_pad), (0, 0)], mode='constant', constant_values=0)

        # swap timestep and feature axis
        sensor_arrays = np.swapaxes(sensor_arrays, 1, 2)

        # add padded arrays to list of Window objects
        for i in range(0, len(recordings)):
            recording = recordings[i]
            padded_sensor_array = sensor_arrays[i]
            recording_window = Window(padded_sensor_array, recording.activity, recording.subject)
            windows.append(recording_window)
        return windows
