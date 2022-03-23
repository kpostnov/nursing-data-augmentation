import datetime
import os
import numpy as np
import wandb
from models.RainbowModel import RainbowModel
from tensorflow.keras.layers import Conv1D, Dense, Dropout  # type: ignore
from tensorflow.keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape  # type: ignore
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout  # type: ignore
from tensorflow.keras.losses import CategoricalCrossentropy  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from utils import settings
from utils.Window import Window
from tensorflow.keras.backend import int_shape


from utils.Recording import Recording
from utils.array_operations import transform_to_subarrays
from utils.typing import assert_type  # type: ignore


class MultiOrhan(RainbowModel):
    """
    - !!! dont shuffle input windows !!! - model.predict() depends on a previous execution of model.predict() - CONTEXT
    - low level classifier, internal representation into another statful LSTM for final high level output
    """

    def __init__(self, dropout=0.4, epochs=10, class_weight=None, **kwargs):
        """
        LSTM
        dropout: float
        :param kwargs:
            window_size: int
            stride_size: int
            test_percentage: float
            n_features: int
            n_outputs: int
        """

        # hyper params to instance vars
        super().__init__(**kwargs)
        self.window_size = kwargs['window_size']
        self.stride_size = kwargs['stride_size']
        self.dropout = dropout
        self.class_weight = class_weight

        self.epochs = epochs
        self.batch_size = 32

        self.verbose = 0

        # wandb.config = {
        #     **kwargs
        # }

        # create model
        self.model = self.__create_model(kwargs['n_features'], kwargs['n_outputs'])

    def squeeze_excite_block(self, input):
        ''' Create a squeeze-excite block
        Args:
            input: input tensor
            filters: number of output filters
            k: width factor
        Returns: a keras tensor
        '''
        filters = input.shape[-1]  # channel_axis = -1 for TF
        se = GlobalAveragePooling1D()(input)
        se = Reshape((1, filters))(se)
        se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        se = multiply([input, se])
        return se

    def __create_model(self, n_features, n_outputs):
        print(f"Building model for {self.window_size} timesteps (window_size) and {n_features} features")
        ip = Input(shape=(self.window_size, n_features), batch_size=1)

        x = Permute((2, 1))(ip)
        x = LSTM(8)(x)  # out (None, 8)
        x = Dropout(self.dropout)(x)

        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y)

        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y)

        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])  # out (None, 136)

        # What the fuck are we really doing here?
        x = Reshape((-1, int_shape(x)[1]))(x)  # type: ignore out (None, 1, 136) (batch_size, timesteps, features)

        x = LSTM(8, stateful=True)(x)  # highlevel LSTM

        out = Dense(n_outputs, activation='softmax')(x)

        model = Model(ip, out)
        model.compile(loss=CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

        return model

    def windowize_convert(self, recordings_train: 'list[Recording]') -> 'tuple[np.ndarray,np.ndarray]':
        """
        - we dont want to shuffle here, to keep the context within the recordings (recordings are shuffled before)
        - compare to RainbowModel
        """
        windows_train: list[list[Window]] = self.windowize(recordings_train)
        X_train, y_train = self.convert(windows_train)
        return X_train, y_train

    def windowize(self, recordings: 'list[Recording]') -> 'list[list[Window]]':
        """
        - !! Different return type to keep the context of the recordings, per recording one list of windows
        based on the hyper param for window size, windowizes the recording_frames
        convertion to numpy arrays
        """
        assert_type([(recordings[0], Recording)])

        assert self.window_size is not None, "window_size has to be set in the constructor of your concrete model class please, you stupid ass"
        assert self.stride_size is not None, "stride_size has to be set in the constructor of your concrete model class, please"

        recordings_windows: 'list[list[Window]]' = []
        for recording in recordings:
            sensor_array = recording.sensor_frame.to_numpy()
            sensor_subarrays = transform_to_subarrays(sensor_array, self.window_size, self.stride_size)
            recording_windows = list(map(lambda sensor_subarray: Window(sensor_subarray, recording.activity, recording.subject), sensor_subarrays))
            recordings_windows.append(recording_windows)
        return recordings_windows

    def convert(self, recordings_windows: 'list[list[Window]]') -> 'tuple[np.ndarray, np.ndarray]':
        """
        converts the windows to two numpy arrays as needed for the concrete model
        sensor_array (data) and activity_array (labels)

        output is np only
        hours wasted: 1h
        """
        assert_type([(recordings_windows[0][0], Window)])

        recordings_sensor_arrays: list[list[np.ndarray]] = []
        recordings_activity_vectors: list[list[np.ndarray]] = []
        for recording_windows in recordings_windows:
            # X
            def window_sensor_array(window): return window.sensor_array
            recording_sensor_arrays: list[np.ndarray] = np.array(list(map(window_sensor_array, recording_windows)))
            recordings_sensor_arrays.append(recording_sensor_arrays)

            # y
            def window_activity_number(window): return settings.ACTIVITIES[window.activity]
            recording_activity_index_array = np.array(list(map(window_activity_number, recording_windows)))

            # to_categorical converts the activity_array to the dimensions needed
            recording_activity_vectors = to_categorical(recording_activity_index_array, num_classes=len(settings.ACTIVITIES))
            recordings_activity_vectors.append(recording_activity_vectors)

        return np.array(recordings_sensor_arrays), np.array(recordings_activity_vectors)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        takes a list of window lists, each list is a recording
        """
        assert_type([(X_train, (np.ndarray, np.generic)), (y_train, (np.ndarray, np.generic))])
        assert X_train.shape[0] == y_train.shape[0], "X_train and y_train have to have the same length"

        log_dir = os.path.join(settings.BP_PATH, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        assert self.epochs is not None, "epochs has to be set in the constructor of your concrete model class, please"
        for epoch in range(self.epochs):
            for i in range(X_train.shape[0]):
                recording_sensor_arrays = np.array(X_train[i])
                recording_activity_vectors = y_train[i]
                self.model.fit(recording_sensor_arrays, recording_activity_vectors, validation_split=0, epochs=epoch+1,
                               initial_epoch=epoch, batch_size=1, verbose=self.verbose, class_weight=self.class_weight, callbacks=[tensorboard_callback])
                self.model.reset_states()
            print("epoch progress: ", epoch, "/", self.epochs)

    def predict(self, X_test: np.ndarray) -> list[np.ndarray]:
        """
        takes sensor_arrays former windows with recording context 

        np.array([recording_01_sensor_arrays, recording_02_sensor_arrays, ...])
        (recording_n_sensor_arrays = np.array([sensor_array_01, sensor_array_02, ...]))

        returns the prediction vectors (one prediction vector per window aka. sensor_array) in the recording context
        [recording_01_prediction_vectors, recording_02_prediction_vectors, ...]
        (recording_n_prediction_vectors = np.array([prediction_vector_01, prediction_vector_02, ...]))
        """
        assert_type([(X_test, (np.ndarray, np.generic))])

        recordings_prediction_vectors: list[np.ndarray] = []
        for recording_sensor_arrays in X_test.tolist():
            recording_prediction_vectors: np.ndarray = self.model.predict(recording_sensor_arrays, batch_size=1)
            recordings_prediction_vectors.append(recording_prediction_vectors)
            self.model.reset_states()

        return recordings_prediction_vectors
