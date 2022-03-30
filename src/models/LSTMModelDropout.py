from models.RainbowModelLeaveRecsOut import RainbowModelLeaveRecsOut
from tensorflow.keras.layers import Conv1D, Dense, Dropout  # type: ignore
from tensorflow.keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape  # type: ignore
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout  # type: ignore
from tensorflow.keras.models import Model  # type: ignore


class LSTMModelDropout(RainbowModelLeaveRecsOut):
    def __init__(self, **kwargs):
        """
        LSTM
        :param kwargs:
            window_size: int
            stride_size: int
            test_percentage: float
            n_features: int
            n_outputs: int
        """

        # hyper params to instance vars
        self.window_size = kwargs["window_size"]
        self.stride_size = kwargs["stride_size"]
        self.test_percentage = kwargs["test_percentage"]

        self.verbose = 1
        self.epochs = 10
        self.batch_size = 32

        # create model
        self.model = self.__create_model(kwargs["n_features"], kwargs["n_outputs"])
        self.model.summary()

    def squeeze_excite_block(self, input):
        """Create a squeeze-excite block
        Args:
            input: input tensor
            filters: number of output filters
            k: width factor
        Returns: a keras tensor
        """
        filters = input.shape[-1]  # channel_axis = -1 for TF
        se = GlobalAveragePooling1D()(input)
        se = Reshape((1, filters))(se)
        se = Dense(
            filters // 16,
            activation="relu",
            kernel_initializer="he_normal",
            use_bias=False,
        )(se)
        se = Dense(
            filters,
            activation="sigmoid",
            kernel_initializer="he_normal",
            use_bias=False,
        )(se)
        se = multiply([input, se])
        return se

    def __create_model(self, n_features, n_outputs):
        print(
            f"Building model for {self.window_size} timesteps (window_size) and {n_features} features"
        )
        ip = Input(shape=(self.window_size, n_features))
        ip = Dropout(0.2)(ip)

        x = Permute((2, 1))(ip)
        x = LSTM(8)(x)
        x = Dropout(0.8)(x)

        y = Conv1D(128, 8, padding="same", kernel_initializer="he_uniform")(ip)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        y = self.squeeze_excite_block(y)

        y = Conv1D(256, 5, padding="same", kernel_initializer="he_uniform")(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        y = self.squeeze_excite_block(y)

        y = Conv1D(128, 3, padding="same", kernel_initializer="he_uniform")(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])

        out = Dense(n_outputs, activation="softmax")(x)

        model = Model(ip, out)
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        return model
