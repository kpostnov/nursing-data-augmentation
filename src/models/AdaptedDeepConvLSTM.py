from models.RainbowModel import RainbowModel
from keras.models import Model
from keras.initializers import Orthogonal
from keras.layers import (
    Input,
    Conv2D,
    Dense,
    Flatten,
    LSTM,
    Reshape,
    LSTM,
)


class AdaptedDeepConvLSTM(RainbowModel):
    """
    We adapt the DeepConvLSTM model to perform better on our data.
    """

    # General
    batch_size = 100
    n_filters = 64
    kernel_size = 5
    n_lstm_units = 128

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "AdaptedDeepConvLSTM"

        # Create model
        self.model = self._create_model()
        print(
            f"Building model for {self.window_size} timesteps (window_size) and {kwargs['n_features']} features."
        )

    def _create_model(self):

        initializer = Orthogonal()

        def conv_layer(n_filters): return lambda the_input: Conv2D(
            filters=n_filters,
            strides=(5, 1),
            kernel_size=(5, 1),
            activation="relu",
            kernel_initializer=initializer,
        )(the_input)

        def lstm_layer(the_input): return LSTM(
            units=32, dropout=0.1, return_sequences=True, kernel_initializer=initializer
        )(the_input)

        i = Input(shape=(self.window_size, self.n_features, 1))

        # Adding 2 CNN layers.
        x = Reshape(target_shape=(self.window_size, self.n_features, 1))(i)
        conv_n_filters = [32, 64]
        for n_filters in conv_n_filters:
            x = conv_layer(n_filters=n_filters)(x)

        x = Reshape((int(x.shape[1]), int(x.shape[2]) * int(x.shape[3]),))(x)

        for _ in range(1):
            x = lstm_layer(x)

        x = Flatten()(x)
        x = Dense(units=self.n_outputs, activation="softmax")(x)

        model = Model(i, x)

        print(model.summary())

        model.compile(
            optimizer="RMSprop",
            loss="CategoricalCrossentropy",
            metrics=["accuracy"],
        )

        return model
