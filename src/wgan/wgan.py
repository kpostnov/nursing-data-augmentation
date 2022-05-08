import time
from keras.layers import Input, Dense, Reshape, Flatten, LSTM
from keras.models import Sequential, Model
import numpy as np
import keras.backend as K
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def build_generator(num_layers=3, window_shape=(120, 70), num_features=70):

    noise_shape = window_shape

    model = Sequential()

    for i in range(num_layers):
        model.add(LSTM(units=num_features*4,
                      return_sequences=True,
                      name=f'LSTM_{i + 1}'))

    model.add(Dense(units=num_features,
                    activation='sigmoid',
                    name='OUT'))
    model.add(Reshape(target_shape=window_shape))

    noise = Input(shape=noise_shape)
    img = model(noise)

    print(model.summary())

    return Model(noise, img)


def build_discriminator(num_layers=3, window_shape=(120, 70), num_features=70):

    model = Sequential()

    for i in range(num_layers):
        model.add(LSTM(units=num_features*4,
                      input_shape=window_shape,
                      return_sequences=True,
                      name=f'LSTM_{i + 1}'))

    model.add(Dense(1, activation='linear', name='OUT'))

    img = Input(shape=window_shape)
    validity = model(img)

    print(model.summary())

    return Model(img, validity)


def train(dataset=None, epochs=10000, batch_size=128, n_critic=5, num_layers=3, n_outputs=1000):
    # TODO: Normalize

    # Load the dataset
    X_train = dataset
    window_shape = X_train.shape[1:]
    num_features = window_shape[1]

    optimizer = RMSprop(learning_rate=0.00005, clipvalue=0.01) 

    discriminator = build_discriminator(window_shape=window_shape, num_features=num_features)
    discriminator.compile(loss=wasserstein_loss,
        optimizer=optimizer,
        metrics=['accuracy'])

    generator = build_generator(window_shape=window_shape, num_features=num_features)
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)

    # Random input to the generator
    z = Input(shape=window_shape)
    img = generator(z)

    discriminator.trainable = False  

    valid = discriminator(img)

    combined = Model(z, valid)
    combined.compile(loss=wasserstein_loss, optimizer=optimizer)

    # Begin training
    half_batch = int(batch_size / 2)
    batch_per_epoch = int(X_train.shape[0] / batch_size)
    n_steps = batch_per_epoch * epochs

    for step in range(n_steps):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        d_loss = 0
        # Discriminator is trained several times per generator update in Wasserstein GAN
        for _ in range(n_critic):
            # Select a random half batch of real images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            windows = X_train[idx]

            # Generate a half batch of fake images
            noise = np.random.normal(0, 1, (half_batch, window_shape[0], window_shape[1]))
            gen_windows = generator.predict(noise)

            # Train the discriminator on real and fake images
            d_loss_real = discriminator.train_on_batch(windows, -np.ones((half_batch, 1)))
            d_loss_fake = discriminator.train_on_batch(gen_windows, np.ones((half_batch, 1)))

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) 

        # ---------------------
        #  Train Generator
        # ---------------------

        # Create noise vectors as input for generator.
        noise = np.random.normal(0, 1, (batch_size, window_shape[0], window_shape[1]))

        valid_y = np.array([-1] * batch_size)

        g_loss = combined.train_on_batch(noise, tf.cast(valid_y, tf.float32))

        if step % batch_per_epoch == 0:
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (int(step / batch_per_epoch), d_loss[0], 100*d_loss[1], g_loss))

    file_name = f'generator_model_{int(time.time())}.h5'
    tf.saved_model.save(generator, file_name)

    noise = np.random.normal(0, 1, (n_outputs, window_shape[0], window_shape[1]))
    gen_windows = generator.predict(noise)

    return gen_windows
    