from keras import Sequential
from keras.layers import Dense, Reshape, BatchNormalization, Conv2DTranspose, Activation
from keras.optimizers import Adam


class Generator:

    def __init__(self, z_dim, img_shape):
        self.z_dim = z_dim
        self.img_shape = img_shape
        self.actual_generator = self.build_generator()

    def build_generator(self):
        model = Sequential()

        model.add(Dense(1024 * 1 * 1, input_shape=(self.z_dim,)))

        model.add(Reshape((1, 1, 1024)))
        model.add(BatchNormalization(momentum=0.5))

        model.add(Conv2DTranspose(
            1024, kernel_size=4))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(
            512, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(
            256, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(
            128, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(
            64, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(
            32, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(
            self.img_shape[2], kernel_size=5, strides=1, padding='same'))
        model.add(Activation('tanh'))

        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

        return model

    def load_generator_weights(self, saved_weights_file_name):
        self.actual_generator.load_weights("./../../generator_model/" + saved_weights_file_name)
