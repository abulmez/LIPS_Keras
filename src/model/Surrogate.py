from keras import Sequential
from keras.layers import Conv2D, Activation
from keras.optimizers import Adam


class Surrogate:

    def __init__(self, img_shape):
        self.img_shape = img_shape
        self.actual_surrogate = self.build_surrogate()

    def build_surrogate(self):
        model = Sequential()

        model.add(Conv2D(16, kernel_size=4, strides=1, input_shape=self.img_shape, padding='same'))
        model.add(Conv2D(3, kernel_size=5, strides=1, padding='same'))
        model.add(Activation('tanh'))

        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=0.000005, beta_1=0.5), metrics=['accuracy'])

        return model

    def load_surrogate_weights(self, saved_weights_file_name):
        self.actual_surrogate.load_weights("./../../surrogate_model/" + saved_weights_file_name)

    def save_surrogate_weights(self, current_time):
        self.actual_surrogate.save_weights(
            "./../../surrogate_model/" + current_time + "surrogate.h5")


