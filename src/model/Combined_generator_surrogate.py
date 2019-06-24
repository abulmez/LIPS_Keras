from keras import Sequential
from keras.optimizers import Adam


class CombinedGeneratorSurrogate:

    def __init__(self, generator, surrogate):
        self.actual_combined_generator_surrogate = self.build_combined_generator_surrogate(generator, surrogate)

    def build_combined_generator_surrogate(self, generator, surrogate):
        model = Sequential()
        model.add(generator.actual_generator)
        model.add(surrogate.actual_surrogate)
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=0.000005, beta_1=0.5),
                      metrics=['accuracy'])
        return model

