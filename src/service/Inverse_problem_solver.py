import random
import time
from tkinter import END

import numpy

import numpy as np
import os

from PIL import Image, ImageTk
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

from src.model.Combined_generator_surrogate import CombinedGeneratorSurrogate
from src.model.Generator import Generator
from src.model.Surrogate import Surrogate

plotly.tools.set_credentials_file(username='bair2059', api_key='R8rbIch9IbzPmFtrkbBM')

from src.service.Evolutionary_algorithm import EvolutionaryAlgorithm


class InverseProblemSolver:

    def __init__(self, generator_weights_file_name, training_data_folder_name, training_epochs, canvas_array, console,
                 sub_algorithm_training_steps=50, noise_vector_size=128, train_surrogate=True,
                 surrogate_weights_file_name=None):

        self.generator_weights_file_name = generator_weights_file_name
        self.img_names_list = None
        self.canvas_array = canvas_array
        self.z_dim = noise_vector_size
        self.console = console
        self.training_epochs = training_epochs
        self.training_data_folder_name = training_data_folder_name
        self.sub_algorithm_training_steps = sub_algorithm_training_steps
        self.combined_surrogate_losses = []
        self.img_shape = self.determine_training_data_shape()
        self.generator = Generator(self.z_dim, self.img_shape)
        self.load_generator_weights()
        self.surrogate = Surrogate(self.img_shape)
        self.combined_generator_surrogate = CombinedGeneratorSurrogate(self.generator, self.surrogate)
        self.random_samples = None
        self.canvas_image = [[None, None, None, None], [None, None, None, None], [None, None, None, None]]
        self.current_displayed_image = [[None, None, None, None], [None, None, None, None], [None, None, None, None]]
        self.train_surrogate = train_surrogate
        self.surrogate_weights_file_name = surrogate_weights_file_name
        if not self.train_surrogate:
            self.surrogate.load_surrogate_weights(self.surrogate_weights_file_name)

    def prepare_random_samples(self, number_of_training_images):
        if self.train_surrogate:
            indexes = list(range(0, number_of_training_images))
            random.shuffle(indexes)
            self.random_samples = indexes[0:4]
        else:
            self.random_samples = [0]

    def determine_training_data_shape(self):
        training_data_folder = os.listdir("./../../data/" + self.training_data_folder_name)
        if len(training_data_folder) > 0:
            pic = Image.open("./../../data/" + self.training_data_folder_name + "/" + training_data_folder[0])
            channels = len(pic.getbands())
            img_shape = (128, 128, channels)
            return img_shape
        else:
            raise Exception("Training data folder is empty!")

    def load_generator_weights(self):
        saved_weights_dir = os.listdir("./../../generator_model")
        if self.generator_weights_file_name in saved_weights_dir:
            self.generator.load_generator_weights(self.generator_weights_file_name)
        else:
            raise Exception("Generator weights file not found!!!")

    def load_training_data(self):
        training_images = []
        self.img_names_list = os.listdir("./../../data/" + self.training_data_folder_name)
        for img_name in self.img_names_list:
            pic = Image.open("./../../data/" + self.training_data_folder_name + "/" + img_name)
            pic = pic.resize((self.img_shape[0], self.img_shape[1]), resample=Image.LANCZOS)
            pix = numpy.array(pic)
            pix = pix / 127.5 - 1.
            training_images.append(pix)
        return numpy.array(training_images)

    def get_actual_z_noise(self, z_noise, weights):
        actual_z_noise = []
        for i in range(0, self.z_dim):
            actual_z_noise.append(z_noise[i] * weights[i])
        return actual_z_noise

    def train(self):
        starting_loss = []
        ending_loss = []
        training_pics = self.load_training_data()
        z_noise = np.random.normal(0, 1, (len(training_pics), self.z_dim))
        evolutionary_algorithms_vec = []

        sgd_weights = []
        for i in range(0, len(z_noise)):
            self.console.insert(END, "\n" + "Initializing evolutionary algorithm " + str(i))
            self.console.see("end")
            evolutionary_algorithm = EvolutionaryAlgorithm(20, self.z_dim, 0.1, self.sub_algorithm_training_steps,
                                                           z_noise[i], training_pics[i],
                                                           self.combined_generator_surrogate)
            evolutionary_algorithm.init_population()
            sgd_weights.append(evolutionary_algorithm.train())
            evolutionary_algorithms_vec.append(evolutionary_algorithm)

        for i in range(0, self.training_epochs):

            if self.train_surrogate:
                self.combined_generator_surrogate.trainable = True
                for j in range(0, self.sub_algorithm_training_steps):
                    actual_z_noise = []
                    for k in range(0, len(z_noise)):
                        actual_z_noise.append(self.get_actual_z_noise(z_noise[k], sgd_weights[k]))
                    actual_z_noise = np.array(actual_z_noise)
                    combined_surrogate_loss = self.combined_generator_surrogate.actual_combined_generator_surrogate.train_on_batch(
                        actual_z_noise,
                        training_pics)

                    console_output = "Ep. {} Part1 It. {} [Loss: {:f}, acc.: {:f}]".format(i, j,
                                                                                           combined_surrogate_loss[0],
                                                                                           100 *
                                                                                           combined_surrogate_loss[
                                                                                               1])

                    self.console.insert(END, "\n" + console_output)
                    self.console.see("end")

                    self.combined_surrogate_losses.append(combined_surrogate_loss[0])

            self.combined_generator_surrogate.trainable = False

            for j in range(0, len(training_pics)):
                sgd_weights[j] = evolutionary_algorithms_vec[j].train()
                actual_z_noise = self.get_actual_z_noise(z_noise[j], sgd_weights[j])
                combined_surrogate_loss = self.combined_generator_surrogate.actual_combined_generator_surrogate.evaluate(
                    np.array([actual_z_noise]),
                    np.array(
                        [training_pics[j]]),
                    verbose=0)

                if i == 0:
                    starting_loss.append(combined_surrogate_loss[0])
                if i == self.training_epochs - 1:
                    ending_loss.append(combined_surrogate_loss[0])

                console_output = "Ep. {} Noise {}, Loss: {:f}".format(i, j, combined_surrogate_loss[0])
                self.console.insert(END, "\n" + console_output)
                self.console.see("end")

            self.draw_images_on_canvases(z_noise, sgd_weights, training_pics)

        for i in range(0, len(training_pics)):
            self.console.insert(END, "\n" + str(starting_loss[i]) + "->" + str(ending_loss[i]))
            self.console.see("end")

        for i in range(0, len(z_noise)):
            pix = self.generator.actual_generator.predict(
                np.array([self.get_actual_z_noise(z_noise[i], sgd_weights[i])]))

            pix = pix[0]
            pix = (pix + 1) * 127.5
            pix = pix.astype(int)
            pic = Image.fromarray(numpy.uint8(pix))
            pic.save("./../../output/" + self.img_names_list[i], "JPEG")

        current_time = str(int(time.time()))
        if self.train_surrogate:
            self.surrogate.save_surrogate_weights(current_time)
            self.plot_surrogate_loss(self.training_epochs)
            self.apply_filter_on_control_pics()
        self.plot_evolutionary_algorithms_losses(evolutionary_algorithms_vec, self.training_epochs)

    def normalized_img_to_real_img(self, img):
        img = (img + 1) * 127.5
        img = img.astype(int)
        pic = Image.fromarray(numpy.uint8(img))
        return pic

    def draw_images_on_canvases(self, z_noise, sgd_weights, training_images):
        if self.random_samples is None:
            self.prepare_random_samples(len(training_images))

        for i in range(0, len(self.canvas_array)):
            if i == 0:
                for j in range(0, len(self.canvas_array[i])):
                    actual_z_noise = self.get_actual_z_noise(z_noise[self.random_samples[j]],
                                                             sgd_weights[self.random_samples[j]])
                    generated_image = self.generator.actual_generator.predict(np.array([actual_z_noise]))
                    pillow_image = self.normalized_img_to_real_img(generated_image[0])
                    self.draw_image_on_canvas(i, j, pillow_image)
            if i == 1:
                for j in range(0, len(self.canvas_array[i])):
                    pillow_image = self.normalized_img_to_real_img(training_images[self.random_samples[j]])
                    self.draw_image_on_canvas(i, j, pillow_image)
            if i == 2:
                for j in range(0, len(self.canvas_array[i])):
                    actual_z_noise = self.get_actual_z_noise(z_noise[self.random_samples[j]],
                                                             sgd_weights[self.random_samples[j]])
                    generated_image_with_filter = self.combined_generator_surrogate.actual_combined_generator_surrogate.predict(
                        np.array([actual_z_noise]))
                    pillow_image = self.normalized_img_to_real_img(generated_image_with_filter[0])
                    self.draw_image_on_canvas(i, j, pillow_image)

    def draw_image_on_canvas(self, i, j, pillow_image):
        self.current_displayed_image[i][j] = ImageTk.PhotoImage(pillow_image)
        if self.canvas_image[i][j] is None:
            self.canvas_image[i][j] = self.canvas_array[i][j].create_image(0, 0,
                                                                           image=
                                                                           self.current_displayed_image[i][
                                                                               j], anchor="nw")
        else:
            self.canvas_array[i][j].itemconfigure(self.canvas_image[i][j],
                                                  image=self.current_displayed_image[i][j])

    def plot_evolutionary_algorithms_losses(self, evolutionary_algorithms_vec, training_epochs):
        layout = go.Layout(
            yaxis=dict(
                range=[0, 1]
            )
        )

        for i in range(0, len(evolutionary_algorithms_vec)):
            trace = go.Scatter(
                x=list(range(1, ((training_epochs + 1) * self.sub_algorithm_training_steps))),
                y=evolutionary_algorithms_vec[i].bestLosses,
                mode='lines',
                name='Z-input loss'
            )
            fig = go.Figure(data=[trace], layout=layout)
            filename = 'Z-input' + str(i) + '_losses'
            py.plot(fig, filename=filename)

    def plot_surrogate_loss(self, training_epochs):
        layout = go.Layout(
            yaxis=dict(
                range=[0, 1]
            )
        )

        surrogate_trace = go.Scatter(
            x=list(range(1, (training_epochs + 1) * self.sub_algorithm_training_steps)),
            y=self.combined_surrogate_losses,
            mode='lines',
            name='Surrogate loss'
        )

        surrogate_fig = go.Figure(data=[surrogate_trace], layout=layout)
        py.plot(surrogate_fig, filename='surrogate_losses')

    def apply_filter_on_control_pics(self):
        training_images = []
        img_names_list = os.listdir("./../../data/control")
        for img_name in self.img_names_list:
            pic = Image.open("./../../data/" + self.training_data_folder_name + "/" + img_name)
            pic = pic.resize((self.img_shape[0], self.img_shape[1]), resample=Image.LANCZOS)
            pix = numpy.array(pic)
            pix = pix / 127.5 - 1.
            training_images.append(pix)
        training_images = numpy.array(training_images)
        output_images = self.surrogate.actual_surrogate.predict(training_images)
        for i in range(0, len(output_images)):
            pix = output_images[i]
            pix = (pix + 1) * 127.5
            pix = pix.astype(int)
            pic = Image.fromarray(numpy.uint8(pix))
            pic.save("./../../output_control/" + img_names_list[i], "JPEG")
