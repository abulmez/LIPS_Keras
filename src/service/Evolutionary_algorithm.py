import random

import numpy
import numpy as np

from src.model.Chromosome import Chromosome
from src.model.Population import Population


class EvolutionaryAlgorithm:

    def __init__(self, population_size, chromosome_size, mutation_chance, training_epochs, initial_best_guess,
                 expected_image,
                 combined_generator_surrogate):
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        self.mutation_chance = mutation_chance
        self.training_epochs = training_epochs
        self.initial_best_guess = initial_best_guess
        self.combined_generator_surrogate = combined_generator_surrogate
        self.expected_image = expected_image
        self.population = self.init_population()
        self.bestLosses = []

    def init_population(self):
        population = Population(self.population_size)
        for i in range(0, self.population_size):
            chromosome = Chromosome(np.random.normal(0, 1, self.chromosome_size))
            population.add_chromosome(chromosome)
        return population

    def get_actual_z_noise(self, z_noise, weights):
        actual_z_noise = []
        for i in range(0, self.chromosome_size):
            actual_z_noise.append(z_noise[i] * weights[i])
        return actual_z_noise

    def fitness(self, chromosome):
        actual_z_noise = self.get_actual_z_noise(self.initial_best_guess, chromosome.data)
        combined_surrogate_loss = self.combined_generator_surrogate.actual_combined_generator_surrogate.evaluate(
            np.array([actual_z_noise]),
            np.array(
                [self.expected_image]),
            verbose=0)
        return 1 / combined_surrogate_loss[0]

    def evaluate_population(self):
        losses = []
        for i in range(0, self.population_size):
            losses.append(self.fitness(self.population.chromosomes[i]))
        return losses

    def crossover(self, father, mother):
        child_values = (father.data + mother.data) / 2
        return Chromosome(child_values)

    def mutation(self, chromosome):
        for i in range(0, self.chromosome_size):
            if random.uniform(0, 1) < 0.5:
                chromosome.data[i] += numpy.random.beta(2, 5) / 10
            else:
                chromosome.data[i] -= numpy.random.beta(2, 5) / 10

    def vasInv(self, cumulative_sum_vector, random_v):
        for i in range(0, len(cumulative_sum_vector)):
            if random_v < cumulative_sum_vector[i]:
                return i
        return -1

    def get_best_chromosome_in_population(self, losses=None):
        if losses is None:
            losses = self.evaluate_population()
        best_index = 0
        highest_loss = -np.Inf
        for i in range(0, len(losses)):
            if losses[i] > highest_loss:
                highest_loss = losses[i]
                best_index = i
        return highest_loss, self.population.chromosomes[best_index]

    def train(self):
        self.combined_generator_surrogate.trainable = False
        for i in range(0, self.training_epochs):
            losses = self.evaluate_population()
            losses_sum = sum(losses)
            chances_vec = []
            for j in range(0, len(losses)):
                chances_vec.append(losses[j] / losses_sum)
            cumulative_sum_vector = np.cumsum(chances_vec)
            best_loss, best_chromosome = self.get_best_chromosome_in_population(losses)
            self.bestLosses.append(1 / best_loss)
            new_population = Population(self.population_size)
            for j in range(0, self.population_size - 1):
                mother_index = self.vasInv(cumulative_sum_vector, random.uniform(0, 1))
                mother = self.population.chromosomes[mother_index]
                father_index = self.vasInv(cumulative_sum_vector, random.uniform(0, 1))
                while father_index == mother_index:
                    father_index = self.vasInv(cumulative_sum_vector, random.uniform(0, 1))
                father = self.population.chromosomes[father_index]
                child = self.crossover(father, mother)
                self.mutation(child)
                new_population.add_chromosome(child)
            new_population.add_chromosome(best_chromosome)
            self.population = new_population

        self.combined_generator_surrogate.trainable = True
        return self.get_best_chromosome_in_population()[1].data
