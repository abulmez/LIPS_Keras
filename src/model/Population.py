class Population:

    def __init__(self, population_size):
        self.chromosomes = []
        self.population_size = population_size

    def add_chromosome(self, chromosome):
        self.chromosomes.append(chromosome)
