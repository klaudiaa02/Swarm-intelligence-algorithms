import numpy as np
from scipy.special import gamma

class CuckooSearchAlgorithm:
    def __init__(self, num_nests, iterations, num_dimensions, function, bounds, pa=0.25, alpha=1.0, beta=1.5):
        self.num_nests = num_nests
        self.num_dimensions = num_dimensions
        self.iterations = iterations
        self.function = function
        self.bounds = bounds
        self.pa = pa
        self.alpha = alpha
        self.beta = beta

        self.nests = np.random.uniform(bounds[0], bounds[1], (self.num_nests, self.num_dimensions))
        self.fitness = np.array([self.function(nest) for nest in self.nests])

        self.best_solution = self.nests[np.argmin(self.fitness)]
        self.best_fitness = min(self.fitness)

        self.best_scores = []
        self.history_positions = []

    def levy_flight(self, current):
        sigma = (gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) /
                 (gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
        u = np.random.normal(0, sigma, size=len(current))
        v = np.random.normal(0, 1, size=len(current))
        step = u / (np.abs(v) ** (1 / self.beta))
        return current + self.alpha * step

    def optimize(self):
        for iteration in range(self.iterations):
            for i in range(self.num_nests):
                new_nest = self.levy_flight(self.nests[i])
                new_nest = np.clip(new_nest, self.bounds[0], self.bounds[1])
                new_fitness = self.function(new_nest)

                if new_fitness < self.fitness[i]:
                    self.nests[i] = new_nest
                    self.fitness[i] = new_fitness

            sorted_indices = np.argsort(self.fitness)
            self.nests = self.nests[sorted_indices]
            self.fitness = self.fitness[sorted_indices]

            num_to_replace = int(self.pa * self.num_nests)
            for i in range(self.num_nests - num_to_replace, self.num_nests):
                self.nests[i] = np.random.uniform(self.bounds[0], self.bounds[1], self.num_dimensions)
                self.fitness[i] = self.function(self.nests[i])

            current_best_index = np.argmin(self.fitness)
            if self.fitness[current_best_index] < self.best_fitness:
                self.best_solution = self.nests[current_best_index]
                self.best_fitness = self.fitness[current_best_index]

            self.best_scores.append(self.best_fitness)
            self.history_positions.append(self.nests.copy())

        return self.best_scores, self.history_positions

def cuckoo_search_algorithm(num_nests, num_dimensions, iterations, function, bounds):
    cuckoo = CuckooSearchAlgorithm(num_nests=num_nests, iterations=iterations, num_dimensions=num_dimensions,
                                    function=function, bounds=bounds)
    best_scores, history_positions = cuckoo.optimize()
    return best_scores, history_positions
