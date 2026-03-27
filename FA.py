import numpy as np

class FireflyAlgorithm:
    def __init__(self, num_fireflies, iterations, num_dimensions, function, bounds, alpha=0.8, beta0=1, gamma=1.0):

        self.num_fireflies = num_fireflies
        self.num_dimensions = num_dimensions
        self.iterations = iterations
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.bounds = bounds
        self.function = function

        self.positions = np.random.uniform(bounds[0], bounds[1], (num_fireflies, num_dimensions))
        self.intensities = np.array([self.function(pos) for pos in self.positions])
        self.best_scores = []
        self.history_positions = []

    def move_fireflies(self, i, j):
        distance = np.linalg.norm(self.positions[i] - self.positions[j])
        beta = self.beta0 * np.exp(-self.gamma * distance ** 2)
        random_component = self.alpha * (np.random.rand(self.num_dimensions) - 0.5)
        self.positions[i] += beta * (self.positions[j] - self.positions[i]) + random_component
        self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])

    def optimize(self):
        for iteration in range(self.iterations):
            for i in range(self.num_fireflies):
                for j in range(self.num_fireflies):
                    if self.intensities[i] > self.intensities[j]:
                        self.move_fireflies(i, j)
                    self.intensities[i] = self.function(self.positions[i])

            best_firefly_index = np.argmin(self.intensities)
            self.best_scores.append(self.intensities[best_firefly_index])

            self.history_positions.append(np.copy(self.positions))

        return self.best_scores, self.history_positions

def firefly_algorithm(num_agents, num_dimensions, iterations, function, bounds):
    firefly = FireflyAlgorithm(num_fireflies=num_agents, iterations=iterations, num_dimensions=num_dimensions,
                               function=function, bounds=bounds)
    best_scores, history_positions = firefly.optimize()
    return best_scores, history_positions
