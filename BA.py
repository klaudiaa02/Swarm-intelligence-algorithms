import numpy as np


class BatAlgorithm:
    def __init__(self, n_bats, max_iter, n_dim, function, bounds, A_init=0.7, r_init=0.1, f_min=0, f_max=0.2, alpha=0.95,
                 gamma=1.5):
        self.n_dim = n_dim
        self.n_bats = n_bats
        self.max_iter = max_iter
        self.bounds = bounds
        self.A_init = A_init
        self.r_init = r_init
        self.f_min = f_min
        self.f_max = f_max
        self.alpha = alpha
        self.gamma = gamma
        self.function = function

        self.position = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_bats, self.n_dim))
        self.velocity = np.zeros((self.n_bats, self.n_dim))
        self.loudness = np.random.uniform(self.A_init, self.A_init, self.n_bats)
        self.pulse_rate = np.random.uniform(self.r_init, self.r_init, self.n_bats)
        self.ri = self.pulse_rate
        self.frequency = np.random.uniform(self.f_min, self.f_max, self.n_bats)

        self.fitness = np.array([self.function(pos) for pos in self.position])
        self.best_fitness = np.min(self.fitness)
        self.best_position = self.position[np.argmin(self.fitness)]

        self.history_position = []
        self.best_scores = []

    def optimize(self):

        for t in range(self.max_iter):
            self.history_position.append(self.position.copy())
            for i in range(self.n_bats):
                self.velocity[i] += (self.position[i] - self.best_position) * self.frequency[i]
                new_position = self.position[i] + self.velocity[i]
                new_position = np.clip(new_position, self.bounds[0], self.bounds[1])

                if np.random.rand() > self.pulse_rate[i]:
                    epsilon = np.random.uniform(-1, 1, size=self.n_dim)
                    new_position = self.best_position + epsilon * np.mean(self.loudness)
                    new_position = np.clip(new_position, self.bounds[0], self.bounds[1])

                new_score = self.function(new_position)

                if new_score < self.fitness[i] and np.random.rand() < self.loudness[i]:
                    self.position[i] = new_position
                    self.fitness[i] = new_score
                    self.loudness[i] *= self.alpha
                    self.pulse_rate[i] = self.r_init * (1 - np.exp(-self.gamma * t))

                    if new_score < self.best_fitness:
                        self.best_fitness = new_score
                        self.best_position = new_position
            self.best_scores.append(self.best_fitness)

        return self.best_scores, self.history_position

def bat_algorithm(num_agents, num_dimensions, iterations, function, bounds):
    bat = BatAlgorithm(n_bats=num_agents, max_iter=iterations, n_dim=num_dimensions,
                       function=function, bounds=bounds)
    best_scores, history_position = bat.optimize()
    return best_scores, history_position
