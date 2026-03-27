import numpy as np

class ParticleSwarmOptimizer:
    def __init__(self, num_particles, iterations, dimensions, bounds, function, w=0.8, c1=2, c2=2):
        self.dimensions = dimensions
        self.num_particles = num_particles
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.bounds = bounds
        self.function = function

        self.particles = np.random.uniform(self.bounds[0], self.bounds[1], (self.num_particles, self.dimensions))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dimensions))
        self.personal_best = self.particles.copy()
        self.personal_best_scores = np.array([self.function(p) for p in self.particles])
        self.global_best = self.personal_best[np.argmin(self.personal_best_scores)]

        self.positions_history = [self.particles.copy()]
        self.global_best_scores = [self.function(self.global_best)]

    def optimize(self):
        for _ in range(self.iterations):
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                self.velocities[i] = (
                        self.w * self.velocities[i]
                        + self.c1 * r1 * (self.personal_best[i] - self.particles[i])
                        + self.c2 * r2 * (self.global_best - self.particles[i])
                )
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.bounds[0], self.bounds[1])
                score = self.function(self.particles[i])
                if score < self.personal_best_scores[i]:
                    self.personal_best[i] = self.particles[i]
                    self.personal_best_scores[i] = score

            self.global_best = self.personal_best[np.argmin(self.personal_best_scores)]
            self.positions_history.append(self.particles.copy())
            self.global_best_scores.append(self.function(self.global_best))

        return self.global_best_scores, self.positions_history

def pso_algorithm(num_agents, num_dimensions, iterations, function, bounds):
    pso = ParticleSwarmOptimizer(num_particles=num_agents, iterations=iterations, dimensions=num_dimensions,
                                 function=function, bounds=bounds)
    global_best_scores, positions_history = pso.optimize()
    return global_best_scores, positions_history