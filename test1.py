import numpy as np
import matplotlib.pyplot as plt
import time
from PSO import pso_algorithm
from FA import firefly_algorithm
from BA import bat_algorithm
from CS import cuckoo_search_algorithm

config = {
    "num_trials": 1,
    "num_dimensions": 3,
    "iterations": 10000,
    "num_agents": 50,
    "function_name": "Funkcja Rastrigin",
    "picked_algorithm": cuckoo_search_algorithm,
    "algorithm_name": "Algorytm CA",
    "picked_function": "rastrigin",
    "bounds": "bounds_rastrigin"
}

def griewank(x):
    return 1 + np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

def rastrigin(x):
    n = len(x)
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def levy(x):
    x = np.asarray(x)
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0]) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    return term1 + term2 + term3

bounds = {
    "bounds_rastrigin": (-5.12, 5.12),
    "bounds_griewank": (-600, 600),
    "bounds_levy": (-10, 10)
}

def run_trials(algorithm, num_agents, num_trials, num_dimensions, iterations, function, bounds):
    best_scores = []
    execution_times = []
    all_results = []
    success_count = 0
    success_iterations = []

    for trial_idx in range(num_trials):
        start_time = time.time()
        results, positions_history = algorithm(num_agents, num_dimensions, iterations, function, bounds)
        execution_times.append(time.time() - start_time)

        if isinstance(results, list):
            best_score = min(results)
            best_scores.append(best_score)
            all_results.append(results)
            for iter_idx, score in enumerate(results):
                if score <= 0.01:
                    success_iterations.append(iter_idx)
                    break
        else:
            best_score = results
            best_scores.append(best_score)

        print(f"Próba {trial_idx + 1}/{num_trials} - Najlepszy wynik tej próby: {best_score:.4f}")

        if best_score <= 0.01:
            success_count += 1

    success_percentage = (success_count / num_trials) * 100
    avg_success_iteration = np.mean(success_iterations) if success_iterations else None

    return best_scores, execution_times, all_results, success_percentage, positions_history, avg_success_iteration

functions = {
    "rastrigin": rastrigin,
    "griewank": griewank,
    "levy": levy
}

pik_fun = functions[config["picked_function"]]
bounds_fun = bounds[config["bounds"]]

scores, times, results, success_percentage, positions_history, avg_success_iteration = run_trials(
    config["picked_algorithm"], config["num_agents"], config["num_trials"],
    config["num_dimensions"], config["iterations"], pik_fun, bounds_fun
)

if scores:
    mean_ = np.mean(scores)
else:
    mean_ = np.nan

mean_time = np.mean(times) if times else np.nan

print("=== Wyniki porównania algorytmu ===")
print(f"Średnia = {mean_:.4f}, Najlepszy wynik: {min(scores):.4f}" if scores else "Brak wyników")
print(f"Średni czas zbieżności = {mean_time:.4f} s" if not np.isnan(mean_time) else "Brak wyników")
print(f"Procent prób, które osiągnęły cel (wynik <= 0.01): {success_percentage:.2f}%")
if avg_success_iteration is not None:
    print(f"Średnia iteracja osiągnięcia celu: {avg_success_iteration:.2f}")
else:
    print("Funkcja celu nie została osiągnięta w żadnej próbie.")

plt.figure(figsize=(10, 6))

if results:
    avg_convergence = np.min(results, axis=0)
    plt.plot(avg_convergence, label=config["algorithm_name"])
else:
    print("Brak danych do wykresu.")

plt.xlabel("Iteracja")
plt.ylabel("Średnia najlepsza wartość")
plt.title(f"{config['function_name']} - {config['algorithm_name']}, agenci: {config['num_agents']}")
plt.legend()
plt.grid(True)
plot_filename = f"{config['function_name'].replace(' ', '_')}_{config['algorithm_name'].replace(' ', '_')}_agents{config['num_agents']}.eps"
plt.savefig(plot_filename)
print(f"Wykres zbieżności zapisano jako {plot_filename}")
plt.show()

if(config["num_trials"] == 1):
    best_position = positions_history[-1][np.argmin([pik_fun(pos) for pos in positions_history[-1]])]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(positions_history[0][:, 0], positions_history[0][:, 1], color='grey', label='Początkowy rozkład')

    iteration_in_progress = 5
    if len(positions_history) > iteration_in_progress:
        ax.scatter(positions_history[iteration_in_progress][:, 0], positions_history[iteration_in_progress][:, 1],
                   color='purple', label='Rozkład w trakcie')

    ax.scatter(positions_history[-1][:, 0], positions_history[-1][:, 1], color='black', label='Końcowy rozkład')

    ax.scatter(best_position[0], best_position[1], color='limegreen', marker='x', s=100, label='Najlepsza pozycja')

    ax.set_title("Rozkład agentów na początku, w trakcie i na końcu")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    distribution_filename = f"{config['function_name'].replace(' ', '_')}_{config['algorithm_name'].replace(' ', '_')}_distribution.eps"
    plt.savefig(distribution_filename)
    print(f"Wykres rozkładu zapisano jako {distribution_filename}")
    plt.show()