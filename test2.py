import numpy as np
import matplotlib.pyplot as plt
import time
from PSO import pso_algorithm
from FA import firefly_algorithm
from BA import bat_algorithm
from CS import cuckoo_search_algorithm

config = {
    "num_trials": 1,
    "num_dimensions": 2,
    "iterations": 50,
    "agents_list": [25, 50, 100],
    "function_name": "Funkcja Levy",
    "picked_algorithm": pso_algorithm,
    "algorithm_name": "Algorytm PSO",
    "picked_function": "levy",
    "bounds": "bounds_levy"
}

# Definicje funkcji celu
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

functions = {
    "rastrigin": rastrigin,
    "griewank": griewank,
    "levy": levy
}

pik_fun = functions[config["picked_function"]]
bounds_fun = bounds[config["bounds"]]

def run_trials(algorithm, num_agents, num_trials, num_dimensions, iterations, function, bounds):
    best_scores = []
    execution_times = []
    all_results = []
    success_count = 0
    success_iterations = []

    for _ in range(num_trials):
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

        if best_score <= 0.01:
            success_count += 1

    success_percentage = (success_count / num_trials) * 100
    avg_success_iteration = np.mean(success_iterations) if success_iterations else None
    return best_scores, execution_times, all_results, success_percentage, avg_success_iteration

plt.figure(figsize=(12, 8))

for num_agents in config["agents_list"]:
    scores, times, results, success_percentage, avg_success_iteration = run_trials(
        config["picked_algorithm"], num_agents, config["num_trials"],
        config["num_dimensions"], config["iterations"], pik_fun, bounds_fun
    )
    if results:
        avg_convergence = np.mean(results, axis=0)
        plt.plot(avg_convergence, label=f"Agenci: {num_agents}")

    print(f"\n=== Wyniki dla {num_agents} agentów ===")
    if scores:
        mean_, std_ = np.mean(scores), np.std(scores)
        print(f"Średnia = {mean_:.4f}, Najlepszy wynik: {min(scores):.4f}")
        print(f"Odchylenie = {std_:.4f}")
    else:
        print("Brak wyników")

    mean_time = np.mean(times) if times else np.nan
    print(f"Średni czas zbieżności = {mean_time:.4f} s" if not np.isnan(mean_time) else "Brak wyników")
    print(f"Procent prób osiągnięcia celu (wynik <= 0.001): {success_percentage:.2f}%")
    if avg_success_iteration is not None:
        print(f"Średnia iteracja osiągnięcia celu: {avg_success_iteration:.2f}")
    else:
        print("Funkcja celu nie została osiągnięta w żadnej próbie.")

plt.xlabel("Iteracja")
plt.ylabel("Średnia najlepsza wartość")
plt.title(f"{config['function_name']} - {config['algorithm_name']} dla różnej liczby agentów")
plt.legend()
plt.grid(True)

plot_filename = f"{config['function_name'].replace(' ', '_')}_{config['algorithm_name'].replace(' ', '_')}_com_agent.eps"
plt.savefig(plot_filename)
print(f"Wykres zapisano jako {plot_filename}")
plt.show()
