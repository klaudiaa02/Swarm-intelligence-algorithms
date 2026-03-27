import numpy as np
import matplotlib.pyplot as plt
import time
from PSO import pso_algorithm
from CS import cuckoo_search_algorithm
from FA import firefly_algorithm
from BA import bat_algorithm

config = {
    "num_trials": 1,
    "num_dimensions": 2,
    "iterations": 20,
    "num_agents": 50,
    "function_name": "Funkcja Rastrigin",
    "picked_function": "rastrigin",
    "bounds": "bounds_rastrigin"
}

# Funkcje
def levy(x):
    x = np.asarray(x)
    w = 1 + (x - 1) / 4

    term1 = np.sin(np.pi * w[0]) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)

    return term1 + term2 + term3
def rastrigin(x):
    n = len(x)
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def griewank(x):
    return 1 + np.sum(x**2)/4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

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

def run_trials(algorithm,num_agents, num_trials, num_dimensions, iterations, funnction, bounds):
    best_scores = []
    execution_times = []
    all_results = []

    for _ in range(num_trials):
        start_time = time.time()
        results, _ = algorithm(num_agents, num_dimensions, iterations, funnction, bounds)
        execution_times.append(time.time() - start_time)

        if isinstance(results, list):
            best_scores.append(min(results))
            all_results.append(results)
        else:
            best_scores.append(results)

    return best_scores, execution_times, all_results

pso_scores, pso_times, pso_results = run_trials(pso_algorithm, config["num_agents"], config["num_trials"],
                                                config["num_dimensions"], config["iterations"], pik_fun, bounds_fun)
cs_scores, cs_times, cs_results = run_trials(cuckoo_search_algorithm, config["num_agents"], config["num_trials"],
                                                config["num_dimensions"], config["iterations"], pik_fun, bounds_fun)
firefly_scores, firefly_times, firefly_results = run_trials(firefly_algorithm, config["num_agents"], config["num_trials"],
                                                config["num_dimensions"], config["iterations"], pik_fun, bounds_fun)
bat_scores, bat_times, bat_results = run_trials(bat_algorithm, config["num_agents"], config["num_trials"],
                                                config["num_dimensions"], config["iterations"], pik_fun, bounds_fun)

if pso_scores:
    mean_pso = np.mean(pso_scores)
else:
    mean_pso = np.nan

if cs_scores:
    mean_cs = np.mean(cs_scores)
else:
    mean_cs = np.nan

if firefly_scores:
    mean_firefly = np.mean(firefly_scores)
else:
    mean_firefly = np.nan

if bat_scores:
    mean_bat = np.mean(bat_scores)
else:
    mean_bat = np.nan

mean_time_pso = np.mean(pso_times) if pso_times else np.nan
mean_time_cs = np.mean(cs_times) if cs_times else np.nan
mean_time_firefly = np.mean(firefly_times) if firefly_times else np.nan
mean_time_bat = np.mean(bat_times) if bat_times else np.nan

print("=== Wyniki porównania algorytmów ===")
print("Skuteczność:")
print(f"PSO: Średnia = {mean_pso:.4f}, Najlepszy wynik: {min(pso_scores):.4f}" if pso_scores else "Brak wyników PSO")
print(f"CS: Średnia = {mean_cs:.4f}, Najlepszy wynik: {min(cs_scores):.4f}" if cs_scores else "Brak wyników CS")
print(f"Świetlik: Średnia = {mean_firefly:.4f}, Najlepszy wynik: {min(firefly_scores):.4f}" if firefly_scores else "Brak wyników FA")
print(f"Nietoperz: Średnia = {mean_bat:.4f}, Najlepszy wynik: {min(bat_scores):.4f}" if bat_scores else "Brak wyników BA")

print("\nEfektywność obliczeniowa (średni czas wykonania):")
print(f"PSO: Średni czas wykonania = {mean_time_pso:.4f} s" if not np.isnan(mean_time_pso) else "Brak wyników PSO")
print(f"CS: Średni czas wykonania = {mean_time_cs:.4f} s" if not np.isnan(mean_time_cs) else "Brak wyników CS")
print(f"Świetlik: Średni czas wykonania = {mean_time_firefly:.4f} s" if not np.isnan(mean_time_firefly) else "Brak wyników FA")
print(f"Nietoperz: Średni czas wykonania = {mean_time_bat:.4f} s" if not np.isnan(mean_time_bat) else "Brak wyników BA")

# Wizualizacja
plt.figure(figsize=(10, 6))

if pso_results:
    avg_convergence_pso = np.min(pso_results, axis=0)
    plt.plot(avg_convergence_pso, label="PSO")
else:
    print("Brak danych PSO do wykresu.")

if firefly_results:
    avg_convergence_firefly = np.min(firefly_results, axis=0)
    plt.plot(avg_convergence_firefly, label="FA")
else:
    print("Brak danych FA do wykresu.")

if bat_results:
    avg_convergence_bat = np.min(bat_results, axis=0)
    plt.plot(avg_convergence_bat, label="BA")
else:
    print("Brak danych BA do wykresu.")

if cs_results:
    avg_convergence_cs = np.min(cs_results, axis=0)
    plt.plot(avg_convergence_cs, label="CS")
else:
    print("Brak danych CS do wykresu.")

plt.xlabel("Iteracja")
plt.ylabel("Najlepsza wartość")
plt.title(f"Porównanie zbieżności algorytmów - {config['function_name']}")
plt.legend()
plt.grid(True)

plot_filename = f"{config['function_name'].replace(' ', '_')}_dim{config['num_dimensions']}_comparison.eps"
plt.savefig(plot_filename)
print(f"Wykres zapisano jako {plot_filename}")
plt.show()
