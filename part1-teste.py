# import numpy as np
# import matplotlib.pyplot as plt

# # Definição das funções objetivo
# def f1(x):
#     return x[0]**2 + x[1]**2  # Minimizar

# def f2(x):
#     return np.exp(-(x[0]**2 + x[1]**2)) + 2 * np.exp(-((x[0] - 1.7)**2 + (x[1] - 1.7)**2))  # Maximizar

# def f3(x):
#     return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + 20 + np.e  # Minimizar

# def f4(x):
#     return (x[0]**2 - 10 * np.cos(2 * np.pi * x[0]) + 10) + (x[1]**2 - 10 * np.cos(2 * np.pi * x[1]) + 10)  # Minimizar

# def f5(x):
#     return (x[0] * np.cos(x[0]) / 20) + 2 * np.exp(-(x[0]**2) - (x[1] - 1)**2) + 0.01 * x[0] * x[1]  # Maximizar

# def f6(x):
#     return x[0] * np.sin(4 * np.pi * x[0]) - x[1] * np.sin(4 * np.pi * x[1] + np.pi) + 1  # Maximizar

# def f7(x):
#     return -np.sin(x[0]) * np.sin((x[0]**2 / np.pi)**(2 * 10)) - np.sin(x[1]) * np.sin((2 * x[1]**2 / np.pi)**(2 * 10))  # Minimizar

# def f8(x):
#     return -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[0]/2 + (x[1] + 47)))) - x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))  # Minimizar

# # Domínios das variáveis
# domains = {
#     'f1': [(-100, 100), (-100, 100)],
#     'f2': [(-2, 4), (-2, 5)],
#     'f3': [(-8, 8), (-8, 8)],
#     'f4': [(-5.12, 5.12), (-5.12, 5.12)],
#     'f5': [(-10, 10), (-10, 10)],
#     'f6': [(-1, 3), (-1, 3)],
#     'f7': [(0, np.pi), (0, np.pi)],
#     'f8': [(-200, 20), (-200, 20)]
# }

# # Indicação de minimização ou maximização
# minimize = {'f1': True, 'f2': False, 'f3': True, 'f4': True, 'f5': False, 'f6': False, 'f7': True, 'f8': True}

# # Função para calcular a moda usando apenas NumPy
# def compute_mode_np(values):
#     rounded_values = np.round(values, 4)
#     unique_values, counts = np.unique(rounded_values, return_counts=True)
#     max_count_idx = np.argmax(counts)
#     mode_value = unique_values[max_count_idx]
#     return mode_value

# # Implementação dos algoritmos

# # Hill Climbing
# def hill_climbing(func, domain, max_iter=1000, epsilon=0.1, t=100, minimize=True):
#     x_best = np.array([d[0] for d in domain])  # Início no limite inferior
#     f_best = func(x_best)
#     no_improve = 0
#     for _ in range(max_iter):
#         neighbor = x_best + np.random.uniform(-epsilon, epsilon, size=2)
#         neighbor = np.clip(neighbor, [d[0] for d in domain], [d[1] for d in domain])
#         f_neighbor = func(neighbor)
#         if (minimize and f_neighbor < f_best) or (not minimize and f_neighbor > f_best):
#             x_best = neighbor
#             f_best = f_neighbor
#             no_improve = 0
#         else:
#             no_improve += 1
#         if no_improve >= t:
#             break
#     return x_best, f_best

# # Local Random Search
# def local_random_search(func, domain, max_iter=1000, sigma=0.5, t=100, minimize=True):
#     x_best = np.random.uniform([d[0] for d in domain], [d[1] for d in domain], size=2)
#     f_best = func(x_best)
#     no_improve = 0
#     for _ in range(max_iter):
#         candidate = x_best + np.random.normal(0, sigma, size=2)
#         candidate = np.clip(candidate, [d[0] for d in domain], [d[1] for d in domain])
#         f_candidate = func(candidate)
#         if (minimize and f_candidate < f_best) or (not minimize and f_candidate > f_best):
#             x_best = candidate
#             f_best = f_candidate
#             no_improve = 0
#         else:
#             no_improve += 1
#         if no_improve >= t:
#             break
#     return x_best, f_best

# # Global Random Search
# def global_random_search(func, domain, max_iter=1000, minimize=True):
#     x_best = np.random.uniform([d[0] for d in domain], [d[1] for d in domain], size=2)
#     f_best = func(x_best)
#     for _ in range(max_iter - 1):
#         candidate = np.random.uniform([d[0] for d in domain], [d[1] for d in domain], size=2)
#         f_candidate = func(candidate)
#         if (minimize and f_candidate < f_best) or (not minimize and f_candidate > f_best):
#             x_best = candidate
#             f_best = f_candidate
#     return x_best, f_best

# # Função para executar experimentos
# def run_experiments(func, domain, minimize=True):
#     results = {'HC': [], 'LRS': [], 'GRS': []}
#     for _ in range(100):
#         x_hc, f_hc = hill_climbing(func, domain, minimize=minimize)
#         results['HC'].append(f_hc)
#         x_lrs, f_lrs = local_random_search(func, domain, minimize=minimize)
#         results['LRS'].append(f_lrs)
#         x_grs, f_grs = global_random_search(func, domain, minimize=minimize)
#         results['GRS'].append(f_grs)
#     return results

# # Função para calcular a moda das soluções
# def compute_mode(results):
#     modes = {}
#     for algo in results:
#         mode_val = compute_mode_np(results[algo])
#         modes[algo] = mode_val
#     return modes

# # Função para visualização
# def plot_function(func, domain, title):
#     x = np.linspace(domain[0][0], domain[0][1], 100)
#     y = np.linspace(domain[1][0], domain[1][1], 100)
#     X, Y = np.meshgrid(x, y)
#     Z = func([X, Y])
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, Z, cmap='viridis')
#     ax.set_xlabel('x1')
#     ax.set_ylabel('x2')
#     ax.set_zlabel('f(x1, x2)')
#     ax.set_title(title)
#     plt.show()

# # Execução dos experimentos para todas as funções
# functions = {'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4, 'f5': f5, 'f6': f6, 'f7': f7, 'f8': f8}
# results_table = {}

# for fname, func in functions.items():
#     print(f"\nExecutando experimentos para {fname}:")
#     # Corrigido: passar minimize[fname] diretamente
#     results = run_experiments(func, domains[fname], minimize[fname])
#     modes = compute_mode(results)
#     results_table[fname] = modes
#     print(f"Moda das soluções para {fname}: {modes}")
#     plot_function(func, domains[fname], f"{fname}(x1, x2)")
# # Exibir tabela de resultados
# print("\nTabela de Resultados (Moda das Soluções):")
# print("Função | HC       | LRS      | GRS")
# print("-------|----------|----------|----------")
# for fname in results_table:
#     hc = results_table[fname]['HC']
#     lrs = results_table[fname]['LRS']
#     grs = results_table[fname]['GRS']
#     print(f"{fname:<6} | {hc:8.4f} | {lrs:8.4f} | {grs:8.4f}")

# # Tuning de hiperparâmetros (exemplo para f1)
# print("\nTuning de hiperparâmetros para f1 (minimização):")
# epsilons = [0.01, 0.05, 0.1, 0.5, 1.0]
# sigmas = [0.1, 0.3, 0.5, 0.7, 0.9]
# optimal = 0.0  # Mínimo conhecido de f1

# for eps in epsilons:
#     x, f = hill_climbing(f1, domains['f1'], epsilon=eps, minimize=True)
#     print(f"HC com ε={eps}: f={f:.4f} (diferença do ótimo: {abs(f - optimal):.4f})")

# for sigma in sigmas:
#     x, f = local_random_search(f1, domains['f1'], sigma=sigma, minimize=True)
#     print(f"LRS com σ={sigma}: f={f:.4f} (diferença do ótimo: {abs(f - optimal):.4f})")
#     x, f = global_random_search(f1, domains['f1'], minimize=True)  # GRS não usa σ diretamente
#     print(f"GRS: f={f:.4f} (diferença do ótimo: {abs(f - optimal):.4f})")