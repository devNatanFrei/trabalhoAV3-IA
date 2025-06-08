import numpy as np
import matplotlib.pyplot as plt
# A linha "from mpl_toolkits.mplot3d import Axes3D" foi removida

# Função para carregar pontos do CSV (assume formato: X,Y,Z por linha, primeira linha é a origem)
def load_points(file_path):
    try:
        data = np.genfromtxt(file_path, delimiter=',')
        origin = data[0]
        points = data[1:]
        return origin, points
    except:
        # Alternativa para teste se o CSV não estiver disponível
        N = 40  # Total de pontos entre 30 e 60 (neste fallback)
        origin = np.array([0, 0, 0])
        points = np.random.uniform(-30, 30, (N, 3))
        return origin, points

# Função de aptidão (fitness): distância Euclidiana total da rota
def fitness(chrom, points, origin):
    route = np.vstack([origin, points[chrom], origin])
    distances = np.linalg.norm(route[1:] - route[:-1], axis=1)
    return np.sum(distances)

# Crossover de Ordem (OX) para garantir que não haja pontos duplicados
def crossover(parent1, parent2):
    N = len(parent1)
    a, b = sorted(np.random.choice(N, 2, replace=False))
    child1 = [-1] * N
    child2 = [-1] * N
    # Copia segmento do pai1 para filho1 e do pai2 para filho2
    child1[a:b+1] = parent1[a:b+1]
    child2[a:b+1] = parent2[a:b+1]
    # Preenche posições restantes com genes do outro pai, evitando duplicatas
    p2_genes = [g for g in parent2 if g not in parent1[a:b+1]]
    p1_genes = [g for g in parent1 if g not in parent2[a:b+1]]
    idx1 = (b + 1) % N
    idx2 = (b + 1) % N
    for g in p2_genes:
        while child1[idx1] != -1:
            idx1 = (idx1 + 1) % N
        child1[idx1] = g
    for g in p1_genes:
        while child2[idx2] != -1:
            idx2 = (idx2 + 1) % N
        child2[idx2] = g
    return child1, child2

# Mutação: troca dois genes com probabilidade especificada
def mutate(chrom, mutation_rate=0.01):
    chrom_copy = chrom.copy()
    if np.random.rand() < mutation_rate:
        i, j = np.random.choice(len(chrom_copy), 2, replace=False)
        chrom_copy[i], chrom_copy[j] = chrom_copy[j], chrom_copy[i]
    return chrom_copy

# Seleção por torneio
def tournament_selection(pop, fitnesses, k=2):
    selected = []
    for _ in range(len(pop)):
        tournament = np.random.choice(len(pop), k, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament]
        best = tournament[np.argmin(tournament_fitnesses)]  # Minimizar distância
        selected.append(pop[best])
    return selected

# Algoritmo Genético
def genetic_algorithm(points, origin, pop_size=100, max_gen=1000, mutation_rate=0.01, tournament_k=2, elitism=True):
    N = len(points)
    # Inicializa população
    pop = [list(np.random.permutation(N)) for _ in range(pop_size)]
    best_solution = None
    best_fitness = float('inf')
    fitness_history = []
    
    for gen in range(max_gen):
        # Avalia aptidão
        fitnesses = [fitness(chrom, points, origin) for chrom in pop]
        
        current_min_fitness_idx = np.argmin(fitnesses)
        current_min_fitness = fitnesses[current_min_fitness_idx]

        if current_min_fitness < best_fitness:
            best_fitness = current_min_fitness
            best_solution = pop[current_min_fitness_idx].copy()
        fitness_history.append(best_fitness) # Guarda o melhor fitness encontrado até agora na geração
        
        # Elitismo (se habilitado, mantém o melhor indivíduo)
        new_pop_candidates = [] # Usado para construir a próxima população
        if elitism:
            # O best_solution já é o elite, então podemos adicioná-lo se ele existir
            if best_solution is not None:
                 new_pop_candidates.append(best_solution.copy())
        
        # Seleção
        selected = tournament_selection(pop, fitnesses, tournament_k)
        
        # Crossover e mutação para preencher o resto da população
        # Quantos indivíduos precisamos gerar
        num_to_generate = pop_size - len(new_pop_candidates)

        children = []
        # Garantir que haja pais suficientes para o crossover
        # E que o número de filhos não exceda o necessário
        idx = 0
        while len(children) < num_to_generate and idx < len(selected) -1 :
            parent1 = selected[idx]
            parent2 = selected[idx+1]
            child1, child2 = crossover(parent1, parent2)
            children.append(mutate(child1, mutation_rate))
            if len(children) < num_to_generate: # Adiciona o segundo filho se ainda houver espaço
                children.append(mutate(child2, mutation_rate))
            idx += 2
        
        # Se faltarem indivíduos após o crossover (ex: pop_size ímpar e elitismo ativado),
        # podemos preencher com os melhores da seleção que não foram usados ou mutar alguns selecionados
        # Por simplicidade, vamos apenas adicionar os filhos gerados
        new_pop_candidates.extend(children)

        # Se new_pop_candidates ainda for menor que pop_size, preencher com os restantes da seleção
        # ou, se necessário, com mutações dos melhores atuais
        current_new_pop_size = len(new_pop_candidates)
        if current_new_pop_size < pop_size:
            # Pega os melhores da seleção que ainda não foram processados ou simplesmente os primeiros da seleção
            # para preencher
            fill_count = pop_size - current_new_pop_size
            # Adiciona mais indivíduos da lista 'selected' (os que não foram usados como pais)
            # ou simplesmente os primeiros da seleção, garantindo que não exceda o tamanho de 'selected'
            additional_individuals = selected[:fill_count] # Pega os primeiros da seleção
            new_pop_candidates.extend(additional_individuals)


        # Garante que a população tenha o tamanho correto, truncando se necessário
        pop = new_pop_candidates[:pop_size]

        # Se a população ficar vazia por algum motivo (improvável com a lógica atual), pare
        if not pop:
            print("Aviso: População ficou vazia. Interrompendo.")
            break
            
    return best_solution, best_fitness, fitness_history

# Execução principal
if __name__ == "__main__":
    # Parâmetros do Algoritmo Genético
    POPULATION_SIZE = 50
    MAX_GENERATIONS = 500 # Definido como uma constante
    MUTATION_RATE = 0.01
    TOURNAMENT_K = 2

    # Carrega pontos (substitua pelo caminho real do arquivo, se disponível)
    origin, points = load_points('CaixeiroGruposGA.csv') # Tente usar um arquivo real se tiver
    # origin, points = load_points('NAO_EXISTE.csv') # Para forçar o fallback

    N_total = len(points)
    print(f"Total de pontos carregados: {N_total}")
    if not (30 <= N_total <= 60):
        print(f"Aviso: Número de pontos ({N_total}) não está entre 30 e 60 (inclusive) como especificado.")

    # Executa AG com elitismo
    print("\nExecutando Algoritmo Genético COM Elitismo...")
    best_solution_e, best_fitness_e, fitness_history_e = genetic_algorithm(
        points, origin, pop_size=POPULATION_SIZE, max_gen=MAX_GENERATIONS, 
        mutation_rate=MUTATION_RATE, tournament_k=TOURNAMENT_K, elitism=True
    )
    
    # Executa AG sem elitismo para comparação
    print("\nExecutando Algoritmo Genético SEM Elitismo...")
    best_solution_ne, best_fitness_ne, fitness_history_ne = genetic_algorithm(
        points, origin, pop_size=POPULATION_SIZE, max_gen=MAX_GENERATIONS,
        mutation_rate=MUTATION_RATE, tournament_k=TOURNAMENT_K, elitism=False
    )
    
    # Imprime resultados
    print(f"\nMelhor aptidão com elitismo: {best_fitness_e if best_fitness_e != float('inf') else 'Nenhuma solução encontrada'}")
    print(f"Melhor aptidão sem elitismo: {best_fitness_ne if best_fitness_ne != float('inf') else 'Nenhuma solução encontrada'}")
    
    # Plota histórico de aptidão
    plt.figure(figsize=(10, 5))
    if fitness_history_e:
        plt.plot(fitness_history_e, label='Com Elitismo')
    if fitness_history_ne:
        plt.plot(fitness_history_ne, label='Sem Elitismo')
    
    if fitness_history_e or fitness_history_ne:
        plt.xlabel('Geração')
        plt.ylabel('Melhor Aptidão (Distância)')
        plt.title('Convergência da Aptidão')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Nenhum histórico de aptidão para plotar.")
    
    # Visualiza a melhor rota em 3D (com elitismo)
    if best_solution_e is not None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection='3d') # Usando projection='3d' diretamente
        route = np.vstack([origin, points[best_solution_e], origin])
        ax.plot(route[:, 0], route[:, 1], route[:, 2], 'r-', label='Rota')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', label='Pontos')
        ax.scatter(origin[0], origin[1], origin[2], c='g', s=100, label='Origem')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title('Melhor Rota com Elitismo')
        plt.show()
    else:
        print("Não foi encontrada uma solução com elitismo para visualizar.")

    # Análise
    # Gerações para atingir "solução aceitável" (assumindo 10% acima da aptidão mínima encontrada)
    print("\nAnálise de Convergência:")
    if best_fitness_e != float('inf') and fitness_history_e:
        threshold_e = best_fitness_e * 1.1
        # Usando MAX_GENERATIONS como valor padrão se o limiar não for atingido
        gens_e = next((i for i, f in enumerate(fitness_history_e) if f <= threshold_e), MAX_GENERATIONS)
        print(f"Gerações para atingir solução aceitável (<= {threshold_e:.2f}) com elitismo: {gens_e if gens_e < MAX_GENERATIONS else f'>{MAX_GENERATIONS-1} (não atingido)'}")
    else:
        print("Não foi possível calcular gerações para solução aceitável com elitismo (sem solução ou histórico).")

    if best_fitness_ne != float('inf') and fitness_history_ne:
        threshold_ne = best_fitness_ne * 1.1
        # Usando MAX_GENERATIONS como valor padrão se o limiar não for atingido
        gens_ne = next((i for i, f in enumerate(fitness_history_ne) if f <= threshold_ne), MAX_GENERATIONS)
        print(f"Gerações para atingir solução aceitável (<= {threshold_ne:.2f}) sem elitismo: {gens_ne if gens_ne < MAX_GENERATIONS else f'>{MAX_GENERATIONS-1} (não atingido)'}")
    else:
        print("Não foi possível calcular gerações para solução aceitável sem elitismo (sem solução ou histórico).")
        
    print("\nAnálise do elitismo: O elitismo tipicamente melhora a velocidade de convergência e a qualidade da solução, ajudando a não perder as melhores soluções encontradas entre as gerações.")