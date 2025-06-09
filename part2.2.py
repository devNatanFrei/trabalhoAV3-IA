import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import functools # Para usar functools.partial
import os # Para obter o número de CPUs

# --- 1. CARREGAMENTO DE DADOS ---
def load_points(file_path):
    """
    Carrega os pontos de um arquivo CSV.
    Formato esperado: X,Y,Z por linha. A primeira linha é a origem.
    Retorna a origem e um array Numpy dos pontos a serem visitados.
    Inclui um fallback para dados aleatórios se o arquivo não puder ser carregado.
    """
    try:
        data = np.genfromtxt(file_path, delimiter=',')
        if data.size == 0:
             raise ValueError("Arquivo CSV vazio.")
        if data.ndim == 1: # Apenas uma linha no arquivo
            if data.shape[0] == 3 : # Apenas origem
                 origin = data
                 points = np.array([])
            else:
                raise ValueError(f"Formato de dados inesperado para linha única no CSV (esperado 3 colunas, obteve {data.shape[0]}).")
        elif data.shape[0] == 1 and data.ndim == 2 : # Uma linha de dados, mas é 2D (provavelmente só a origem)
            if data.shape[1] == 3:
                origin = data[0]
                points = np.array([])
            else:
                raise ValueError(f"Formato de dados inesperado para linha única no CSV (esperado 3 colunas, obteve {data.shape[1]}).")
        else: # Múltiplas linhas
            origin = data[0]
            points = data[1:]
        
        if points.ndim == 1 and points.shape[0] > 0 : 
            points = points.reshape(1, -1) # Garante que 'points' seja 2D
        elif points.shape[0] == 0:
             print("Aviso: Nenhum ponto de destino encontrado no CSV (além da origem).")
        return origin, points
    except Exception as e:
        print(f"Erro ao carregar '{file_path}': {e}. Usando dados de fallback.")
        N_fallback = 40 # Conforme requisito: 30 < Npontos < 60
        origin = np.array([0, 0, 0])
        points = np.random.uniform(-30, 30, (N_fallback, 3))
        return origin, points

# --- 2. FUNÇÕES DO ALGORITMO GENÉTICO ---
def calculate_fitness_for_chromosome(chrom, points_data, origin_data):
    """Calcula a aptidão (distância total da rota) para um cromossomo."""
    if points_data.shape[0] == 0:
        return 0.0 
    try:
        # Cromossomo representa a ordem dos índices dos pontos a serem visitados
        chrom_indices = [int(c) for c in chrom]
    except (ValueError, TypeError):
        return float('inf') # Penaliza cromossomo inválido

    num_points_available = points_data.shape[0]
    if not all(0 <= idx < num_points_available for idx in chrom_indices):
        return float('inf') # Penaliza se algum índice estiver fora do range

    # Constrói a rota: Origem -> Pontos na ordem do cromossomo -> Origem
    route = np.vstack([origin_data, points_data[chrom_indices], origin_data])
    # Calcula a distância Euclidiana entre pontos consecutivos na rota
    distances = np.linalg.norm(route[1:] - route[:-1], axis=1)
    return np.sum(distances)

def crossover(parent1, parent2):
    """Realiza o crossover de Ordem (OX1) entre dois pais."""
    N = len(parent1)
    if N == 0:
        return list(parent1), list(parent2)
        
    a, b = sorted(np.random.choice(N, 2, replace=False)) # Escolhe dois pontos de corte
    child1 = [-1] * N
    child2 = [-1] * N
    
    # Copia o segmento entre os pontos de corte do pai para o filho
    child1[a:b+1] = parent1[a:b+1]
    child2[a:b+1] = parent2[a:b+1]
    
    # Preenche o restante do filho1 com genes do pai2 (mantendo a ordem e evitando duplicatas)
    p2_genes_fill_c1 = [g for g in parent2 if g not in child1[a:b+1]]
    idx_c1 = (b + 1) % N
    for gene in p2_genes_fill_c1:
        while child1[idx_c1] != -1:
            idx_c1 = (idx_c1 + 1) % N
        child1[idx_c1] = gene

    # Preenche o restante do filho2 com genes do pai1
    p1_genes_fill_c2 = [g for g in parent1 if g not in child2[a:b+1]]
    idx_c2 = (b + 1) % N
    for gene in p1_genes_fill_c2:
        while child2[idx_c2] != -1:
            idx_c2 = (idx_c2 + 1) % N
        child2[idx_c2] = gene
        
    return child1, child2

def mutate(chrom, mutation_rate=0.01):
    """Realiza a mutação por troca (swap) em um cromossomo."""
    chrom_copy = list(chrom)
    if len(chrom_copy) < 2: # Mutação não aplicável
        return chrom_copy
    if np.random.rand() < mutation_rate:
        i, j = np.random.choice(len(chrom_copy), 2, replace=False) # Escolhe dois genes para trocar
        chrom_copy[i], chrom_copy[j] = chrom_copy[j], chrom_copy[i]
    return chrom_copy

def tournament_selection(pop, fitnesses, k=2):
    """Seleciona indivíduos da população usando o método do torneio."""
    selected_individuals = []
    population_size = len(pop)
    if population_size == 0:
        return []
    for _ in range(population_size): # Seleciona 'population_size' indivíduos para a próxima geração
        tournament_competitor_indices = np.random.choice(population_size, k, replace=False)
        tournament_fitness_values = [fitnesses[i] for i in tournament_competitor_indices]
        winner_index_in_tournament = np.argmin(tournament_fitness_values) # O melhor (menor fitness) vence
        selected_individuals.append(pop[tournament_competitor_indices[winner_index_in_tournament]])
    return selected_individuals

# --- 3. ALGORITMO GENÉTICO PRINCIPAL (PARALELIZADO) ---
def genetic_algorithm_parallel(points, origin, pop_size, max_gen, 
                               mutation_rate, tournament_k, elitism,
                               num_workers=None,
                               stagnation_generations=None, 
                               target_acceptable_fitness=None):
    """
    Executa o Algoritmo Genético para o Problema do Caixeiro Viajante 3D.
    Inclui paralelização da avaliação de fitness e múltiplos critérios de parada.
    """
    N_points_to_visit = len(points) # Número de "cidades" a serem visitadas

    if N_points_to_visit == 0: # Caso especial: nenhum ponto para o drone visitar
        print("AG: Nenhum ponto de destino para processar. Rota trivial.")
        # Retorna valores que indicam uma execução vazia mas válida
        return [], 0.0, [0.0] * max_gen, "Nenhum ponto para visitar", 0 

    # Inicialização da População: Permutações aleatórias dos índices dos pontos
    pop = [list(np.random.permutation(N_points_to_visit)) for _ in range(pop_size)]
    
    best_solution_overall = None
    best_fitness_overall = float('inf')
    fitness_history = [] # Armazena o melhor fitness de cada geração
    
    # Variáveis para critérios de parada
    stagnation_counter = 0
    last_best_fitness_for_stagnation = float('inf')
    stop_reason = f"Máximo de gerações ({max_gen}) atingido." # Razão padrão para parada
    actual_generations_run = 0 # Contador de gerações efetivamente executadas

    # Prepara a função de fitness para uso com ProcessPoolExecutor
    partial_fitness_func = functools.partial(calculate_fitness_for_chromosome, 
                                             points_data=points, origin_data=origin)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for gen in range(max_gen):
            actual_generations_run = gen + 1 # Atualiza o contador de gerações
            
            # Avaliação da Aptidão (paralelizada)
            fitnesses = list(executor.map(partial_fitness_func, pop))
            
            # Encontra o melhor indivíduo da geração atual
            current_gen_best_fitness_idx = np.argmin(fitnesses)
            current_gen_best_fitness = fitnesses[current_gen_best_fitness_idx]

            # Atualiza o melhor indivíduo e fitness global encontrado até agora
            if current_gen_best_fitness < best_fitness_overall:
                best_fitness_overall = current_gen_best_fitness
                best_solution_overall = pop[current_gen_best_fitness_idx].copy()
            
            fitness_history.append(best_fitness_overall) # Adiciona o melhor fitness (global) ao histórico
            
            # Impressão de Progresso
            if (gen + 1) % (max_gen // 20 if max_gen >=20 else 1) == 0 or gen == max_gen - 1:
                 print(f"Geração {gen+1}/{max_gen} - Melhor Aptidão Global: {best_fitness_overall:.2f}")

            # --- CRITÉRIOS DE PARADA ADICIONAIS ---
            # 1. Parada por Estagnação da Melhor Aptidão
            if stagnation_generations is not None:
                # Verifica se houve melhoria significativa (usando np.isclose para floats)
                if np.isclose(best_fitness_overall, last_best_fitness_for_stagnation) or \
                   best_fitness_overall > last_best_fitness_for_stagnation - 1e-9: # Piora ou melhoria < 1e-9
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0 # Reseta se houver melhoria
                last_best_fitness_for_stagnation = best_fitness_overall # Atualiza para comparação na próxima geração

                if stagnation_counter >= stagnation_generations:
                    stop_reason = f"Parada por estagnação ({stagnation_generations} ger. s/ melhoria) na Geração {gen+1}."
                    print(stop_reason)
                    break # Sai do loop de gerações
            
            # 2. Parada por Atingir um Fitness Alvo Aceitável
            if target_acceptable_fitness is not None:
                if best_fitness_overall <= target_acceptable_fitness:
                    stop_reason = f"Parada por atingir fitness alvo ({target_acceptable_fitness:.2f}) na Geração {gen+1} (Fitness: {best_fitness_overall:.2f})."
                    print(stop_reason)
                    break # Sai do loop de gerações

            # --- CONSTRUÇÃO DA PRÓXIMA GERAÇÃO ---
            new_pop = []
            # Elitismo: Se habilitado, o melhor indivíduo encontrado até agora passa para a próxima geração
            if elitism and best_solution_overall is not None:
                new_pop.append(best_solution_overall.copy())
            
            # Seleção dos Pais
            selected_parents = tournament_selection(pop, fitnesses, tournament_k)
            if not selected_parents and pop_size > 0: # Fallback se a seleção falhar
                print(f"Aviso Geração {gen+1}: Seleção falhou. Usando população anterior como pais.")
                selected_parents = pop if pop else [list(np.random.permutation(N_points_to_visit)) for _ in range(pop_size)]

            # Crossover e Mutação para gerar filhos
            num_children_to_generate = pop_size - len(new_pop) # Quantos filhos precisamos
            children = []
            
            if num_children_to_generate > 0 and selected_parents:
                shuffled_parents = list(selected_parents) # Copia para embaralhar
                np.random.shuffle(shuffled_parents)
                
                parent_idx = 0
                # Gera filhos em pares
                while len(children) < num_children_to_generate and parent_idx < len(shuffled_parents) -1 :
                    parent1 = shuffled_parents[parent_idx]
                    parent2 = shuffled_parents[parent_idx+1]
                    child1, child2 = crossover(parent1, parent2)
                    children.append(mutate(child1, mutation_rate))
                    if len(children) < num_children_to_generate: # Adiciona o segundo filho se houver espaço
                        children.append(mutate(child2, mutation_rate))
                    parent_idx += 2
                
                # Se faltar um filho e houver um pai não usado (caso de num_children_to_generate ímpar)
                if len(children) < num_children_to_generate and parent_idx < len(shuffled_parents):
                     children.append(mutate(shuffled_parents[parent_idx].copy(), mutation_rate))

            new_pop.extend(children) # Adiciona os filhos gerados à nova população
            
            # Preenchimento da População: Se new_pop ainda for menor que pop_size
            fill_idx = 0
            while len(new_pop) < pop_size and fill_idx < len(selected_parents):
                # Reutiliza pais da seleção (com mutação) para completar
                new_pop.append(mutate(selected_parents[fill_idx].copy(), mutation_rate))
                fill_idx +=1
            
            # Fallback final para garantir o tamanho da população (raro, mas defensivo)
            if len(new_pop) < pop_size and pop_size > 0:
                # print(f"Aviso Geração {gen+1}: População incompleta ({len(new_pop)}/{pop_size}). Preenchendo aleatoriamente.")
                while len(new_pop) < pop_size:
                    new_pop.append(list(np.random.permutation(N_points_to_visit)))

            pop = new_pop[:pop_size] # Garante que a população tenha o tamanho correto
            
            # Verificação de sanidade: População não deve ficar vazia se pop_size > 0
            if not pop and pop_size > 0 :
                print(f"Alerta FATAL Geração {gen+1}: População tornou-se vazia! Repopulando para evitar erro.")
                pop = [list(np.random.permutation(N_points_to_visit)) for _ in range(pop_size)]
    
    # Atualiza stop_reason se o loop terminou por `break` antes de completar `max_gen`
    # e o motivo não foi um dos critérios explícitos. (Pouco provável com a lógica atual)
    if actual_generations_run < max_gen and stop_reason == f"Máximo de gerações ({max_gen}) atingido.":
        stop_reason = f"Loop interrompido prematuramente na Geração {actual_generations_run}."

    # Preenche o restante do histórico de fitness se o AG parou antes de max_gen
    # Isso é para que os plots de fitness_history vs max_gen ainda funcionem de forma consistente,
    # mas a análise deve usar `actual_generations_run`.
    remaining_gens_to_pad = max_gen - actual_generations_run
    if remaining_gens_to_pad > 0:
        fill_value = best_fitness_overall if best_fitness_overall != float('inf') else (fitness_history[-1] if fitness_history else float('inf'))
        fitness_history.extend([fill_value] * remaining_gens_to_pad)

    return best_solution_overall, best_fitness_overall, fitness_history, stop_reason, actual_generations_run


if __name__ == "__main__":
 
    POPULATION_SIZE = 200  
    MAX_GENERATIONS = 800  
    
    
    MUTATION_RATE = 0.01    
    TOURNAMENT_K = 3        
    
 
    STAGNATION_LIMIT = 50  
    TARGET_FITNESS_VALUE = None 
                                
    
    NUM_WORKERS = os.cpu_count() 
    print(f"Usando {NUM_WORKERS} workers para paralelização.")


    NOME_ARQUIVO_CSV = 'CaixeiroGruposGA.csv' 
    origin, points = load_points(NOME_ARQUIVO_CSV)

    N_total_pontos_visitar = len(points)
    print(f"Total de pontos a visitar (excluindo origem): {N_total_pontos_visitar}")

    if not (30 <= N_total_pontos_visitar <= 60):
        print(f"Aviso: Número de pontos ({N_total_pontos_visitar}) não está entre 30 e 60 (inclusive), conforme especificação.")

    if N_total_pontos_visitar == 0 and POPULATION_SIZE > 0:
        print("ALERTA: Não há pontos para o drone visitar. O AG terá funcionalidade limitada.")

 
    print(f"\n--- Executando AG COM Elitismo (Paralelo) ---")
    best_solution_e, best_fitness_e, fitness_history_e, stop_reason_e, actual_gens_e = genetic_algorithm_parallel(
        points, origin, pop_size=POPULATION_SIZE, max_gen=MAX_GENERATIONS, 
        mutation_rate=MUTATION_RATE, tournament_k=TOURNAMENT_K, elitism=True,
        num_workers=NUM_WORKERS,
        stagnation_generations=STAGNATION_LIMIT,
        target_acceptable_fitness=TARGET_FITNESS_VALUE
    )
    print(f"Motivo da parada (Com Elitismo): {stop_reason_e}")
    print(f"Executado por {actual_gens_e} gerações.")
    

    print(f"\n--- Executando AG SEM Elitismo (Paralelo) ---")
    best_solution_ne, best_fitness_ne, fitness_history_ne, stop_reason_ne, actual_gens_ne = genetic_algorithm_parallel(
        points, origin, pop_size=POPULATION_SIZE, max_gen=MAX_GENERATIONS,
        mutation_rate=MUTATION_RATE, tournament_k=TOURNAMENT_K, elitism=False,
        num_workers=NUM_WORKERS,
        stagnation_generations=STAGNATION_LIMIT,
        target_acceptable_fitness=TARGET_FITNESS_VALUE
    )
    print(f"Motivo da parada (Sem Elitismo): {stop_reason_ne}")
    print(f"Executado por {actual_gens_ne} gerações.")
    

    print(f"\n--- Resultados Finais ---")
    print(f"Melhor aptidão final com elitismo: {best_fitness_e if best_fitness_e != float('inf') else 'Nenhuma solução válida encontrada'}")
    print(f"Melhor aptidão final sem elitismo: {best_fitness_ne if best_fitness_ne != float('inf') else 'Nenhuma solução válida encontrada'}")
    
    
    plt.figure(figsize=(12, 7))
    title_suffix = ""
    if fitness_history_e and actual_gens_e > 0 :
        plt.plot(range(1, actual_gens_e + 1), fitness_history_e[:actual_gens_e], 
                 label=f'Com Elitismo ({actual_gens_e} gens)')
        title_suffix += f"Elit: {actual_gens_e}g "
    if fitness_history_ne and actual_gens_ne > 0:
        plt.plot(range(1, actual_gens_ne + 1), fitness_history_ne[:actual_gens_ne], 
                 label=f'Sem Elitismo ({actual_gens_ne} gens)')
        title_suffix += f"NoElit: {actual_gens_ne}g"
    
    if (fitness_history_e and actual_gens_e > 0) or \
       (fitness_history_ne and actual_gens_ne > 0) :
        plt.xlabel('Geração')
        plt.ylabel('Melhor Aptidão (Distância)')
        plt.title(f'Convergência da Aptidão ({title_suffix.strip()})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("Nenhum histórico de aptidão para plotar (nenhuma geração executada ou falha).")
    
   
    if best_solution_e is not None and N_total_pontos_visitar > 0:
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(projection='3d')
        route_indices = best_solution_e 
        route_coordinates = np.vstack([origin, points[route_indices], origin])
        
        ax.plot(route_coordinates[:, 0], route_coordinates[:, 1], route_coordinates[:, 2], 
                'r-', marker='o', markersize=4, linewidth=1.5, label='Melhor Rota (Elitismo)')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='deepskyblue', s=40, label='Pontos a Visitar', alpha=0.8)
        ax.scatter(origin[0], origin[1], origin[2], c='limegreen', s=180, marker='P', label='Origem', edgecolors='black')
       
        ax.set_xlabel('X', fontweight='bold'); ax.set_ylabel('Y', fontweight='bold'); ax.set_zlabel('Z', fontweight='bold')
        ax.legend(loc='upper left')
        plt.title(f'Melhor Rota Encontrada (Com Elitismo)\nAptidão: {best_fitness_e:.2f}\n{stop_reason_e}', fontsize=12)
        plt.tight_layout()
        plt.show()
    elif N_total_pontos_visitar == 0:
        print("Visualização da rota não aplicável: nenhum ponto de destino foi carregado.")
    elif best_solution_e is None:
        print("Não foi encontrada uma solução válida com elitismo para visualizar.")


    print("\n--- Análise de Convergência e Elitismo ---")
    ACCEPTABLE_SOLUTION_THRESHOLD_FACTOR = 1.10 

    print("\n1. Gerações para atingir solução 'aceitável':")
   
    if best_fitness_e != float('inf') and fitness_history_e and actual_gens_e > 0:
        hist_e_real = fitness_history_e[:actual_gens_e]
        threshold_val_e = best_fitness_e * ACCEPTABLE_SOLUTION_THRESHOLD_FACTOR
        gens_to_acceptable_e_list = [i+1 for i, f_val in enumerate(hist_e_real) if f_val <= threshold_val_e] # i+1 para geração 1-indexada
        
        if gens_to_acceptable_e_list:
            gens_e_achieved_at = gens_to_acceptable_e_list[0]
            print(f"  Com Elitismo: Sol. aceitável (fitness <= {threshold_val_e:.2f}) alcançada na Geração {gens_e_achieved_at}.")
        else:
            print(f"  Com Elitismo: Sol. aceitável (fitness <= {threshold_val_e:.2f}) NÃO alcançada em {actual_gens_e} gerações.")
            gens_e_achieved_at = actual_gens_e 
    else:
        print("  Com Elitismo: Não foi possível calcular gerações para solução aceitável.")
        gens_e_achieved_at = actual_gens_e if 'actual_gens_e' in locals() else MAX_GENERATIONS

    if best_fitness_ne != float('inf') and fitness_history_ne and actual_gens_ne > 0:
        hist_ne_real = fitness_history_ne[:actual_gens_ne]
        threshold_val_ne = best_fitness_ne * ACCEPTABLE_SOLUTION_THRESHOLD_FACTOR
        gens_to_acceptable_ne_list = [i+1 for i, f_val in enumerate(hist_ne_real) if f_val <= threshold_val_ne]
        
        if gens_to_acceptable_ne_list:
            gens_ne_achieved_at = gens_to_acceptable_ne_list[0]
            print(f"  Sem Elitismo: Sol. aceitável (fitness <= {threshold_val_ne:.2f}) alcançada na Geração {gens_ne_achieved_at}.")
        else:
            print(f"  Sem Elitismo: Sol. aceitável (fitness <= {threshold_val_ne:.2f}) NÃO alcançada em {actual_gens_ne} gerações.")
            gens_ne_achieved_at = actual_gens_ne
    else:
        print("  Sem Elitismo: Não foi possível calcular gerações para solução aceitável.")
        gens_ne_achieved_at = actual_gens_ne if 'actual_gens_ne' in locals() else MAX_GENERATIONS
        
    print("\n2. Análise da necessidade de Elitismo:")
    if best_fitness_e != float('inf') and best_fitness_ne != float('inf'):
        if best_fitness_e < best_fitness_ne:
            print(f"  O elitismo resultou em uma melhor solução final ({best_fitness_e:.2f}) em comparação com a execução sem elitismo ({best_fitness_ne:.2f}).")
            if gens_e_achieved_at < gens_ne_achieved_at :
                 print(f"    Além disso, com elitismo, a solução aceitável (se atingida) foi em menos gerações ({gens_e_achieved_at} vs {gens_ne_achieved_at}).")
            print("  Isso sugere que o elitismo foi benéfico nesta execução, ajudando a preservar as melhores soluções.")
        elif best_fitness_e > best_fitness_ne:
            print(f"  Surpreendentemente, a execução sem elitismo resultou em uma melhor solução final ({best_fitness_ne:.2f}) do que com elitismo ({best_fitness_e:.2f}).")
            print("    Isso pode ocorrer devido à natureza estocástica do AG, ou se o elitismo levou a uma convergência prematura para um ótimo local.")
        else:
            print(f"  Ambas as execuções (com e sem elitismo) encontraram soluções com aptidão final similar ({best_fitness_e:.2f}).")
            if gens_e_achieved_at < gens_ne_achieved_at:
                 print(f"    No entanto, com elitismo, a solução aceitável (se atingida) foi em menos gerações ({gens_e_achieved_at} vs {gens_ne_achieved_at}).")
            elif gens_e_achieved_at > gens_ne_achieved_at:
                 print(f"    No entanto, sem elitismo, a solução aceitável (se atingida) foi em menos gerações ({gens_ne_achieved_at} vs {gens_e_achieved_at}).")
    elif best_fitness_e == float('inf') and best_fitness_ne == float('inf'):
        print("  Nenhuma solução válida foi encontrada em ambas as execuções.")
    else:
        print("  Comparação do elitismo dificultada pois uma das execuções não encontrou solução válida.")

    print("\n  Conclusão Geral sobre Elitismo: Tipicamente, o elitismo é recomendado pois garante que a melhor solução encontrada até o momento não seja perdida entre gerações. No entanto, seu impacto pode variar dependendo do problema e dos outros parâmetros do AG.")
    print("  Os critérios de parada adicionais (estagnação, fitness alvo) são úteis para otimizar o tempo de execução, parando o AG quando um progresso satisfatório é feito ou quando ele deixa de progredir.")