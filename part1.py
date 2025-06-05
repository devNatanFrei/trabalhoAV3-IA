import numpy as np
import matplotlib.pyplot as plt
# mpl_toolkits.mplot3d é implicitamente usado com projection='3d'

# --- Funções Objetivo ---

# 1. Função Esfera (minimizar)
def f1(x1, x2):
    return x1**2 + x2**2

# 2. Função tipo Himmelblau (maximizar) - note o **2 no expoente do segundo termo
def f2(x1, x2):
    return np.exp(-(x1**2 + x2**2)) + 2 * np.exp(-((x1 - 1.7)**2 + (x2 - 1.7)**2))

# 3. Função Ackley (minimizar)
def f3(x1, x2):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) - \
           np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.e

# 4. Função Rastrigin (minimizar)
def f4(x1, x2):
    return (x1**2 - 10 * np.cos(2 * np.pi * x1) + 10) + \
           (x2**2 - 10 * np.cos(2 * np.pi * x2) + 10)

# 5. Função Complexa (maximizar)
def f5(x1, x2):
    return (x1 * np.cos(x1) / 20) + \
           (2 * np.exp(-(x1**2) - (x2 - 1)**2)) + \
           (0.01 * x1 * x2)

# 6. Função Senoidal (maximizar)
def f6(x1, x2):
    return x1 * np.sin(4 * np.pi * x1) - x2 * np.sin(4 * np.pi * x2 + np.pi) + 1

# 7. Função Michalewicz (minimizar) - assumindo 2*10 significa potência de 20
def f7(x1, x2):
    termo1 = -np.sin(x1) * (np.sin(x1**2 / np.pi))**(2 * 10)
    termo2 = -np.sin(x2) * (np.sin(2 * x2**2 / np.pi))**(2 * 10) 
    return termo1 + termo2

# 8. Função tipo Eggholder (minimizar) - assumindo 'p' significa sqrt (raiz quadrada)
def f8(x1, x2):
    termo1_arg = np.abs(x1/2 + (x2 + 47))
    termo2_arg = np.abs(x1 - (x2 + 47))
    
    termo1 = -(x2 + 47) * np.sin(np.sqrt(termo1_arg))
    termo2 = -x1 * np.sin(np.sqrt(termo2_arg))
    return termo1 + termo2

# Função auxiliar para plotagem
def plotar_funcao(funcao_obj, limites_plot, titulo_plot="f(x1,x2)"):
    print(f"Plotando: {titulo_plot}. Feche a janela do gráfico para continuar...")
    x_min, x_max = limites_plot[0]
    y_min, y_max = limites_plot[1]
    
    valores_x = np.linspace(x_min, x_max, 100)
    valores_y = np.linspace(y_min, y_max, 100)
    X1, X2 = np.meshgrid(valores_x, valores_y)
    Y = funcao_obj(X1, X2)
    
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d') 
    ax.plot_surface(X1, X2, Y, cmap='jet', rstride=5, cstride=5, alpha=0.7)
    
    offset_contour = np.min(Y) - abs(np.min(Y)*0.1) if np.min(Y) < 0 else np.min(Y)*0.9
    if np.isnan(offset_contour) or np.isinf(offset_contour): 
        offset_contour = 0 if np.all(np.isnan(Y)) else np.nanmin(Y) * 0.9 if np.nanmin(Y) >=0 else np.nanmin(Y) * 1.1

    ax.contour(X1, X2, Y, zdir='z', offset=offset_contour , cmap='coolwarm')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1,x2)')
    ax.set_title(titulo_plot)
    plt.show()

# --- Algoritmos Meta-heurísticos ---

def verificar_limites(candidato, limites_dominio):
    candidato_limitado = np.zeros_like(candidato)
    for i in range(len(candidato)):
        candidato_limitado[i] = np.clip(candidato[i], limites_dominio[i][0], limites_dominio[i][1])
    return candidato_limitado

def subida_de_encosta(funcao_obj, limites_dominio, num_max_iteracoes, paciencia_parada_antecipada, epsilon, eh_maximizacao=False):
    solucao_atual_x = np.array([lim[0] for lim in limites_dominio])
    solucao_atual_f = funcao_obj(solucao_atual_x[0], solucao_atual_x[1])
    if eh_maximizacao:
        solucao_atual_f = -solucao_atual_f

    melhor_solucao_x = np.copy(solucao_atual_x)
    melhor_solucao_f = solucao_atual_f
    sequencia_sem_melhoria = 0

    for _ in range(num_max_iteracoes):
        candidato_x = solucao_atual_x + np.random.uniform(-epsilon, epsilon, size=len(solucao_atual_x))
        candidato_x = verificar_limites(candidato_x, limites_dominio)
        candidato_f = funcao_obj(candidato_x[0], candidato_x[1])
        if eh_maximizacao:
            candidato_f = -candidato_f
            
        if candidato_f < solucao_atual_f: 
            solucao_atual_x = candidato_x
            solucao_atual_f = candidato_f
            sequencia_sem_melhoria = 0
            if solucao_atual_f < melhor_solucao_f: 
                melhor_solucao_x = np.copy(solucao_atual_x)
                melhor_solucao_f = solucao_atual_f
        else:
            sequencia_sem_melhoria += 1
            
        if sequencia_sem_melhoria >= paciencia_parada_antecipada:
            break
            
    if eh_maximizacao:
        melhor_solucao_f = -melhor_solucao_f
    return melhor_solucao_x, melhor_solucao_f

def busca_local_aleatoria(funcao_obj, limites_dominio, num_max_iteracoes, paciencia_parada_antecipada, sigma, eh_maximizacao=False):
    solucao_atual_x = np.array([np.random.uniform(lim[0], lim[1]) for lim in limites_dominio])
    solucao_atual_f = funcao_obj(solucao_atual_x[0], solucao_atual_x[1])
    if eh_maximizacao:
        solucao_atual_f = -solucao_atual_f
        
    melhor_solucao_x = np.copy(solucao_atual_x)
    melhor_solucao_f = solucao_atual_f
    sequencia_sem_melhoria = 0

    for _ in range(num_max_iteracoes):
        candidato_x = solucao_atual_x + np.random.normal(0, sigma, size=len(solucao_atual_x))
        candidato_x = verificar_limites(candidato_x, limites_dominio)
        candidato_f = funcao_obj(candidato_x[0], candidato_x[1])
        if eh_maximizacao:
            candidato_f = -candidato_f
            
        if candidato_f < solucao_atual_f: 
            solucao_atual_x = candidato_x
            solucao_atual_f = candidato_f
            sequencia_sem_melhoria = 0
            if solucao_atual_f < melhor_solucao_f: 
                melhor_solucao_x = np.copy(solucao_atual_x)
                melhor_solucao_f = solucao_atual_f
        else:
            sequencia_sem_melhoria += 1
            
        if sequencia_sem_melhoria >= paciencia_parada_antecipada:
            break
            
    if eh_maximizacao:
        melhor_solucao_f = -melhor_solucao_f
    return melhor_solucao_x, melhor_solucao_f

def busca_global_aleatoria(funcao_obj, limites_dominio, num_max_iteracoes, paciencia_parada_antecipada, eh_maximizacao=False):
    melhor_solucao_x = np.array([np.random.uniform(lim[0], lim[1]) for lim in limites_dominio])
    melhor_solucao_f = funcao_obj(melhor_solucao_x[0], melhor_solucao_x[1])
    if eh_maximizacao:
        melhor_solucao_f = -melhor_solucao_f

    sequencia_sem_melhoria = 0
    
    for _ in range(num_max_iteracoes):
        candidato_x = np.array([np.random.uniform(lim[0], lim[1]) for lim in limites_dominio])
        candidato_x = verificar_limites(candidato_x, limites_dominio)
        candidato_f = funcao_obj(candidato_x[0], candidato_x[1])
        if eh_maximizacao:
            candidato_f = -candidato_f
            
        if candidato_f < melhor_solucao_f:
            melhor_solucao_x = candidato_x
            melhor_solucao_f = candidato_f
            sequencia_sem_melhoria = 0
        else:
            sequencia_sem_melhoria +=1

        if sequencia_sem_melhoria >= paciencia_parada_antecipada:
            break
            
    if eh_maximizacao:
        melhor_solucao_f = -melhor_solucao_f
    return melhor_solucao_x, melhor_solucao_f

# --- Configuração do Experimento ---
problemas = [
    {"nome": "F1 (Esfera)", "funcao": f1, "limites": [(-100, 100), (-100, 100)], "tipo": "min", "coord_otima": (0,0), "valor_otimo": 0},
    {"nome": "F2 (Tipo Himmelblau)", "funcao": f2, "limites": [(-2, 4), (-2, 5)], "tipo": "max", "coord_otima": (1.7,1.7), "valor_otimo": f2(1.7,1.7)},
    {"nome": "F3 (Ackley)", "funcao": f3, "limites": [(-8, 8), (-8, 8)], "tipo": "min", "coord_otima": (0,0), "valor_otimo": 0},
    {"nome": "F4 (Rastrigin)", "funcao": f4, "limites": [(-5.12, 5.12), (-5.12, 5.12)], "tipo": "min", "coord_otima": (0,0), "valor_otimo": 0},
    {"nome": "F5 (Complexa)", "funcao": f5, "limites": [(-10, 10), (-10, 10)], "tipo": "max"},
    {"nome": "F6 (Senoidal)", "funcao": f6, "limites": [(-1, 3), (-1, 3)], "tipo": "max"},
    {"nome": "F7 (Michalewicz)", "funcao": f7, "limites": [(0, np.pi), (0, np.pi)], "tipo": "min", "valor_otimo": -1.8013},
    {"nome": "F8 (Tipo Eggholder)", "funcao": f8, "limites": [(-200, 20), (-200, 20)], "tipo": "min"}
]

NUM_MAX_ITERACOES = 1000
PACIENCIA_PARADA_ANTECIPADA = 100 
NUM_RODADAS = 100 
EPSILON_HC = 0.1 
SIGMA_LRS = 0.5 
resumo_resultados = []

# --- Executar Experimentos ---
for info_problema in problemas:
    print(f"\n--- Otimizando: {info_problema['nome']} ---")
    print(f"Tipo: {info_problema['tipo']}, Limites: {info_problema['limites']}")
    if "valor_otimo" in info_problema:
        coord_otima_str = str(info_problema.get('coord_otima', 'N/A'))
        print(f"Valor ótimo conhecido: {info_problema['valor_otimo']:.4f} em {coord_otima_str}")
    
    eh_maximizacao_problema = (info_problema["tipo"] == "max")
    
    # <<< MODIFICAÇÃO AQUI: PLOTAR O GRÁFICO DA FUNÇÃO >>>
    plotar_funcao(info_problema["funcao"], info_problema["limites"], info_problema["nome"])
    # <<< FIM DA MODIFICAÇÃO >>>

    resultados_algoritmo_problema = {}

    for nome_algoritmo, funcao_algoritmo_atual, parametros_algo in [
        ("Subida de Encosta", subida_de_encosta, {"epsilon": EPSILON_HC}),
        ("Busca Local Aleatória", busca_local_aleatoria, {"sigma": SIGMA_LRS}),
        ("Busca Global Aleatória", busca_global_aleatoria, {})
    ]:
        print(f"  Executando {nome_algoritmo}...")
        solucoes_f_rodada_atual = []
        solucoes_x_rodada_atual = []

        for idx_rodada in range(NUM_RODADAS):
            if idx_rodada > 0 and idx_rodada % (NUM_RODADAS // 10) == 0 : 
                 print(f"    {nome_algoritmo} - Rodada {idx_rodada}/{NUM_RODADAS}")

            if nome_algoritmo == "Subida de Encosta":
                melhor_x, melhor_f = funcao_algoritmo_atual(
                    info_problema["funcao"], info_problema["limites"], NUM_MAX_ITERACOES, PACIENCIA_PARADA_ANTECIPADA,
                    parametros_algo["epsilon"], eh_maximizacao=eh_maximizacao_problema
                )
            elif nome_algoritmo == "Busca Local Aleatória":
                 melhor_x, melhor_f = funcao_algoritmo_atual(
                    info_problema["funcao"], info_problema["limites"], NUM_MAX_ITERACOES, PACIENCIA_PARADA_ANTECIPADA,
                    parametros_algo["sigma"], eh_maximizacao=eh_maximizacao_problema
                )
            elif nome_algoritmo == "Busca Global Aleatória":
                 melhor_x, melhor_f = funcao_algoritmo_atual(
                    info_problema["funcao"], info_problema["limites"], NUM_MAX_ITERACOES, PACIENCIA_PARADA_ANTECIPADA,
                    eh_maximizacao=eh_maximizacao_problema
                )
            solucoes_f_rodada_atual.append(melhor_f)
            solucoes_x_rodada_atual.append(melhor_x)

        valores_f_arredondados_atuais = [round(f_val, 6) for f_val in solucoes_f_rodada_atual] 
        
        moda_f_calculada = float('nan')
        if not valores_f_arredondados_atuais:
            media_f_calculada = float('nan')
            melhor_f_geral_rodadas = float('nan')
            melhor_x_geral_rodadas = (float('nan'), float('nan'))
        else:
            media_f_calculada = np.mean(solucoes_f_rodada_atual)
            if eh_maximizacao_problema:
                melhor_idx_geral = np.argmax(solucoes_f_rodada_atual)
            else:
                melhor_idx_geral = np.argmin(solucoes_f_rodada_atual)
            melhor_f_geral_rodadas = solucoes_f_rodada_atual[melhor_idx_geral]
            melhor_x_geral_rodadas = solucoes_x_rodada_atual[melhor_idx_geral]

            contagens = {}
            for valor in valores_f_arredondados_atuais:
                contagens[valor] = contagens.get(valor, 0) + 1
            
            if contagens:
                max_contagem = -1
                for valor, contagem_atual_item in contagens.items():
                    if contagem_atual_item > max_contagem:
                        max_contagem = contagem_atual_item
                        moda_f_calculada = valor
            
        resultados_algoritmo_problema[nome_algoritmo] = {
            "moda_f": moda_f_calculada, 
            "media_f": media_f_calculada,
            "melhor_f_geral": melhor_f_geral_rodadas,
            "melhor_x_geral": melhor_x_geral_rodadas
            }
        print(f"    {nome_algoritmo} - Moda f(x): {moda_f_calculada:.4f}, Média f(x): {media_f_calculada:.4f}, Melhor f(x) em {NUM_RODADAS} rodadas: {melhor_f_geral_rodadas:.4f} em x={np.round(melhor_x_geral_rodadas,3)}")

    resumo_resultados.append({
        "nome_problema": info_problema["nome"],
        "algoritmos": resultados_algoritmo_problema,
        "otimo_conhecido": info_problema.get("valor_otimo", "N/A")
    })

# --- Exibir Tabela Final ---
print("\n\n--- Resumo do Experimento ---")
print(f"{'Problema':<25} | {'Algoritmo':<25} | {'Moda f(x)':<12} | {'Média f(x)':<12} | {'Melhor f(x) Geral':<18} | {'Melhor x':<20} | {'Ótimo Conh.':<10}")
print("-" * 135)

for res in resumo_resultados:
    nome_problema_atual = res["nome_problema"]
    otimo_conhecido_str = f"{res['otimo_conhecido']:.4f}" if isinstance(res['otimo_conhecido'], (int, float)) else str(res['otimo_conhecido'])
    
    primeiro_algoritmo = True
    for nome_algoritmo_res, dados_algo in res["algoritmos"].items():
        melhor_x_str = str(np.round(dados_algo['melhor_x_geral'],3)) if isinstance(dados_algo['melhor_x_geral'], np.ndarray) else str(dados_algo['melhor_x_geral'])
        if primeiro_algoritmo:
            print(f"{nome_problema_atual:<25} | {nome_algoritmo_res:<25} | {dados_algo['moda_f']:<12.4f} | {dados_algo['media_f']:<12.4f} | {dados_algo['melhor_f_geral']:<18.4f} | {melhor_x_str:<20} | {otimo_conhecido_str:<10}")
            primeiro_algoritmo = False
        else:
            print(f"{'':<25} | {nome_algoritmo_res:<25} | {dados_algo['moda_f']:<12.4f} | {dados_algo['media_f']:<12.4f} | {dados_algo['melhor_f_geral']:<18.4f} | {melhor_x_str:<20} | {'':<10}")
    print("-" * 135)