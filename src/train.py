import pandas as pd
import numpy as np
import warnings
import os
from tqdm import tqdm
from src.utils import *
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

todos = ['PETR4', 'ITUB4', 'VALE3']
targets = [1.01, 1.015, 1.02]
ibov = 'ibov_filtrado'

janela_temporal = 4
janelas_tamanho = [15, 20, 25]

# Hiperparâmetros para rna
parametros_rna = {
    'hidden_layer_sizes': [(10,), (10, 10), (20, 20)],  # Número de neurônios e camadas
    'activation': ['relu', 'tanh'],  # Função de ativação
    'solver': ['adam', 'sgd'],  # Algoritmo de otimização
    'max_iter': [1000, 2000],  # Número máximo de iterações
}

# Hiperparâmetros para Random Forest
parametros_rf = {
    'n_estimators': [50, 100, 200],  # Número de árvores
    'max_depth': [None, 10, 20],  # Profundidade máxima das árvores
    'min_samples_split': [2, 5, 10],  # Número mínimo de amostras para dividir um nó
}

# Hiperparâmetros para SVC
parametros_svc = {
    'C': [0.1, 1, 10],  # Parâmetro de regularização
    'kernel': ['linear', 'rbf'],  # Tipo de kernel
    'gamma': ['scale', 'auto'],  # Coeficiente do kernel
}

def main():
    ibov_status = carregar_ibov(ibov)
    base_petr4 = carregar_dados('PETR4')

    os.makedirs("./data/raw", exist_ok=True)
    os.makedirs("./data/curated", exist_ok=True)
    os.makedirs("./data/results", exist_ok=True)

    print("\n Iniciando processamento...\n")

    for file in tqdm(todos, desc="Ativos", unit="ativo"):
        # Lista para armazenar os registros detalhados de cada ativo
        todos_registros = []

        # ===== 1º Salvamento: RAW =====
        base_dados_original = carregar_dados(file)
        base_dados_original = padronizar_colunas(base_dados_original)
        base_dados_original = remover_linhas_invalidas(base_dados_original)

        base_dados_original = pd.merge(base_petr4[['Exchange Date']], base_dados_original, on='Exchange Date', how='left')
        base_dados_original['data_adicional'] = np.where(base_dados_original['Open'].isnull(), 1, 0)
        base_dados_original = base_dados_original.ffill()

        base_dados_original = pd.merge(base_dados_original, ibov_status[['Exchange Date', file]], on='Exchange Date', how='left')
        base_dados_original.rename(columns={file: 'ibov_status'}, inplace=True)
        base_dados_original.to_csv(f"./data/raw/ibov_status/{file}.csv", index=False)

        # ===== Processamento por target =====
        for target in tqdm(targets, desc=f"Targets de {file}", leave=False):
            base_dados = base_dados_original.copy()
            base_dados = base_dados[base_dados['ibov_status'] == 1].copy()
            base_dados = calcular_variaveis(base_dados)
            base_dados = calcular_target(base_dados, target, janela_temporal)
            base_dados = base_dados.iloc[14:]

            curated_path = f"./data/curated/{file}_target_{target}.csv"
            base_dados.to_csv(curated_path, index=False)

            x_dados = x_split(base_dados)
            y_dados = y_split(base_dados)
            z_dados = z_split(base_dados)

            # ===== Processamento por janela =====
            for janela in tqdm(janelas_tamanho, desc=f"Janelas - {file} T{target}", leave=False):
                qtd_treinamentos = len(base_dados) - janela - 1

                resultados_acumulados = {
                    'RNA': {'previsoes': [], 'y_real': [], 'probabilidades': []},
                    'Random Forest': {'previsoes': [], 'y_real': [], 'probabilidades': []},
                    'SVC': {'previsoes': [], 'y_real': [], 'probabilidades': []}
                }

                # registros detalhados da janela
                registros = []

                for i in range(qtd_treinamentos):
                    x_janela_atual = x_dados[i:i+janela]
                    y_janela_atual = y_dados[i:i+janela]

                    x_janela_atual_filtrado = features_selection(x_janela_atual, correlation_threshold=0.8)
                    x_treino = x_janela_atual_filtrado.values
                    y_treino = y_janela_atual

                    x_teste = x_dados.iloc[i+janela:i+janela+1]
                    x_teste = x_teste[x_janela_atual_filtrado.columns].values

                    # data e target real
                    data_atual = z_dados.at[i+janela+1, 'Exchange Date']
                    resultado_real = z_dados.at[i+janela+1, 'resultado_real']
                    target_real = z_dados.at[i+janela+1, 'target']
                    y_real_atual = y_dados[i+janela+1]

                    # treinar e avaliar (ajuste de acordo com as técnicas que deseja)
                    resultados_parciais = treinar_e_avaliar(
                        x_treino,
                        y_treino,
                        x_teste,
                        y_real_atual,
                        parametros_rna,
                        parametros_rf,
                        parametros_svc
                    )

                    # salvar resultados acumulados e detalhados
                    for modelo in resultados_parciais:
                        resultados_acumulados[modelo]['previsoes'].extend(resultados_parciais[modelo]['previsoes'])
                        resultados_acumulados[modelo]['y_real'].extend(resultados_parciais[modelo]['y_real'])
                        resultados_acumulados[modelo]['probabilidades'].extend(resultados_parciais[modelo]['probabilidades'])

                        for pred, real in zip(resultados_parciais[modelo]['previsoes'], resultados_parciais[modelo]['y_real']):
                            registros.append({
                                "ativo": file,
                                "target": target,
                                "janela": janela,
                                "tecnica": modelo,
                                "data": data_atual,
                                "target_real": target_real,
                                "target_pred": pred,
                                 "resultado_real": resultado_real
                            })

                # exibir resultados da janela
                print(f"\nAtivo: {file} - Target: {target} - Janela {janela}")
                exibir_resultados(resultados_acumulados, janela, file, target, "./data/analytics/results/resultados_finais1.csv")

                # acumula os registros da janela no total do ativo
                todos_registros.extend(registros)

        # ===== Salvar CSV detalhado do ativo =====
        df_registros_ativo = pd.DataFrame(todos_registros)
        caminho_arquivo = f"./data/train_out/target_previsto_{file}.csv"
        df_registros_ativo.to_csv(caminho_arquivo, index=False)
        print(f"\nResultados detalhados de {file} salvos em {caminho_arquivo}")

    print("\nProcessamento finalizado!")

main()
