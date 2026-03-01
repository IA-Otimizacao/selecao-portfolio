import pandas as pd
import os
from tqdm import tqdm
from src.utils import *
import warnings

warnings.filterwarnings("ignore")

todos = ['PETR4', 'ITUB4', 'VALE3']
targets = [1.01, 1.015, 1.02]
janelas_tamanho = [60, 75, 90]


# Hiperparâmetros para os modelos de Machine Learning
parametros_rna = {
    'hidden_layer_sizes': [(10,), (10, 10), (20, 20)],  # arquitetura das camadas
    'activation': ['relu', 'tanh'],                     # função de ativação
    'solver': ['adam', 'sgd'],                          # algoritmo de otimização
    'max_iter': [1000, 2000],                           # número máximo de iterações
}


parametros_rf = {
    'n_estimators': [50, 100, 200],   # número de árvores
    'max_depth': [None, 10, 20],      # profundidade máxima
    'min_samples_split': [2, 5, 10],  # mínimo de amostras para dividir nó
}

parametros_svc = {
    'C': [0.1, 1, 10],                # parâmetro de regularização
    'kernel': ['linear', 'rbf'],      # tipo de kernel
    'gamma': ['scale', 'auto'],       # coeficiente do kernel
}

def main():

    os.makedirs("./data/train", exist_ok=True)
    os.makedirs("./data/train/analytics/results", exist_ok=True)

    print("\nIniciando modelagem e avaliação...\n")

    # Loop principal para cada ativo
    for file in tqdm(todos, desc="Ativos", unit="ativo"):

        # Listas que armazenarão todos os resultados
        todos_registros_out = []
        todos_registros_in = []

        # Loop para cada target de retorno
        for target in tqdm(targets, desc=f"Targets de {file}", leave=False):

            # Carrega base curated gerada anteriormente
            curated_path = f"./data/curated/{file}_target_{target}.csv"
            base_dados = pd.read_csv(curated_path)

            # Separa variáveis explicativas (features)
            x_dados = x_split(base_dados)

            # Separa variável target
            y_dados = y_split(base_dados)

            # Separa variáveis auxiliares (datas e resultados reais)
            z_dados = z_split(base_dados)

            # Loop para diferentes tamanhos de janela temporal
            for janela in tqdm(janelas_tamanho, desc=f"Janelas - {file} T{target}", leave=False):

                # Quantidade de treinamentos possíveis com janela deslizante
                qtd_treinamentos = len(base_dados) - janela - 1

                # Walk-forward validation (janela deslizante)
                for i in range(qtd_treinamentos):

                    # Dados de treino da janela atual
                    x_janela_atual = x_dados[i:i+janela]
                    y_janela_atual = y_dados[i:i+janela]

                    # Seleção de features removendo alta correlação
                    x_janela_filtrada = features_selection(
                        x_janela_atual,
                        correlation_threshold=0.8
                    )

                    # Converte para arrays usados pelos modelos
                    x_treino = x_janela_filtrada.values
                    y_treino = y_janela_atual

                    # Define o ponto de teste (próximo dia)
                    x_teste = x_dados.iloc[i+janela:i+janela+1]

                    # Mantém apenas as mesmas features selecionadas
                    x_teste = x_teste[x_janela_filtrada.columns].values

                    # Informações de data para análise posterior
                    data_inicio_janela = z_dados.at[i, 'Exchange Date']
                    data_final_janela = z_dados.at[i+janela-1, 'Exchange Date']
                    data_atual = z_dados.at[i+janela+1, 'Exchange Date']

                    # Resultados reais do mercado
                    resultado_real = z_dados.at[i+janela+1, 'resultado_real']
                    target_real = z_dados.at[i+janela+1, 'target']
                    y_real_atual = y_dados[i+janela+1]

                    # Treina modelos e gera previsões
                    resultados_parciais = treinar_e_avaliar(
                        file,
                        x_treino, y_treino,
                        x_teste, y_real_atual,
                        parametros_rna,
                        parametros_rf,
                        parametros_svc
                    )

                    # Armazena resultados de cada modelo
                    for modelo in resultados_parciais:

                        # ---------------- OUT-OF-SAMPLE -----------------
                        for pred, real in zip(
                                resultados_parciais[modelo]['previsoes'],
                                resultados_parciais[modelo]['y_real']):

                            todos_registros_out.append({
                                "ativo": file,
                                "target": target,
                                "janela": janela,
                                "tecnica": modelo,
                                "data": data_atual,
                                "target_real": target_real,
                                "target_pred": pred,
                                "resultado_real": resultado_real
                            })

                        # ---------------- IN-SAMPLE -----------------
                        y_treino_vals = list(
                            resultados_parciais[modelo].get('y_treino', [])
                        )

                        y_in_vals = list(
                            resultados_parciais[modelo].get('y_in', [])
                        )

                        precision_in = resultados_parciais[modelo].get(
                            'precision_in', None
                        )

                        todos_registros_in.append({
                            "ativo": file,
                            "target": target,
                            "janela": janela,
                            "data_inicio_janela": data_inicio_janela,
                            "data_final_janela": data_final_janela,
                            "tecnica": modelo,
                            "y_treino": y_treino_vals,
                            "y_in": y_in_vals,
                            "precision": precision_in
                        })

                print(f"\nAtivo: {file} - Target: {target} - Janela {janela}")


        # Salva resultados OUT-OF-SAMPLE
        df_out = pd.DataFrame(todos_registros_out)

        caminho_out = f"./data/train/outputs/target_previsto_{file}.csv"

        df_out.to_csv(caminho_out, index=False)

        print(f"Resultados out-of-sample salvos em {caminho_out}")

        # Salva resultados IN-SAMPLE
        df_in = pd.DataFrame(todos_registros_in)

        caminho_in = f"./data/train/inputs/target_in_{file}.csv"

        df_in.to_csv(caminho_in, index=False)

        print(f"Resultados in-sample salvos em {caminho_in}")

    print("\nModelagem concluída!")


if __name__ == "__main__":
    main()