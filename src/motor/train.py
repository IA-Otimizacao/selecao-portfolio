import pandas as pd
import os
from tqdm import tqdm
from src.utils import *
import warnings
warnings.filterwarnings("ignore")

todos = ['PETR4', 'ITUB4', 'VALE3']
targets = [1.01, 1.015, 1.02]
janelas_tamanho = [60, 75, 90]

# Hiperparâmetros
parametros_rna = {
    'hidden_layer_sizes': [(10,), (10, 10), (20, 20)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'max_iter': [1000, 2000],
}

parametros_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

parametros_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
}

def main():
    os.makedirs("./data/train", exist_ok=True)
    os.makedirs("./data/train/analytics/results", exist_ok=True)

    print("\nIniciando modelagem e avaliação...\n")

    for file in tqdm(todos, desc="Ativos", unit="ativo"):
        todos_registros_out = []
        todos_registros_in = []

        for target in tqdm(targets, desc=f"Targets de {file}", leave=False):
            curated_path = f"./data/curated/{file}_target_{target}.csv"
            base_dados = pd.read_csv(curated_path)

            x_dados = x_split(base_dados)
            y_dados = y_split(base_dados)
            z_dados = z_split(base_dados)

            for janela in tqdm(janelas_tamanho, desc=f"Janelas - {file} T{target}", leave=False):
                qtd_treinamentos = len(base_dados) - janela - 1

                for i in range(qtd_treinamentos):
                    x_janela_atual = x_dados[i:i+janela]
                    y_janela_atual = y_dados[i:i+janela]

                    x_janela_filtrada = features_selection(x_janela_atual, correlation_threshold=0.8)
                    x_treino = x_janela_filtrada.values
                    y_treino = y_janela_atual

                    x_teste = x_dados.iloc[i+janela:i+janela+1]
                    x_teste = x_teste[x_janela_filtrada.columns].values

                    data_inicio_janela = z_dados.at[i, 'Exchange Date']
                    data_final_janela = z_dados.at[i+janela-1, 'Exchange Date']
                    data_atual = z_dados.at[i+janela+1, 'Exchange Date']
                    resultado_real = z_dados.at[i+janela+1, 'resultado_real']
                    target_real = z_dados.at[i+janela+1, 'target']
                    y_real_atual = y_dados[i+janela+1]

                    resultados_parciais = treinar_e_avaliar(
                        file,
                        x_treino, y_treino,
                        x_teste, y_real_atual,
                        parametros_rna, parametros_rf, parametros_svc
                    )

                    for modelo in resultados_parciais:
                        # --- OUT-OF-SAMPLE ---
                        for pred, real in zip(resultados_parciais[modelo]['previsoes'], resultados_parciais[modelo]['y_real']):
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

                        # --- IN-SAMPLE ---
                        y_treino_vals = list(resultados_parciais[modelo].get('y_treino', []))
                        y_in_vals = list(resultados_parciais[modelo].get('y_in', []))
                        precision_in = resultados_parciais[modelo].get('precision_in', None)

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

        # Salvar OUT-OF-SAMPLE
        df_out = pd.DataFrame(todos_registros_out)
        caminho_out = f"./data/train/outputs/target_previsto_{file}.csv"
        df_out.to_csv(caminho_out, index=False)
        print(f"Resultados out-of-sample salvos em {caminho_out}")

        # Salvar IN-SAMPLE
        df_in = pd.DataFrame(todos_registros_in)
        caminho_in = f"./data/train/inputs/target_in_{file}.csv"
        df_in.to_csv(caminho_in, index=False)
        print(f"Resultados in-sample salvos em {caminho_in}")

    print("\nModelagem concluída!")

if __name__ == "__main__":
    main()