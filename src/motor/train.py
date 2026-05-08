import pandas as pd
import os
from tqdm import tqdm
from src.utils import *
import warnings
import time
import json

warnings.filterwarnings("ignore")

targets = [1.01, 1.015, 1.02]
janelas_tamanho = [60, 75, 90]

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


# ================= ATIVOS AUTOMÁTICOS =================

def obter_ativos(caminho="./data/pre_process/curated/"):

    arquivos = os.listdir(caminho)
    ativos = set()

    for nome in arquivos:
        if nome.endswith(".csv") and "_target_" in nome:
            ativo = nome.split("_target_")[0]
            ativos.add(ativo)

    return sorted(list(ativos))


# ================= CHECKPOINT =================

def salvar_checkpoint(path, estado):
    with open(path, "w") as f:
        json.dump(estado, f)


def carregar_checkpoint(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def salvar_registros(caminho, registros, subset):
    if not registros:
        return

    df_novo = pd.DataFrame(registros)

    if os.path.exists(caminho):
        df_atual = pd.read_csv(caminho)
        df = pd.concat([df_atual, df_novo], ignore_index=True)
    else:
        df = df_novo

    for coluna in subset:
        if "data" in coluna.lower() and coluna in df.columns:
            df[coluna] = pd.to_datetime(df[coluna], errors="coerce").dt.strftime("%Y-%m-%d")

    df = df.drop_duplicates(subset=subset, keep="last")
    df.to_csv(caminho, index=False)


def salvar_progresso(file, registros_out, registros_in):
    caminho_out = f"./data/train/outputs/target_previsto_{file}.csv"
    caminho_in = f"./data/train/inputs/target_in_{file}.csv"

    salvar_registros(
        caminho_out,
        registros_out,
        subset=["ativo", "target", "janela", "tecnica", "data"]
    )

    salvar_registros(
        caminho_in,
        registros_in,
        subset=[
            "ativo",
            "target",
            "janela",
            "tecnica",
            "data_inicio_janela",
            "data_final_janela"
        ]
    )

    registros_out.clear()
    registros_in.clear()


def datas_esperadas(base_dados, janela):
    return set(
        pd.to_datetime(
            base_dados['Exchange Date'].iloc[janela + 1:],
            errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    )


def combinacao_completa(file, target, janela, base_dados):
    caminho_out = f"./data/train/outputs/target_previsto_{file}.csv"

    if not os.path.exists(caminho_out):
        return False

    esperado = datas_esperadas(base_dados, janela)

    if not esperado:
        return True

    try:
        df = pd.read_csv(caminho_out, usecols=["target", "janela", "tecnica", "data"])
    except ValueError:
        return False

    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df["janela"] = pd.to_numeric(df["janela"], errors="coerce")

    df = df[
        (df["target"].round(3) == round(target, 3)) &
        (df["janela"] == janela)
    ].copy()

    if df.empty:
        return False

    df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.strftime("%Y-%m-%d")

    for tecnica in ["RNA", "Random Forest", "SVC"]:
        datas_tecnica = set(df.loc[df["tecnica"] == tecnica, "data"].dropna())

        if not esperado.issubset(datas_tecnica):
            return False

    return True


# ================= MAIN =================

def main():

    os.makedirs("./data/train/outputs", exist_ok=True)
    os.makedirs("./data/train/inputs", exist_ok=True)

    checkpoint_path = "./data/train/checkpoint.json"
    todos = obter_ativos()

    intervalo_salvamento = 1800
    ultimo_salvamento = time.time()

    checkpoint = carregar_checkpoint(checkpoint_path)

    print("\nIniciando modelagem e avaliação...\n")

    total_ativos = len(todos)

    for idx_file, file in enumerate(todos):

        print(f"\n🔹 Ativo {idx_file + 1}/{total_ativos}: {file}")

        todos_registros_out = []
        todos_registros_in = []

        for idx_target, target in enumerate(targets):

            curated_path = f"./data/pre_process/curated/{file}_target_{target}.csv"

            # 🔥 evita erro se não existir
            if not os.path.exists(curated_path):
                print(f"⚠️ Arquivo não encontrado: {curated_path}")
                continue

            base_dados = pd.read_csv(curated_path)

            # ✅ garante datetime
            base_dados['Exchange Date'] = pd.to_datetime(base_dados['Exchange Date'])

            x_dados = x_split(base_dados)
            y_dados = y_split(base_dados)
            z_dados = z_split(base_dados)

            for idx_janela, janela in enumerate(janelas_tamanho):

                qtd_treinamentos = len(base_dados) - janela - 1

                if combinacao_completa(file, target, janela, base_dados):
                    print(f"✅ Já completo: {file} | target {target} | janela {janela}")
                    continue

                print(f"▶ Treinando: {file} | target {target} | janela {janela} | {qtd_treinamentos} datas")

                # 🔥 CACHE FEATURE SELECTION
                features_por_ano = {}
                inicio_combinacao = time.time()
                intervalo_log = max(250, qtd_treinamentos // 10)

                for i in range(qtd_treinamentos):

                    # ================= DADOS =================

                    x_janela_atual = x_dados[i:i+janela]
                    y_janela_atual = y_dados[i:i+janela]

                    x_teste = x_dados.iloc[i+janela:i+janela+1]

                    data_inicio_janela = z_dados.at[i, 'Exchange Date']
                    data_final_janela = z_dados.at[i+janela-1, 'Exchange Date']
                    data_atual = z_dados.at[i+janela+1, 'Exchange Date']

                    if i == 0 or (i + 1) == qtd_treinamentos or (i + 1) % intervalo_log == 0:
                        percentual = ((i + 1) / qtd_treinamentos) * 100
                        print(
                            f"   andamento {file} | target {target} | janela {janela}: "
                            f"{i + 1}/{qtd_treinamentos} ({percentual:.1f}%) | data {data_atual.date()}"
                        )

                    ano_atual = data_atual.year

                    resultado_real = z_dados.at[i+janela+1, 'resultado_real']
                    target_real = z_dados.at[i+janela+1, 'target']
                    y_real_atual = y_dados[i+janela+1]

                    # ================= FEATURE SELECTION =================

                    chave_fs = (ano_atual, janela)

                    if chave_fs not in features_por_ano:

                        x_janela_filtrada = features_selection(
                            x_janela_atual,
                            correlation_threshold=0.8
                        )

                        features_por_ano[chave_fs] = x_janela_filtrada.columns

                    else:
                        colunas = features_por_ano[chave_fs]
                        x_janela_filtrada = x_janela_atual[colunas]

                    # ================= TREINO =================

                    x_treino = x_janela_filtrada.values
                    y_treino = y_janela_atual

                    x_teste = x_teste[x_janela_filtrada.columns].values

                    resultados_parciais = treinar_e_avaliar(
                        file,
                        x_treino, y_treino,
                        x_teste, y_real_atual,
                        parametros_rna,
                        parametros_rf,
                        parametros_svc,
                        ano_atual=ano_atual,
                        janela=janela
                    )

                    # ================= RESULTADOS =================

                    for modelo in resultados_parciais:

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

                    # ================= CHECKPOINT =================

                    tempo_atual = time.time()

                    if tempo_atual - ultimo_salvamento >= intervalo_salvamento:

                        print("\n💾 Salvando checkpoint...")

                        salvar_progresso(
                            file,
                            todos_registros_out,
                            todos_registros_in
                        )

                        estado = {
                            "file": idx_file,
                            "target": idx_target,
                            "janela": idx_janela,
                            "i": i + 1
                        }

                        salvar_checkpoint(checkpoint_path, estado)

                        ultimo_salvamento = tempo_atual

                        print("✅ Checkpoint salvo!")

                duracao = time.time() - inicio_combinacao
                print(f"✅ Concluído: {file} | target {target} | janela {janela} | {duracao / 60:.1f} min")

        salvar_progresso(
            file,
            todos_registros_out,
            todos_registros_in
        )

        print(f"\n✔ Finalizado ativo: {file}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print("\n🎯 Modelagem concluída!")


if __name__ == "__main__":
    main()
