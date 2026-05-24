import os
import re
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TECNICA_ALEATORIA = "aleatorio"
SEED_PADRAO = 42


def extrair_ativo(nome_arquivo):
    match = re.match(r"target_previsto_(.+)\.csv$", nome_arquivo)

    if not match:
        raise ValueError(f"Ativo nao encontrado no arquivo: {nome_arquivo}")

    return match.group(1)


def formatar_target_decimal(target):
    return f"{float(target):.12g}"


def formatar_target_arquivo(target):
    return formatar_target_decimal(target).replace(".", "_")


def extrair_target_nome(nome_arquivo):
    match = re.search(r"target_(\d+[_\.]\d+)", nome_arquivo)

    if not match:
        raise ValueError(f"Target nao encontrado no arquivo: {nome_arquivo}")

    return float(match.group(1).replace("_", "."))


def valor_sinal(valor):
    if pd.isna(valor):
        return 0

    return int(float(valor) == 1.0)


def valor_float(valor, padrao=0.0):
    if pd.isna(valor):
        return padrao

    try:
        return float(valor)
    except (TypeError, ValueError):
        return padrao


def listar_outputs(input_folder):
    return sorted([
        arquivo
        for arquivo in os.listdir(input_folder)
        if arquivo.startswith("target_previsto_") and arquivo.endswith(".csv")
    ])


def gerar_outputs_aleatorios(
    input_folder="./data/train/outputs/",
    output_folder="./data/train/random/",
    seed=SEED_PADRAO
):
    os.makedirs(output_folder, exist_ok=True)
    rng = np.random.default_rng(seed)

    arquivos = listar_outputs(input_folder)

    for arquivo in arquivos:
        caminho = os.path.join(input_folder, arquivo)
        df = pd.read_csv(caminho)

        if "target_pred" not in df.columns:
            print(f"Pulando {arquivo}: coluna target_pred nao encontrada.")
            continue

        df["target_pred"] = rng.integers(0, 2, size=len(df), dtype=np.int8)

        output_path = os.path.join(output_folder, arquivo)
        df.to_csv(output_path, index=False)
        print(f"Random salvo: {output_path}")

    print("\nArquivos random criados em data/train/random.")


def carregar_curated(ativo, target, curated_folder):
    target_decimal = formatar_target_decimal(target)
    caminho = os.path.join(curated_folder, f"{ativo}_target_{target_decimal}.csv")

    if not os.path.exists(caminho):
        print(f"  Aviso: curated nao encontrado para {ativo} target {target_decimal}.")
        return None

    df = pd.read_csv(caminho, usecols=["Exchange Date", "Open", "High", "Close"])
    df["Exchange Date"] = pd.to_datetime(df["Exchange Date"], errors="coerce")

    for coluna in ["Open", "High", "Close"]:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    df = df.dropna(subset=["Exchange Date"])
    df = df.drop_duplicates(subset=["Exchange Date"], keep="last")

    return df


def preparar_sinais_ativo(caminho_random):
    df = pd.read_csv(
        caminho_random,
        usecols=["target", "janela", "tecnica", "data", "target_pred"]
    )

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["target_pred"] = (
        pd.to_numeric(df["target_pred"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    df = df.dropna(subset=["data", "target"])
    df = df.sort_values(["target", "data", "janela", "tecnica"])

    # Uma moeda por ativo/data/target. Os arquivos random continuam completos,
    # mas a carteira usa uma decisao unica por dia para cada ativo.
    df = df.drop_duplicates(subset=["target", "data"], keep="first")
    df = df.rename(columns={"target_pred": TECNICA_ALEATORIA})

    return df[["target", "data", TECNICA_ALEATORIA]]


def calcular_retornos_aleatorios(df_sinais, ativo, target, curated_folder):
    df_curated = carregar_curated(ativo, target, curated_folder)

    if df_curated is None:
        return None

    df = df_sinais.merge(
        df_curated,
        left_on="data",
        right_on="Exchange Date",
        how="left"
    )
    df = df.drop(columns=["Exchange Date"])
    df = df.sort_values("data").reset_index(drop=True)

    compra = np.zeros(len(df))
    rend_decisao = np.zeros(len(df))
    rend_venda = np.zeros(len(df))
    dias = np.zeros(len(df))

    preco_compra = 0.0
    contador = 0
    sinal_shift = df[TECNICA_ALEATORIA].shift(1)
    threshold = float(target) - 1

    for i in range(len(df)):
        if contador == 0:
            open_price = valor_float(df.loc[i, "Open"])

            if valor_sinal(sinal_shift.iloc[i]) == 1 and open_price > 0:
                preco_compra = open_price
                compra[i] = preco_compra
                contador = 1
            else:
                dias[i] = 0
                continue
        else:
            compra[i] = preco_compra

        high_price = valor_float(df.loc[i, "High"])
        close_price = valor_float(df.loc[i, "Close"])

        if preco_compra > 0 and high_price > 0:
            rend_decisao[i] = (high_price - preco_compra) / preco_compra

        if preco_compra > 0 and close_price > 0:
            rend_venda[i] = (close_price - preco_compra) / preco_compra

        if rend_decisao[i] >= threshold:
            dias[i] = 0
            contador = 0
            preco_compra = 0.0
        else:
            dias[i] = contador
            contador += 1

            if contador > 4:
                contador = 0
                preco_compra = 0.0

    saida = df[["data", TECNICA_ALEATORIA]].copy()
    saida[f"rend_decisao_{TECNICA_ALEATORIA}"] = rend_decisao
    saida[f"rend_venda_{TECNICA_ALEATORIA}"] = rend_venda
    saida[f"dias_{TECNICA_ALEATORIA}"] = dias
    saida[f"compra_{TECNICA_ALEATORIA}"] = compra

    return saida


def gerar_targets_aleatorios(
    random_folder="./data/train/random/",
    curated_folder="./data/pre_process/curated/",
    output_folder="./data/ensemble/18_targets_aleatorios/"
):
    os.makedirs(output_folder, exist_ok=True)
    arquivos = listar_outputs(random_folder)
    dfs_targets = {}

    for arquivo in arquivos:
        ativo = extrair_ativo(arquivo)
        caminho = os.path.join(random_folder, arquivo)

        print(f"\nMontando sinais aleatorios: {ativo}")
        df_sinais = preparar_sinais_ativo(caminho)

        for target, grupo in df_sinais.groupby("target", sort=True):
            retornos = calcular_retornos_aleatorios(
                grupo[["data", TECNICA_ALEATORIA]].copy(),
                ativo=ativo,
                target=target,
                curated_folder=curated_folder
            )

            if retornos is None:
                continue

            retornos = retornos.rename(columns={
                TECNICA_ALEATORIA: f"{ativo}_{TECNICA_ALEATORIA}",
                f"rend_decisao_{TECNICA_ALEATORIA}": (
                    f"{ativo}_rend_decisao_{TECNICA_ALEATORIA}"
                ),
                f"rend_venda_{TECNICA_ALEATORIA}": (
                    f"{ativo}_rend_venda_{TECNICA_ALEATORIA}"
                ),
            })

            retornos = retornos[[
                "data",
                f"{ativo}_{TECNICA_ALEATORIA}",
                f"{ativo}_rend_decisao_{TECNICA_ALEATORIA}",
                f"{ativo}_rend_venda_{TECNICA_ALEATORIA}",
            ]]

            target_key = formatar_target_arquivo(target)

            if target_key not in dfs_targets:
                dfs_targets[target_key] = retornos
            else:
                dfs_targets[target_key] = dfs_targets[target_key].merge(
                    retornos,
                    on="data",
                    how="outer"
                )

    for target_key, df_target in dfs_targets.items():
        df_target = df_target.sort_values("data").reset_index(drop=True)
        output_path = os.path.join(
            output_folder,
            f"{TECNICA_ALEATORIA}_target_{target_key}.csv"
        )
        df_target.to_csv(output_path, index=False)
        print(f"Target aleatorio salvo: {output_path}")

    print("\nTargets aleatorios concluidos.")


def identificar_ativos_target(df, tecnica):
    sufixo = f"_{tecnica}"
    ignorados = (
        f"_rend_decisao_{tecnica}",
        f"_rend_venda_{tecnica}",
    )

    return sorted([
        coluna[:-len(sufixo)]
        for coluna in df.columns
        if coluna.endswith(sufixo) and not coluna.endswith(ignorados)
    ])


def processar_estrategia_aleatoria_rapida(
    input_path="./data/ensemble/18_targets_aleatorios/",
    output_path="./data/ensemble/18_1n_aleatorio/",
    tecnica=TECNICA_ALEATORIA,
    capital_inicial=100.0
):
    os.makedirs(output_path, exist_ok=True)

    arquivos = sorted([
        arquivo
        for arquivo in os.listdir(input_path)
        if arquivo.endswith(".csv")
    ])

    for arquivo in arquivos:
        caminho = os.path.join(input_path, arquivo)
        print(f"\nCalculando 1/n aleatorio rapido: {arquivo}")

        df = pd.read_csv(caminho)
        df = df.fillna(0)
        ativos = identificar_ativos_target(df, tecnica)

        if not ativos:
            print(f"  Pulando {arquivo}: nenhum ativo encontrado.")
            continue

        target = extrair_target_nome(arquivo)
        threshold = target - 1
        linhas = len(df)
        colunas_sinal = [f"{ativo}_{tecnica}" for ativo in ativos]
        colunas_rend = [f"{ativo}_rend_decisao_{tecnica}" for ativo in ativos]
        colunas_rend_venda = [f"{ativo}_rend_venda_{tecnica}" for ativo in ativos]

        sinais = (
            df[colunas_sinal]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_numpy(dtype=float)
        )
        rend_decisao = (
            df[colunas_rend]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_numpy(dtype=float)
        )
        rend_venda = (
            df[colunas_rend_venda]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_numpy(dtype=float)
        )

        n_ativos = len(ativos)
        inicio = np.zeros((linhas, n_ativos), dtype=float)
        fim = np.zeros((linhas, n_ativos), dtype=float)
        retido = np.zeros((linhas, n_ativos), dtype=float)
        disponivel = np.zeros((linhas, n_ativos), dtype=float)
        bin_aux = np.zeros((linhas, n_ativos), dtype=float)

        total_dividir = np.zeros(linhas, dtype=float)
        total_n = np.zeros(linhas, dtype=float)
        n_sinais = np.zeros(linhas, dtype=float)
        total_verdadeiro = np.zeros(linhas, dtype=float)

        total_dividir[0] = capital_inicial
        disponivel[0, :] = capital_inicial / n_ativos
        bin_aux[0, :] = sinais[0, :]
        n_sinais[0] = max(1.0, bin_aux[0, :].sum())
        total_n[0] = total_dividir[0] / n_sinais[0]
        total_verdadeiro[0] = disponivel[0, :].sum()

        for i in range(1, linhas):
            prev_inicio = inicio[i - 1, :]
            prev_fim = fim[i - 1, :]
            prev_sinais = sinais[i - 1, :]

            travados_prev = (np.abs(prev_inicio - prev_fim) < 1e-12) & (
                np.abs(prev_inicio) > 1e-12
            )

            for j in range(n_ativos):
                inicio_ant = prev_inicio[j]
                fim_ant = prev_fim[j]
                outros_mask = np.ones(n_ativos, dtype=bool)
                outros_mask[j] = False
                cond_outros = bool(travados_prev[outros_mask].all())

                if abs(inicio_ant - fim_ant) < 1e-12 and abs(inicio_ant) > 1e-12:
                    inicio_atual = inicio_ant
                elif cond_outros and prev_sinais[j] == 1:
                    inicio_atual = total_dividir[i - 1]
                elif prev_sinais[j] == 1:
                    inicio_atual = total_n[i - 1]
                else:
                    inicio_atual = 0.0

                inicio[i, j] = inicio_atual

                if rend_decisao[i, j] >= threshold:
                    fim_atual = inicio_atual * target
                else:
                    fim_atual = inicio_atual

                if i >= 3:
                    fim_1 = fim[i - 1, j]
                    fim_2 = fim[i - 2, j]
                    fim_3 = fim[i - 3, j]

                    if (
                        abs(fim_1 - fim_2) < 1e-9
                        and abs(fim_1 - fim_3) < 1e-9
                    ):
                        fim_atual = inicio_atual * (1 + rend_venda[i, j])

                fim[i, j] = fim_atual
                bin_aux[i, j] = (
                    0.0
                    if abs(inicio_atual - fim_atual) < 1e-12
                    and abs(inicio_atual) > 1e-12
                    else sinais[i, j]
                )

            soma_bin = bin_aux[i, :].sum()
            n_sinais[i] = 1.0 if soma_bin == 0 else soma_bin
            retido[i, :] = np.where(
                np.abs(inicio[i, :] - fim[i, :]) < 1e-12,
                inicio[i, :],
                0.0
            )

            for j in range(n_ativos):
                outros_mask = np.ones(n_ativos, dtype=bool)
                outros_mask[j] = False

                inicio_ant = prev_inicio[j]
                fim_ant = prev_fim[j]
                bin_ativo = prev_sinais[j]
                bin_outros_zero = bool((prev_sinais[outros_mask] == 0).all())
                outros_travados = bool(travados_prev[outros_mask].all())

                cond1 = (
                    abs(inicio_ant - fim_ant) < 1e-12
                    and abs(inicio_ant) > 1e-12
                    and bin_ativo == 1
                    and bin_outros_zero
                )
                cond2 = (
                    abs(inicio_ant - fim_ant) >= 1e-12
                    and bin_ativo == 0
                    and outros_travados
                )

                if cond1 or cond2:
                    disponivel[i, j] = total_n[i - 1]

            total_fim = fim[i, :].sum()
            total_retido = retido[i, :].sum()
            total_disp = disponivel[i, :].sum()
            ontem_todos_zero = bool((bin_aux[i - 1, :] == 0).all())

            if ontem_todos_zero:
                total_dividir[i] = total_fim - total_retido + total_dividir[i - 1]
                total_verdadeiro[i] = total_fim + total_dividir[i - 1]
            else:
                total_dividir[i] = total_fim - total_retido + total_disp
                total_verdadeiro[i] = total_fim + total_disp

            if (
                abs(total_fim) < 1e-12
                and abs(total_retido) < 1e-12
                and abs(total_disp) < 1e-12
            ):
                total_verdadeiro[i] = total_verdadeiro[i - 1]
                total_dividir[i] = total_verdadeiro[i - 1]

            total_n[i] = total_dividir[i] / n_sinais[i]

        colunas_aux = {
            "total_dividir": total_dividir,
            "total_n": total_n,
            "N": n_sinais,
            "total_verdadeiro": total_verdadeiro,
        }

        for idx, ativo in enumerate(ativos):
            colunas_aux[f"inicio_{ativo}_{tecnica}"] = inicio[:, idx]
            colunas_aux[f"fim_{ativo}_{tecnica}"] = fim[:, idx]
            colunas_aux[f"retido_{ativo}"] = retido[:, idx]
            colunas_aux[f"disponivel_{ativo}"] = disponivel[:, idx]
            colunas_aux[f"bin_aux_{ativo}"] = bin_aux[:, idx]

        df_saida = pd.concat([df, pd.DataFrame(colunas_aux)], axis=1)
        output_file = os.path.join(output_path, arquivo)
        df_saida.to_csv(output_file, index=False)
        print(f"1/n aleatorio salvo: {output_file}")

    print("\nEstrategia 1/n aleatoria concluida.")


def run_aleatorio(
    outputs_folder="./data/train/outputs/",
    random_folder="./data/train/random/",
    curated_folder="./data/pre_process/curated/",
    targets_folder="./data/ensemble/18_targets_aleatorios/",
    output_1n_folder="./data/ensemble/18_1n_aleatorio/",
    seed=SEED_PADRAO,
    capital_inicial=100
):
    gerar_outputs_aleatorios(
        input_folder=outputs_folder,
        output_folder=random_folder,
        seed=seed
    )
    gerar_targets_aleatorios(
        random_folder=random_folder,
        curated_folder=curated_folder,
        output_folder=targets_folder
    )
    processar_estrategia_aleatoria_rapida(
        input_path=targets_folder,
        output_path=output_1n_folder,
        tecnica=TECNICA_ALEATORIA,
        capital_inicial=capital_inicial
    )

    print("\nPipeline aleatorio 1/n concluido.")


if __name__ == "__main__":
    run_aleatorio()
