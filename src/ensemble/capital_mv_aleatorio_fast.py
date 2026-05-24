import os
import re

import numpy as np
import pandas as pd


EPSILON = 1e-12


def extrair_target(nome_arquivo):
    match = re.search(r"target_(\d+[_\.]\d+)", nome_arquivo)

    if not match:
        raise ValueError(f"Target nao encontrado em: {nome_arquivo}")

    return float(match.group(1).replace("_", "."))


def identificar_ativos_pesos(df, estrategia):
    sufixo = f"_{estrategia}"
    prefixo = "peso_"
    ativos = []

    for coluna in df.columns:
        if coluna.startswith(prefixo) and coluna.endswith(sufixo):
            ativos.append(coluna[len(prefixo):-len(sufixo)])

    return ativos


def valor_float(valor, padrao=0.0):
    if pd.isna(valor):
        return padrao

    try:
        return float(valor)
    except (TypeError, ValueError):
        return padrao


def carregar_dados(caminho_pesos, caminho_targets):
    df_pesos = pd.read_csv(caminho_pesos)
    df_targets = pd.read_csv(caminho_targets)

    df_pesos["data"] = pd.to_datetime(df_pesos["data"], errors="coerce")
    df_targets["data"] = pd.to_datetime(df_targets["data"], errors="coerce")

    df_pesos = df_pesos.dropna(subset=["data"])
    df_targets = df_targets.dropna(subset=["data"])

    df = df_pesos.merge(df_targets, on="data", how="left")
    return df.sort_values("data").reset_index(drop=True)


def matriz_numerica(df, colunas):
    if not colunas:
        return np.zeros((len(df), 0), dtype=float)

    return (
        df[colunas]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def normalizar_linha(pesos):
    pesos_limpos = np.maximum(0.0, pesos)
    soma = pesos_limpos.sum()

    if soma <= EPSILON:
        return pesos_limpos

    return pesos_limpos / soma


def processar_arquivo(
    caminho_pesos,
    targets_folder,
    output_folder,
    estrategia,
    capital_inicial=100.0,
    dias_max_posicao=4,
    tecnica="aleatorio"
):
    nome_arquivo = os.path.basename(caminho_pesos)
    caminho_targets = os.path.join(targets_folder, nome_arquivo)

    if not os.path.exists(caminho_targets):
        print(f"Pulando {nome_arquivo}: arquivo de targets nao encontrado.")
        return

    print(f"\nDistribuindo capital {estrategia} aleatorio rapido: {nome_arquivo}")

    target = extrair_target(nome_arquivo)
    threshold = target - 1
    df = carregar_dados(caminho_pesos, caminho_targets)
    ativos_pesos = identificar_ativos_pesos(df, estrategia)

    ativos_com_retorno = [
        ativo
        for ativo in ativos_pesos
        if (
            f"{ativo}_rend_decisao_{tecnica}" in df.columns
            and f"{ativo}_rend_venda_{tecnica}" in df.columns
        )
    ]

    ativos_ignorados = sorted(set(ativos_pesos) - set(ativos_com_retorno))
    if ativos_ignorados:
        print(
            "  Ativos ignorados sem colunas de retorno: "
            f"{', '.join(ativos_ignorados)}"
        )

    linhas = len(df)
    n_ativos = len(ativos_pesos)

    peso_cols = [f"peso_{ativo}_{estrategia}" for ativo in ativos_pesos]
    pesos = matriz_numerica(df, peso_cols)

    idx_retorno = [
        idx
        for idx, ativo in enumerate(ativos_pesos)
        if ativo in ativos_com_retorno
    ]
    retorno_mask = np.zeros(n_ativos, dtype=bool)
    retorno_mask[idx_retorno] = True

    rend_decisao = np.zeros((linhas, n_ativos), dtype=float)
    rend_venda = np.zeros((linhas, n_ativos), dtype=float)

    for idx, ativo in enumerate(ativos_pesos):
        if not retorno_mask[idx]:
            continue

        rend_decisao[:, idx] = matriz_numerica(
            df,
            [f"{ativo}_rend_decisao_{tecnica}"]
        )[:, 0]
        rend_venda[:, idx] = matriz_numerica(
            df,
            [f"{ativo}_rend_venda_{tecnica}"]
        )[:, 0]

    peso_alocacao = np.zeros((linhas, n_ativos), dtype=float)
    inicio = np.zeros((linhas, n_ativos), dtype=float)
    fim = np.zeros((linhas, n_ativos), dtype=float)
    retido = np.zeros((linhas, n_ativos), dtype=float)
    retorno_ativo = np.zeros((linhas, n_ativos), dtype=float)
    dias = np.zeros((linhas, n_ativos), dtype=int)
    status = np.full((linhas, n_ativos), "sem_posicao", dtype=object)

    for idx in range(n_ativos):
        if not retorno_mask[idx]:
            status[:, idx] = "sem_retorno"

    capital_disponivel_inicio = np.zeros(linhas, dtype=float)
    capital_alocado_novo = np.zeros(linhas, dtype=float)
    capital_disponivel_fim = np.zeros(linhas, dtype=float)
    capital_retido = np.zeros(linhas, dtype=float)
    capital_total = np.zeros(linhas, dtype=float)
    retorno_total = np.zeros(linhas, dtype=float)
    soma_pesos_original = np.zeros(linhas, dtype=float)
    soma_pesos_alocacao = np.zeros(linhas, dtype=float)

    pos_capital = np.zeros(n_ativos, dtype=float)
    pos_dias = np.zeros(n_ativos, dtype=int)
    capital_disponivel = float(capital_inicial)
    capital_total_anterior = float(capital_inicial)

    for i in range(linhas):
        capital_inicio = capital_disponivel
        capital_disponivel_inicio[i] = capital_inicio

        pesos_originais = np.maximum(0.0, pesos[i, :])
        soma_pesos_original[i] = pesos_originais.sum()

        ativos_livres = (pos_capital <= EPSILON) & retorno_mask
        pesos_para_alocar = np.zeros(n_ativos, dtype=float)
        pesos_para_alocar[ativos_livres] = normalizar_linha(
            pesos_originais[ativos_livres]
        )
        peso_alocacao[i, :] = pesos_para_alocar
        soma_pesos_alocacao[i] = pesos_para_alocar.sum()

        novos_valores = capital_inicio * pesos_para_alocar
        abriu_posicao = novos_valores > EPSILON
        pos_capital[abriu_posicao] = novos_valores[abriu_posicao]
        pos_dias[abriu_posicao] = 1
        capital_alocado_novo[i] = novos_valores.sum()
        capital_disponivel -= capital_alocado_novo[i]

        for j in range(n_ativos):
            if not retorno_mask[j]:
                continue

            capital_posicao = pos_capital[j]
            if capital_posicao <= EPSILON:
                continue

            dias_posicao = pos_dias[j]
            capital_fim = capital_posicao
            retorno = 0.0
            status_posicao = "mantido"
            manter_posicao = True

            if rend_decisao[i, j] >= threshold:
                retorno = threshold
                capital_fim = capital_posicao * target
                status_posicao = "target"
                manter_posicao = False
            elif dias_posicao >= dias_max_posicao:
                retorno = rend_venda[i, j]
                capital_fim = capital_posicao * (1 + rend_venda[i, j])
                status_posicao = "venda_limite_dias"
                manter_posicao = False

            inicio[i, j] = capital_posicao
            fim[i, j] = capital_fim
            retorno_ativo[i, j] = retorno
            dias[i, j] = dias_posicao
            status[i, j] = status_posicao

            if manter_posicao:
                retido[i, j] = capital_fim
                pos_capital[j] = capital_fim
                pos_dias[j] = dias_posicao + 1
            else:
                capital_disponivel += capital_fim
                pos_capital[j] = 0.0
                pos_dias[j] = 0

        capital_retido[i] = pos_capital.sum()
        capital_total[i] = capital_disponivel + capital_retido[i]

        if capital_total_anterior > EPSILON:
            retorno_total[i] = (capital_total[i] / capital_total_anterior) - 1

        capital_disponivel_fim[i] = capital_disponivel
        capital_total_anterior = capital_total[i]

        if i % 250 == 0 or i == linhas - 1:
            print(
                f"  andamento {nome_arquivo}: {i + 1}/{linhas} | "
                f"capital {capital_total[i]:.2f}"
            )

    saida = {
        "data": df["data"],
        f"capital_disponivel_inicio_{estrategia}": capital_disponivel_inicio,
        f"capital_alocado_novo_{estrategia}": capital_alocado_novo,
        f"capital_disponivel_fim_{estrategia}": capital_disponivel_fim,
        f"capital_retido_{estrategia}": capital_retido,
        f"capital_total_{estrategia}": capital_total,
        f"retorno_total_{estrategia}": retorno_total,
        f"soma_pesos_original_{estrategia}": soma_pesos_original,
        f"soma_pesos_alocacao_{estrategia}": soma_pesos_alocacao,
    }

    status_col = f"status_{estrategia}"
    ativos_col = f"ativos_{estrategia}"

    if status_col in df.columns:
        saida[status_col] = df[status_col]

    if ativos_col in df.columns:
        saida[ativos_col] = df[ativos_col]

    for idx, ativo in enumerate(ativos_pesos):
        saida[f"peso_{ativo}_{estrategia}"] = pesos[:, idx]
        saida[f"peso_alocacao_{ativo}_{estrategia}"] = peso_alocacao[:, idx]
        saida[f"inicio_{ativo}_{estrategia}"] = inicio[:, idx]
        saida[f"fim_{ativo}_{estrategia}"] = fim[:, idx]
        saida[f"retido_{ativo}_{estrategia}"] = retido[:, idx]
        saida[f"retorno_{ativo}_{estrategia}"] = retorno_ativo[:, idx]
        saida[f"dias_{ativo}_{estrategia}"] = dias[:, idx]
        saida[f"status_{ativo}_{estrategia}"] = status[:, idx]

    df_saida = pd.DataFrame(saida)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, nome_arquivo)
    df_saida.to_csv(output_path, index=False)
    print(f"Salvo: {output_path}")


def run_capital_mv_aleatorio(
    weights_folder,
    targets_folder,
    output_folder,
    estrategia,
    capital_inicial=100.0,
    dias_max_posicao=4
):
    os.makedirs(output_folder, exist_ok=True)

    arquivos = sorted([
        arquivo
        for arquivo in os.listdir(weights_folder)
        if arquivo.endswith(".csv")
    ])

    for arquivo in arquivos:
        processar_arquivo(
            os.path.join(weights_folder, arquivo),
            targets_folder=targets_folder,
            output_folder=output_folder,
            estrategia=estrategia,
            capital_inicial=capital_inicial,
            dias_max_posicao=dias_max_posicao
        )

    print(f"\nCapital {estrategia} aleatorio rapido concluido.")
