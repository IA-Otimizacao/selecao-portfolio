from pathlib import Path
import re

import numpy as np
import pandas as pd


FAMILIAS_TECNICAS = ["RNA", "SVC", "RandomForest"]
EPSILON = 1e-12


def extrair_target(nome_arquivo):
    match = re.search(r"target_(\d+[_\.]\d+)", nome_arquivo)

    if not match:
        raise ValueError(f"Target nao encontrado em: {nome_arquivo}")

    return float(match.group(1).replace("_", "."))


def extrair_algoritmo(nome_arquivo):
    match = re.match(
        rf"^({'|'.join(FAMILIAS_TECNICAS)})_(\d+)_target_",
        nome_arquivo,
    )

    if not match:
        raise ValueError(f"Tecnica+janela nao encontrada em: {nome_arquivo}")

    return f"{match.group(1)}_{match.group(2)}"


def identificar_ativos_pesos(df, metodo):
    ativos = []

    for col in df.columns:
        match = re.fullmatch(rf"peso_(.+)_{re.escape(metodo)}", col)

        if match:
            ativos.append(match.group(1))

    return ativos


def valor_float(valor, padrao=0.0):
    if pd.isna(valor):
        return padrao

    try:
        return float(valor)
    except (TypeError, ValueError):
        return padrao


def normalizar_pesos(pesos):
    pesos_limpos = {
        ativo: max(0.0, valor_float(peso))
        for ativo, peso in pesos.items()
    }
    soma = sum(pesos_limpos.values())

    if soma <= EPSILON:
        return {}

    return {
        ativo: peso / soma
        for ativo, peso in pesos_limpos.items()
        if peso > EPSILON
    }


def preparar_dados(caminho_pesos, caminho_targets):
    df_pesos = pd.read_csv(caminho_pesos)
    df_targets = pd.read_csv(caminho_targets)

    df_pesos["data"] = pd.to_datetime(df_pesos["data"], errors="coerce")
    df_targets["data"] = pd.to_datetime(df_targets["data"], errors="coerce")

    df_pesos = df_pesos.dropna(subset=["data"])
    df_targets = df_targets.dropna(subset=["data"])

    df = df_pesos.merge(df_targets, on="data", how="left")
    df = df.sort_values("data").reset_index(drop=True)

    return df


def arrays_numericos(df, colunas):
    return {
        col: pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=float)
        for col in colunas
        if col in df.columns
    }


def processar_arquivo(
    caminho_pesos,
    targets_folder,
    output_folder,
    metodo,
    capital_inicial=100.0,
    dias_max_posicao=4,
):
    nome_arquivo = caminho_pesos.name
    caminho_targets = targets_folder / nome_arquivo

    if not caminho_targets.exists():
        print(f"Pulando {nome_arquivo}: arquivo de targets nao encontrado.")
        return False

    print(f"\nDistribuindo capital {metodo}: {nome_arquivo}")

    target = extrair_target(nome_arquivo)
    threshold = target - 1
    algoritmo = extrair_algoritmo(nome_arquivo)

    df = preparar_dados(caminho_pesos, caminho_targets)
    n_linhas = len(df)
    ativos_pesos = identificar_ativos_pesos(df, metodo)

    ativos_com_retorno = [
        ativo
        for ativo in ativos_pesos
        if (
            f"{ativo}_rend_decisao_{algoritmo}" in df.columns
            and f"{ativo}_rend_venda_{algoritmo}" in df.columns
        )
    ]

    ativos_ignorados = sorted(set(ativos_pesos) - set(ativos_com_retorno))
    if ativos_ignorados:
        print(
            "  Ativos ignorados sem colunas de retorno: "
            f"{', '.join(ativos_ignorados)}"
        )

    status_col = f"status_{metodo}"
    ativos_col = f"ativos_{metodo}"

    peso_cols = {
        ativo: f"peso_{ativo}_{metodo}"
        for ativo in ativos_pesos
    }
    peso_arrays = arrays_numericos(df, peso_cols.values())

    rend_decisao = {
        ativo: pd.to_numeric(
            df[f"{ativo}_rend_decisao_{algoritmo}"],
            errors="coerce",
        ).fillna(0).to_numpy(dtype=float)
        for ativo in ativos_com_retorno
    }
    rend_venda = {
        ativo: pd.to_numeric(
            df[f"{ativo}_rend_venda_{algoritmo}"],
            errors="coerce",
        ).fillna(0).to_numpy(dtype=float)
        for ativo in ativos_com_retorno
    }

    saida = {}
    saida["data"] = df["data"]

    if status_col in df.columns:
        saida[status_col] = df[status_col].fillna("")

    if ativos_col in df.columns:
        saida[ativos_col] = df[ativos_col].fillna("")

    gerais = [
        f"capital_disponivel_inicio_{metodo}",
        f"capital_alocado_novo_{metodo}",
        f"capital_disponivel_fim_{metodo}",
        f"capital_retido_{metodo}",
        f"capital_total_{metodo}",
        f"retorno_total_{metodo}",
        f"soma_pesos_original_{metodo}",
        f"soma_pesos_alocacao_{metodo}",
    ]
    geral_arrays = {
        col: np.zeros(n_linhas, dtype=float)
        for col in gerais
    }

    por_ativo = {}
    for ativo in ativos_pesos:
        por_ativo[f"peso_{ativo}_{metodo}"] = np.zeros(n_linhas, dtype=float)
        por_ativo[f"peso_alocacao_{ativo}_{metodo}"] = np.zeros(n_linhas, dtype=float)
        por_ativo[f"inicio_{ativo}_{metodo}"] = np.zeros(n_linhas, dtype=float)
        por_ativo[f"fim_{ativo}_{metodo}"] = np.zeros(n_linhas, dtype=float)
        por_ativo[f"retido_{ativo}_{metodo}"] = np.zeros(n_linhas, dtype=float)
        por_ativo[f"retorno_{ativo}_{metodo}"] = np.zeros(n_linhas, dtype=float)
        por_ativo[f"dias_{ativo}_{metodo}"] = np.zeros(n_linhas, dtype=int)

        status_padrao = "sem_posicao" if ativo in ativos_com_retorno else "sem_retorno"
        por_ativo[f"status_{ativo}_{metodo}"] = np.full(
            n_linhas,
            status_padrao,
            dtype=object,
        )

    posicoes = {
        ativo: {
            "capital": 0.0,
            "dias": 0,
        }
        for ativo in ativos_com_retorno
    }

    capital_disponivel = float(capital_inicial)
    capital_total_anterior = float(capital_inicial)

    for i in range(n_linhas):
        capital_inicio = capital_disponivel

        pesos_originais = {
            ativo: valor_float(peso_arrays.get(peso_cols[ativo], np.zeros(n_linhas))[i])
            for ativo in ativos_pesos
        }
        soma_pesos_original = sum(
            max(0.0, peso)
            for peso in pesos_originais.values()
        )

        ativos_livres = [
            ativo
            for ativo in ativos_com_retorno
            if posicoes[ativo]["capital"] <= EPSILON
        ]

        pesos_para_alocar = normalizar_pesos({
            ativo: pesos_originais[ativo]
            for ativo in ativos_livres
        })

        capital_alocado_novo = 0.0

        for ativo, peso_alocacao in pesos_para_alocar.items():
            valor_alocado = capital_inicio * peso_alocacao

            posicoes[ativo]["capital"] = valor_alocado
            posicoes[ativo]["dias"] = 1
            capital_alocado_novo += valor_alocado

            por_ativo[f"peso_alocacao_{ativo}_{metodo}"][i] = peso_alocacao

        capital_disponivel -= capital_alocado_novo

        for ativo in ativos_pesos:
            peso_original = max(0.0, pesos_originais[ativo])
            por_ativo[f"peso_{ativo}_{metodo}"][i] = peso_original

            if ativo not in ativos_com_retorno:
                continue

            capital_posicao = posicoes[ativo]["capital"]

            if capital_posicao <= EPSILON:
                continue

            dias_posicao = posicoes[ativo]["dias"]
            retorno_decisao = rend_decisao[ativo][i]
            retorno_venda = rend_venda[ativo][i]

            capital_fim = capital_posicao
            retorno_ativo = 0.0
            status_posicao = "mantido"
            manter_posicao = True

            if retorno_decisao >= threshold:
                retorno_ativo = threshold
                capital_fim = capital_posicao * target
                status_posicao = "target"
                manter_posicao = False
            elif dias_posicao >= dias_max_posicao:
                retorno_ativo = retorno_venda
                capital_fim = capital_posicao * (1 + retorno_venda)
                status_posicao = "venda_limite_dias"
                manter_posicao = False

            por_ativo[f"inicio_{ativo}_{metodo}"][i] = capital_posicao
            por_ativo[f"fim_{ativo}_{metodo}"][i] = capital_fim
            por_ativo[f"retorno_{ativo}_{metodo}"][i] = retorno_ativo
            por_ativo[f"dias_{ativo}_{metodo}"][i] = dias_posicao
            por_ativo[f"status_{ativo}_{metodo}"][i] = status_posicao

            if manter_posicao:
                por_ativo[f"retido_{ativo}_{metodo}"][i] = capital_fim
                posicoes[ativo]["capital"] = capital_fim
                posicoes[ativo]["dias"] = dias_posicao + 1
            else:
                capital_disponivel += capital_fim
                posicoes[ativo]["capital"] = 0.0
                posicoes[ativo]["dias"] = 0

        capital_retido = sum(
            posicao["capital"]
            for posicao in posicoes.values()
        )
        capital_total = capital_disponivel + capital_retido

        if capital_total_anterior > EPSILON:
            retorno_total = (capital_total / capital_total_anterior) - 1
        else:
            retorno_total = 0.0

        geral_arrays[f"capital_disponivel_inicio_{metodo}"][i] = capital_inicio
        geral_arrays[f"capital_alocado_novo_{metodo}"][i] = capital_alocado_novo
        geral_arrays[f"capital_disponivel_fim_{metodo}"][i] = capital_disponivel
        geral_arrays[f"capital_retido_{metodo}"][i] = capital_retido
        geral_arrays[f"capital_total_{metodo}"][i] = capital_total
        geral_arrays[f"retorno_total_{metodo}"][i] = retorno_total
        geral_arrays[f"soma_pesos_original_{metodo}"][i] = soma_pesos_original
        geral_arrays[f"soma_pesos_alocacao_{metodo}"][i] = sum(
            pesos_para_alocar.values()
        )

        capital_total_anterior = capital_total

        if i % 1000 == 0 or i == n_linhas - 1:
            print(
                f"  andamento {nome_arquivo}: {i + 1}/{n_linhas} | "
                f"capital {capital_total:.2f}"
            )

    for col in gerais:
        saida[col] = geral_arrays[col]

    for ativo in ativos_pesos:
        for col in [
            f"peso_{ativo}_{metodo}",
            f"peso_alocacao_{ativo}_{metodo}",
            f"inicio_{ativo}_{metodo}",
            f"fim_{ativo}_{metodo}",
            f"retido_{ativo}_{metodo}",
            f"retorno_{ativo}_{metodo}",
            f"dias_{ativo}_{metodo}",
            f"status_{ativo}_{metodo}",
        ]:
            saida[col] = por_ativo[col]

    output_folder.mkdir(parents=True, exist_ok=True)
    caminho_saida = output_folder / nome_arquivo
    pd.DataFrame(saida).to_csv(caminho_saida, index=False)

    print(f"Salvo: {caminho_saida}")
    return True


def run_capital_mv_tecnicas(
    weights_folder,
    targets_folder,
    output_folder,
    metodo,
    capital_inicial=100.0,
    dias_max_posicao=4,
):
    weights_folder = Path(weights_folder)
    targets_folder = Path(targets_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    arquivos = sorted(weights_folder.glob("*.csv"))

    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {weights_folder}")

    total_gerados = 0

    for caminho in arquivos:
        gerou = processar_arquivo(
            caminho,
            targets_folder=targets_folder,
            output_folder=output_folder,
            metodo=metodo,
            capital_inicial=capital_inicial,
            dias_max_posicao=dias_max_posicao,
        )

        if gerou:
            total_gerados += 1

    print("\nResumo")
    print(f"Arquivos de pesos: {len(arquivos)}")
    print(f"Arquivos gerados: {total_gerados}")
