from pathlib import Path
import re

import numpy as np
import pandas as pd


FAMILIAS_TECNICAS = ["RNA", "SVC", "RandomForest"]


def chave_ordenacao_algoritmo(nome):
    match = re.fullmatch(r"(.+)_(\d+)", nome)

    if not match:
        return nome, 0

    return match.group(1), int(match.group(2))


def extrair_familia(nome_arquivo):
    for familia in FAMILIAS_TECNICAS:
        if (
            nome_arquivo.startswith(f"{familia}_target_")
            or re.match(rf"^{re.escape(familia)}_\d+_target_", nome_arquivo)
        ):
            return familia

    raise ValueError(f"Familia de tecnica nao identificada em: {nome_arquivo}")


def extrair_target(nome_arquivo):
    match = re.search(r"target_(\d+[_\.]\d+)", nome_arquivo)

    if not match:
        raise ValueError(f"Target nao encontrado em: {nome_arquivo}")

    sufixo = match.group(1).replace(".", "_")
    valor = float(match.group(1).replace("_", "."))

    return valor, sufixo


def coluna_binaria_algoritmo(coluna, algoritmo):
    return (
        coluna.endswith(f"_{algoritmo}")
        and f"_rend_decisao_{algoritmo}" not in coluna
        and f"_rend_venda_{algoritmo}" not in coluna
    )


def identificar_algoritmos(df, familia):
    algoritmos = set()

    for coluna in df.columns:
        if "_rend_decisao_" in coluna or "_rend_venda_" in coluna:
            continue

        match = re.fullmatch(rf".+_({re.escape(familia)}_\d+)", coluna)
        if not match:
            continue

        algoritmo = match.group(1)
        ativo = coluna[: -(len(algoritmo) + 1)]

        if (
            f"{ativo}_rend_decisao_{algoritmo}" in df.columns
            and f"{ativo}_rend_venda_{algoritmo}" in df.columns
        ):
            algoritmos.add(algoritmo)

    algoritmos = sorted(algoritmos, key=chave_ordenacao_algoritmo)

    if not algoritmos:
        raise ValueError(
            f"Nenhuma janela encontrada para {familia}. "
            f"Colunas disponiveis: {df.columns.tolist()}"
        )

    return algoritmos


def identificar_ativos(df, algoritmo):
    ativos = []

    for coluna in df.columns:
        if not coluna_binaria_algoritmo(coluna, algoritmo):
            continue

        ativo = coluna[: -(len(algoritmo) + 1)]

        if (
            f"{ativo}_rend_decisao_{algoritmo}" in df.columns
            and f"{ativo}_rend_venda_{algoritmo}" in df.columns
        ):
            ativos.append(ativo)

    ativos = sorted(set(ativos))

    if not ativos:
        raise ValueError(f"Nenhum ativo encontrado para {algoritmo}")

    return ativos


def montar_base_algoritmo(df, algoritmo, ativos):
    colunas = ["data"]

    for ativo in ativos:
        colunas.extend(
            [
                f"{ativo}_{algoritmo}",
                f"{ativo}_rend_decisao_{algoritmo}",
                f"{ativo}_rend_venda_{algoritmo}",
            ]
        )

    base = df[colunas].copy()
    base.fillna(0, inplace=True)

    colunas_numericas = [col for col in base.columns if col != "data"]
    base[colunas_numericas] = (
        base[colunas_numericas]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    return base


def calcular_estrategia_1n(df_base, algoritmo, ativos, target, capital_inicial):
    n_linhas = len(df_base)
    n_ativos = len(ativos)

    sinal_cols = [f"{ativo}_{algoritmo}" for ativo in ativos]
    rend_cols = [f"{ativo}_rend_decisao_{algoritmo}" for ativo in ativos]
    venda_cols = [f"{ativo}_rend_venda_{algoritmo}" for ativo in ativos]

    sinais = df_base[sinal_cols].to_numpy(dtype=float)
    rend_decisao = df_base[rend_cols].to_numpy(dtype=float)
    rend_venda = df_base[venda_cols].to_numpy(dtype=float)

    total_dividir = np.zeros(n_linhas, dtype=float)
    total_n = np.zeros(n_linhas, dtype=float)
    n_operacoes = np.zeros(n_linhas, dtype=float)
    total_verdadeiro = np.zeros(n_linhas, dtype=float)

    inicio = np.zeros((n_linhas, n_ativos), dtype=float)
    fim = np.zeros((n_linhas, n_ativos), dtype=float)
    retido = np.zeros((n_linhas, n_ativos), dtype=float)
    disponivel = np.zeros((n_linhas, n_ativos), dtype=float)
    bin_aux = np.zeros((n_linhas, n_ativos), dtype=float)

    if n_linhas == 0:
        return montar_resultado(df_base, algoritmo, ativos, total_dividir, total_n,
                                n_operacoes, inicio, fim, retido, disponivel,
                                bin_aux, total_verdadeiro)

    threshold = target - 1

    total_dividir[0] = capital_inicial
    disponivel[0, :] = capital_inicial / n_ativos
    bin_aux[0, :] = sinais[0, :]
    n_operacoes[0] = max(1, bin_aux[0, :].sum())
    total_n[0] = total_dividir[0] / n_operacoes[0]
    total_verdadeiro[0] = disponivel[0, :].sum()

    for i in range(1, n_linhas):
        inicio_ant = inicio[i - 1, :]
        fim_ant = fim[i - 1, :]
        locked_prev = (inicio_ant == fim_ant) & (inicio_ant != 0)
        locked_count = locked_prev.sum()
        cond_outros = (locked_count - locked_prev.astype(int)) == (n_ativos - 1)

        bin_prev = sinais[i - 1, :]
        inicio_i = np.zeros(n_ativos, dtype=float)

        inicio_i[locked_prev] = inicio_ant[locked_prev]

        livres = ~locked_prev
        usar_total = livres & cond_outros & (bin_prev == 1)
        usar_n = livres & ~usar_total & (bin_prev == 1)

        inicio_i[usar_total] = total_dividir[i - 1]
        inicio_i[usar_n] = total_n[i - 1]

        fim_i = np.where(
            rend_decisao[i, :] >= threshold,
            inicio_i * target,
            inicio_i,
        )

        if i >= 3:
            mesma_saida_3_dias = (
                np.isclose(fim[i - 1, :], fim[i - 2, :], atol=1e-9)
                & np.isclose(fim[i - 1, :], fim[i - 3, :], atol=1e-9)
            )
            fim_i[mesma_saida_3_dias] = (
                inicio_i[mesma_saida_3_dias]
                * (1 + rend_venda[i, mesma_saida_3_dias])
            )

        inicio[i, :] = inicio_i
        fim[i, :] = fim_i

        locked_now = (inicio_i == fim_i) & (inicio_i != 0)
        bin_aux[i, :] = np.where(locked_now, 0, sinais[i, :])

        soma_bin = bin_aux[i, :].sum()
        n_operacoes[i] = 1 if soma_bin == 0 else soma_bin

        retido[i, :] = np.where(inicio_i == fim_i, inicio_i, 0)

        bin_outros_zero = (bin_prev.sum() - bin_prev) == 0
        cond1 = locked_prev & (bin_prev == 1) & bin_outros_zero
        cond2 = (
            (inicio_ant != fim_ant)
            & (bin_prev == 0)
            & cond_outros
        )
        disponivel[i, :] = np.where(cond1 | cond2, total_n[i - 1], 0)

        total_fim = fim[i, :].sum()
        total_retido = retido[i, :].sum()
        total_disp = disponivel[i, :].sum()

        ontem_todos_zero = np.all(bin_aux[i - 1, :] == 0)

        if ontem_todos_zero:
            total_dividir[i] = total_fim - total_retido + total_dividir[i - 1]
            total_verdadeiro[i] = total_fim + total_dividir[i - 1]
        else:
            total_dividir[i] = total_fim - total_retido + total_disp
            total_verdadeiro[i] = total_fim + total_disp

        if total_fim == 0.0 and total_retido == 0.0 and total_disp == 0.0:
            total_verdadeiro[i] = total_verdadeiro[i - 1]
            total_dividir[i] = total_verdadeiro[i - 1]

        total_n[i] = total_dividir[i] / n_operacoes[i]

    return montar_resultado(
        df_base,
        algoritmo,
        ativos,
        total_dividir,
        total_n,
        n_operacoes,
        inicio,
        fim,
        retido,
        disponivel,
        bin_aux,
        total_verdadeiro,
    )


def montar_resultado(
    df_base,
    algoritmo,
    ativos,
    total_dividir,
    total_n,
    n_operacoes,
    inicio,
    fim,
    retido,
    disponivel,
    bin_aux,
    total_verdadeiro,
):
    colunas_auxiliares = {
        "total_dividir": total_dividir,
        "total_n": total_n,
        "N": n_operacoes,
    }

    for idx, ativo in enumerate(ativos):
        colunas_auxiliares[f"inicio_{ativo}_{algoritmo}"] = inicio[:, idx]
        colunas_auxiliares[f"fim_{ativo}_{algoritmo}"] = fim[:, idx]
        colunas_auxiliares[f"retido_{ativo}"] = retido[:, idx]
        colunas_auxiliares[f"disponivel_{ativo}"] = disponivel[:, idx]
        colunas_auxiliares[f"bin_aux_{ativo}"] = bin_aux[:, idx]

    colunas_auxiliares["total_verdadeiro"] = total_verdadeiro

    return pd.concat(
        [df_base.reset_index(drop=True), pd.DataFrame(colunas_auxiliares)],
        axis=1,
    )


def processar_estrategia_tecnicas(
    input_path="./data/tecnicas/targets_por_tecnica_tecnicas_6/",
    output_path="./data/tecnicas/otimi_tecnicas_6/",
    capital_inicial=100,
):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    arquivos = sorted(input_path.glob("*.csv"))

    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {input_path}")

    total_gerados = 0

    for caminho in arquivos:
        familia = extrair_familia(caminho.name)
        target, sufixo_target = extrair_target(caminho.name)

        print(f"\nArquivo: {caminho.name}")

        df = pd.read_csv(caminho)
        df.fillna(0, inplace=True)

        algoritmos = identificar_algoritmos(df, familia)
        print(f"Janelas encontradas: {', '.join(algoritmos)}")

        for algoritmo in algoritmos:
            ativos = identificar_ativos(df, algoritmo)
            base = montar_base_algoritmo(df, algoritmo, ativos)
            resultado = calcular_estrategia_1n(
                base,
                algoritmo,
                ativos,
                target,
                capital_inicial,
            )

            caminho_saida = output_path / f"{algoritmo}_target_{sufixo_target}.csv"
            resultado.to_csv(caminho_saida, index=False)
            total_gerados += 1

            print(
                f"  Salvo: {caminho_saida.name} "
                f"({len(resultado)} linhas, {len(resultado.columns)} colunas, "
                f"{len(ativos)} ativos)"
            )

    print("\nResumo")
    print(f"Arquivos de entrada: {len(arquivos)}")
    print(f"Arquivos gerados: {total_gerados}")


def run_estrategia_1n_tecnicas():
    processar_estrategia_tecnicas()


if __name__ == "__main__":
    run_estrategia_1n_tecnicas()
