import os
from pathlib import Path
import re
import sys

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import carregar_dados, padronizar_colunas, remover_linhas_invalidas


FAMILIAS_TECNICAS = ["RNA", "SVC", "RandomForest"]


def chave_ordenacao_algoritmo(nome):
    match = re.fullmatch(r"(.+)_(\d+)", nome)

    if not match:
        return nome, 0

    return match.group(1), int(match.group(2))


def extrair_algoritmo(nome_arquivo):
    match = re.match(
        rf"^({'|'.join(FAMILIAS_TECNICAS)})_(\d+)_target_",
        nome_arquivo,
    )

    if not match:
        raise ValueError(f"Tecnica+janela nao encontrada no arquivo: {nome_arquivo}")

    return f"{match.group(1)}_{match.group(2)}"


def extrair_target(nome_arquivo):
    match = re.search(r"target_(\d+[_\.]\d+)", nome_arquivo)

    if match is None:
        raise ValueError(f"Target nao encontrado no arquivo: {nome_arquivo}")

    return float(match.group(1).replace("_", "."))


def valor_float(valor, padrao=0.0):
    if pd.isna(valor):
        return padrao

    try:
        return float(valor)
    except (TypeError, ValueError):
        return padrao


def valor_binario(valor):
    return int(valor_float(valor) == 1.0)


def identificar_algoritmos_input(input_folder):
    algoritmos = set()

    for caminho in sorted(input_folder.glob("*.csv")):
        algoritmos.add(extrair_algoritmo(caminho.name))

    return sorted(algoritmos, key=chave_ordenacao_algoritmo)


def identificar_ativo_por_coluna(coluna, algoritmos):
    for algoritmo in algoritmos:
        sufixos = [
            f"_rend_decisao_{algoritmo}",
            f"_rend_venda_{algoritmo}",
            f"_{algoritmo}",
        ]

        for sufixo in sufixos:
            if coluna.endswith(sufixo):
                ativo = coluna[: -len(sufixo)]
                return ativo or None

    return None


def identificar_ativos_input(input_folder, price_folder):
    algoritmos = identificar_algoritmos_input(input_folder)
    ativos = set()

    arquivos = sorted(input_folder.glob("*.csv"))

    for caminho in arquivos:
        colunas = pd.read_csv(caminho, nrows=0).columns

        for coluna in colunas:
            ativo = identificar_ativo_por_coluna(coluna, algoritmos)

            if ativo:
                ativos.add(ativo)

    ativos = sorted(ativos)
    ativos_com_preco = [
        ativo
        for ativo in ativos
        if (price_folder / f"{ativo}.csv").exists()
    ]
    ativos_sem_preco = sorted(set(ativos) - set(ativos_com_preco))

    if ativos_sem_preco:
        print(
            "Ativos ignorados sem arquivo de preco: "
            f"{', '.join(ativos_sem_preco)}"
        )

    return ativos_com_preco


def calcular_bin_aux_ativo(df, ativo, algoritmo, target):
    coluna_sinal = f"{ativo}_{algoritmo}"
    coluna_rend = f"{ativo}_rend_decisao_{algoritmo}"
    coluna_rend_venda = f"{ativo}_rend_venda_{algoritmo}"

    if coluna_sinal not in df.columns:
        return pd.Series(0, index=df.index)

    if coluna_rend not in df.columns or coluna_rend_venda not in df.columns:
        return (
            pd.to_numeric(df[coluna_sinal], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    threshold = target - 1
    bin_aux = []
    posicao_aberta = False
    dias_posicao = 0

    for i in range(len(df)):
        if i > 0 and not posicao_aberta:
            sinal_anterior = valor_binario(df.loc[i - 1, coluna_sinal])

            if sinal_anterior == 1:
                posicao_aberta = True
                dias_posicao = 1

        if posicao_aberta:
            rend = valor_float(df.loc[i, coluna_rend])

            if rend >= threshold:
                posicao_aberta = False
                dias_posicao = 0
            elif dias_posicao >= 4:
                posicao_aberta = False
                dias_posicao = 0
            else:
                dias_posicao += 1

        if posicao_aberta:
            bin_aux.append(0)
        else:
            bin_aux.append(valor_binario(df.loc[i, coluna_sinal]))

    return pd.Series(bin_aux, index=df.index)


def carregar_close_ativo(ativo, price_folder):
    caminho = price_folder / f"{ativo}.csv"

    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de preco nao encontrado: {caminho}")

    df = carregar_dados(str(caminho))
    df = padronizar_colunas(df)
    df = remover_linhas_invalidas(df)

    df = df[["Exchange Date", "Close"]].copy()
    df = df.dropna(subset=["Exchange Date", "Close"])
    df = df.drop_duplicates(subset=["Exchange Date"], keep="last")

    df = df.rename(
        columns={
            "Exchange Date": "data",
            "Close": f"{ativo}_Close",
        }
    )

    df["data"] = pd.to_datetime(df["data"], errors="coerce")

    return df


def carregar_fechamentos(ativos, price_folder):
    fechamentos = None

    for idx, ativo in enumerate(ativos, start=1):
        if idx % 25 == 0 or idx == len(ativos):
            print(f"  Precos carregados: {idx}/{len(ativos)}")

        df_ativo = carregar_close_ativo(ativo, price_folder)

        if fechamentos is None:
            fechamentos = df_ativo
        else:
            fechamentos = fechamentos.merge(df_ativo, on="data", how="outer")

    if fechamentos is None:
        return pd.DataFrame(columns=["data"])

    return fechamentos.sort_values("data").reset_index(drop=True)


def montar_base_sinais(caminho_sinais, ativos, fechamentos):
    nome_arquivo = caminho_sinais.name
    algoritmo = extrair_algoritmo(nome_arquivo)
    target = extrair_target(nome_arquivo)

    df = pd.read_csv(caminho_sinais).reset_index(drop=True)
    df["data"] = pd.to_datetime(df["data"], errors="coerce")

    sinais = {}
    bin_aux = {}

    for ativo in ativos:
        coluna_sinal = f"{ativo}_{algoritmo}"

        if coluna_sinal in df.columns:
            sinais[f"{ativo}_sinal"] = (
                pd.to_numeric(df[coluna_sinal], errors="coerce")
                .fillna(0)
                .astype(int)
            )
        else:
            sinais[f"{ativo}_sinal"] = pd.Series(0, index=df.index)

        bin_aux[f"{ativo}_bin_aux"] = calcular_bin_aux_ativo(
            df,
            ativo=ativo,
            algoritmo=algoritmo,
            target=target,
        )

    base = pd.concat(
        [
            df[["data"]],
            pd.DataFrame(sinais, index=df.index),
            pd.DataFrame(bin_aux, index=df.index),
        ],
        axis=1,
    )

    base = base.merge(fechamentos, on="data", how="left")
    base = base.sort_values("data").reset_index(drop=True)

    colunas_saida = (
        ["data"]
        + [f"{ativo}_sinal" for ativo in ativos]
        + [f"{ativo}_bin_aux" for ativo in ativos]
        + [f"{ativo}_Close" for ativo in ativos]
    )

    return base[colunas_saida]


def run_base_mv_sharpe_tecnicas(
    input_folder="./data/tecnicas/targets_por_tecnica_tecnicas_6/",
    output_folder="./data/tecnicas/base_mv_sharpe_tecnicas_7/",
    price_folder="./data/pre_process/raw/refinitiv/",
    ativos=None,
):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    price_folder = Path(price_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    arquivos = sorted(input_folder.glob("*.csv"))

    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {input_folder}")

    if ativos is None:
        ativos = identificar_ativos_input(input_folder, price_folder)

    if not ativos:
        print("Nenhum ativo encontrado para montar as bases MV Sharpe.")
        return

    print(f"Usando {len(ativos)} ativos encontrados na pasta de input.")
    print("Carregando precos de fechamento...")
    fechamentos = carregar_fechamentos(ativos, price_folder)

    for caminho in arquivos:
        print(f"\nMontando base MV Sharpe tecnicas: {caminho.name}")

        base = montar_base_sinais(
            caminho,
            ativos=ativos,
            fechamentos=fechamentos,
        )

        caminho_saida = output_folder / caminho.name
        base.to_csv(caminho_saida, index=False)

        print(
            f"Salvo: {caminho_saida} "
            f"({len(base)} linhas, {len(base.columns)} colunas)"
        )

    print("\nBases intermediarias MV Sharpe por tecnicas concluidas.")


if __name__ == "__main__":
    run_base_mv_sharpe_tecnicas()
