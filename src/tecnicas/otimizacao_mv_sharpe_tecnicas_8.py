from pathlib import Path
import warnings

import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import expected_returns
from pypfopt import risk_models


def identificar_ativos(df):
    ativos = []

    for col in df.columns:
        if not col.endswith("_sinal"):
            continue

        ativo = col.replace("_sinal", "")

        if f"{ativo}_Close" in df.columns:
            ativos.append(ativo)

    return ativos


def valor_sinal(valor):
    if pd.isna(valor):
        return 0

    return int(float(valor) == 1.0)


def coluna_sinal_investivel(df, ativo):
    coluna_bin_aux = f"{ativo}_bin_aux"

    if coluna_bin_aux in df.columns:
        return coluna_bin_aux

    return f"{ativo}_sinal"


def pesos_iguais(ativos):
    if not ativos:
        return {}

    peso = 1 / len(ativos)
    return {ativo: peso for ativo in ativos}


def limpar_pesos(pesos, ativos):
    pesos_limpos = {
        ativo: max(0.0, float(pesos.get(ativo, 0.0)))
        for ativo in ativos
    }

    soma = sum(pesos_limpos.values())

    if soma <= 0:
        return pesos_iguais(ativos)

    return {
        ativo: peso / soma
        for ativo, peso in pesos_limpos.items()
    }


def calcular_pesos_markowitz(df, i, ativos, janela=60):
    ativos_selecionados = [
        ativo
        for ativo in ativos
        if valor_sinal(df.loc[i - 1, coluna_sinal_investivel(df, ativo)]) == 1
    ]

    if not ativos_selecionados:
        return {}, "sem_ativos"

    if len(ativos_selecionados) == 1:
        return {ativos_selecionados[0]: 1.0}, "ativo_unico"

    colunas_close = [
        f"{ativo}_Close"
        for ativo in ativos_selecionados
    ]

    dados = df.loc[:i - 1, colunas_close].tail(janela).copy()
    dados.columns = ativos_selecionados
    dados = dados.dropna(axis=1, thresh=janela)
    dados = dados.ffill().dropna()

    ativos_validos = list(dados.columns)

    if len(dados) < janela or len(ativos_validos) < 2:
        ativos_fallback = ativos_validos if ativos_validos else ativos_selecionados
        return pesos_iguais(ativos_fallback), "pesos_iguais_historico_insuficiente"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu = expected_returns.mean_historical_return(dados)
            s = risk_models.sample_cov(dados)
            ef = EfficientFrontier(mu, s, weight_bounds=(0, 1))
            pesos = ef.max_sharpe()

        return limpar_pesos(pesos, ativos_validos), "max_sharpe"

    except Exception as exc:
        data = df.loc[i, "data"]
        print(f"  Aviso: Markowitz falhou em {data.date()}: {exc}")
        return pesos_iguais(ativos_validos), "pesos_iguais_falha_otimizador"


def processar_arquivo(caminho, output_folder, janela=60):
    nome_arquivo = caminho.name

    print(f"\nCalculando pesos MV Sharpe tecnicas: {nome_arquivo}")

    df = pd.read_csv(caminho)
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.sort_values("data").reset_index(drop=True)

    ativos = identificar_ativos(df)
    pesos_por_ativo = {
        ativo: [0.0] * len(df)
        for ativo in ativos
    }
    status = ["sem_pesos"] * len(df)
    ativos_usados = [""] * len(df)

    for i in range(1, len(df)):
        pesos, status_i = calcular_pesos_markowitz(
            df,
            i,
            ativos,
            janela=janela,
        )

        for ativo, peso in pesos.items():
            pesos_por_ativo[ativo][i] = peso

        status[i] = status_i
        ativos_usados[i] = ",".join(pesos.keys())

        if i % 250 == 0 or i == len(df) - 1:
            print(
                f"  andamento {nome_arquivo}: {i + 1}/{len(df)} | "
                f"data {df.loc[i, 'data'].date()} | status {status_i}"
            )

    dados_saida = {
        "data": df["data"],
        "status_mv_sharpe": status,
        "ativos_mv_sharpe": ativos_usados,
    }

    for ativo in ativos:
        dados_saida[f"peso_{ativo}_mv_sharpe"] = pesos_por_ativo[ativo]

    saida = pd.DataFrame(dados_saida)
    output_folder.mkdir(parents=True, exist_ok=True)
    caminho_saida = output_folder / nome_arquivo
    saida.to_csv(caminho_saida, index=False)

    print(f"Salvo: {caminho_saida}")


def run_otimizacao_mv_sharpe_tecnicas(
    input_folder="./data/tecnicas/base_mv_sharpe_tecnicas_7/",
    output_folder="./data/tecnicas/mv_sharpe_tecnicas_8/",
    janela=60,
):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    arquivos = sorted(input_folder.glob("*.csv"))

    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {input_folder}")

    for caminho in arquivos:
        processar_arquivo(
            caminho,
            output_folder=output_folder,
            janela=janela,
        )

    print("\nPesos MV Sharpe por tecnicas concluidos.")


if __name__ == "__main__":
    run_otimizacao_mv_sharpe_tecnicas()
