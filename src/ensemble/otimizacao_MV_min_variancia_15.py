# from pypfopt import EfficientFrontier
# from pypfopt import risk_models
# from pypfopt import expected_returns
# import yfinance as yf

# # preços
# tickers = ["VALE3.SA", "PETR4.SA", "ITUB4.SA"]

# dados = yf.download(tickers, start="2023-01-01")["Close"]

# # retornos esperados
# mu = expected_returns.mean_historical_return(dados)

# # matriz de covariância
# S = risk_models.sample_cov(dados)

# # otimização de Markowitz
# ef = EfficientFrontier(mu, S)

# pesos = ef.max_sharpe()

# print(pesos)

# ef.portfolio_performance(verbose=True)


import os
import warnings

import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models

# PASSO A PASSO DO SCRIPT
# 1. Le cada base intermediaria gerada pela etapa 13.
# 2. Identifica os ativos que possuem *_sinal e *_Close.
# 3. Em cada data, olha o sinal investivel do dia anterior (*_bin_aux ou *_sinal).
# 4. Seleciona apenas os ativos com sinal igual a 1.
# 5. Usa os ultimos 60 dias de fechamento desses ativos para estimar a matriz
#    de covariancia.
# 6. Calcula os pesos Markowitz de minima variancia com peso entre 0 e 1.
# 7. Quando nao ha historico suficiente ou o otimizador falha, usa pesos iguais.
# 8. Salva um CSV com os pesos diarios de cada ativo para a etapa de capital.


def identificar_ativos(df):
    # Ativo elegivel precisa ter coluna de sinal e coluna de fechamento.
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


def calcular_pesos_min_variancia(df, i, ativos, janela=60):
    # Calcula os pesos do dia i usando somente informacao disponivel ate i-1.
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
        # Minima variancia usa apenas a covariancia; nao estima retorno esperado.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = risk_models.sample_cov(dados)

        ef = EfficientFrontier(None, s, weight_bounds=(0, 1))
        pesos = ef.min_volatility()

        return limpar_pesos(pesos, ativos_validos), "min_variancia"

    except Exception as exc:
        data = df.loc[i, "data"]
        print(f"  Aviso: minima variancia falhou em {data.date()}: {exc}")
        return pesos_iguais(ativos_validos), "pesos_iguais_falha_otimizador"


def processar_arquivo(caminho, output_folder, janela=60):
    # Processa um arquivo tecnica/target e gera pesos diarios de minima variancia.
    nome_arquivo = os.path.basename(caminho)

    print(f"\nCalculando pesos minima variancia: {nome_arquivo}")

    df = pd.read_csv(caminho)
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.sort_values("data").reset_index(drop=True)

    ativos = identificar_ativos(df)

    saida = df[["data"]].copy()
    saida["status_min_variancia"] = "sem_pesos"
    saida["ativos_min_variancia"] = ""

    for ativo in ativos:
        saida[f"peso_{ativo}_min_variancia"] = 0.0

    for i in range(1, len(df)):
        pesos, status = calcular_pesos_min_variancia(
            df,
            i,
            ativos,
            janela=janela
        )

        for ativo, peso in pesos.items():
            saida.loc[i, f"peso_{ativo}_min_variancia"] = peso

        saida.loc[i, "status_min_variancia"] = status
        saida.loc[i, "ativos_min_variancia"] = ",".join(pesos.keys())

        if i % 250 == 0 or i == len(df) - 1:
            print(
                f"  andamento {nome_arquivo}: {i + 1}/{len(df)} | "
                f"data {df.loc[i, 'data'].date()} | status {status}"
            )

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, nome_arquivo)
    saida.to_csv(output_path, index=False)

    print(f"Salvo: {output_path}")


def run_otimizacao_mv_min_variancia(
    input_folder="./data/ensemble/13_base_mv_sharpe/",
    output_folder="./data/ensemble/15_mv_min_variancia/",
    janela=60
):
    # Executa a etapa 15 para todos os arquivos da base MV.
    os.makedirs(output_folder, exist_ok=True)

    arquivos = sorted([
        arquivo
        for arquivo in os.listdir(input_folder)
        if arquivo.endswith(".csv")
    ])

    for arquivo in arquivos:
        processar_arquivo(
            os.path.join(input_folder, arquivo),
            output_folder=output_folder,
            janela=janela
        )

    print("\nPesos Markowitz minima variancia concluidos.")


if __name__ == "__main__":
    run_otimizacao_mv_min_variancia()
