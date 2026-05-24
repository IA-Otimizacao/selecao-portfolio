import os
import re
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import carregar_dados, padronizar_colunas, remover_linhas_invalidas

# PASSO A PASSO DO SCRIPT
# 1. Le os CSVs da etapa 11, que ja estao separados por tecnica e target.
# 2. Identifica automaticamente os ativos presentes nesses arquivos.
# 3. Para cada ativo, cria uma coluna *_sinal com o sinal binario da tecnica.
# 4. Cria uma coluna *_bin_aux para impedir nova entrada enquanto a posicao
#    anterior ainda estaria aberta pela regra de target/limite de dias.
# 5. Busca os fechamentos historicos de cada ativo na pasta Refinitiv.
# 6. Junta sinais, bin_aux e precos de fechamento em uma base unica por arquivo.
# 7. Salva a base intermediaria usada pelas otimizacoes Markowitz.

# As tecnicas existentes aparecem no nome do arquivo e no sufixo das colunas.
TECNICAS = ["esmble_jan_tot", "esmble_jan_par", "in_precision", "aleatorio"]


def normalizar_tecnicas(tecnicas=None):
    if tecnicas is None:
        return TECNICAS

    return list(tecnicas)


def extrair_tecnica(nome_arquivo, tecnicas=None):
    tecnicas = normalizar_tecnicas(tecnicas)
    tecnica = next(
        (tec for tec in tecnicas if nome_arquivo.startswith(f"{tec}_target_")),
        None
    )

    if tecnica is None:
        raise ValueError(f"Tecnica nao encontrada no arquivo: {nome_arquivo}")

    return tecnica


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


def identificar_ativos_input(input_folder, price_folder, tecnicas=None):
    # Varre os arquivos de sinais para descobrir quais ativos existem no input
    # e mantem somente os que tambem possuem arquivo de preco.
    tecnicas = normalizar_tecnicas(tecnicas)
    ativos = set()

    arquivos = sorted([
        arquivo
        for arquivo in os.listdir(input_folder)
        if arquivo.endswith(".csv")
    ])

    sufixos_tecnicas = []
    for tecnica in TECNICAS:
        sufixos_tecnicas.extend([
            f"_rend_decisao_{tecnica}",
            f"_rend_venda_{tecnica}",
            f"_{tecnica}",
        ])

    for arquivo in arquivos:
        caminho = os.path.join(input_folder, arquivo)
        colunas = pd.read_csv(caminho, nrows=0).columns

        for coluna in colunas:
            for sufixo in sufixos_tecnicas:
                if coluna.endswith(sufixo):
                    ativo = coluna[:-len(sufixo)]
                    if ativo:
                        ativos.add(ativo)
                    break

    ativos = sorted(ativos)
    ativos_com_preco = [
        ativo
        for ativo in ativos
        if os.path.exists(os.path.join(price_folder, f"{ativo}.csv"))
    ]
    ativos_sem_preco = sorted(set(ativos) - set(ativos_com_preco))

    if ativos_sem_preco:
        print(
            "Ativos ignorados sem arquivo de preco: "
            f"{', '.join(ativos_sem_preco)}"
        )

    return ativos_com_preco


def calcular_bin_aux_ativo(df, ativo, tecnica, target):
    # Reconstroi uma regra simples de posicao aberta:
    # se o ativo entrou ontem e ainda nao bateu target nem passou do limite de
    # dias, ele fica com bin_aux=0 para nao receber nova alocacao.
    coluna_sinal = f"{ativo}_{tecnica}"
    coluna_rend = f"{ativo}_rend_decisao_{tecnica}"
    coluna_rend_venda = f"{ativo}_rend_venda_{tecnica}"

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
    caminho = os.path.join(price_folder, f"{ativo}.csv")

    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo de preco nao encontrado: {caminho}")

    # Reaproveita o leitor do projeto, que ja aceita o formato novo da Refinitiv.
    df = carregar_dados(caminho)
    df = padronizar_colunas(df)
    df = remover_linhas_invalidas(df)

    df = df[["Exchange Date", "Close"]].copy()
    df = df.dropna(subset=["Exchange Date", "Close"])
    df = df.drop_duplicates(subset=["Exchange Date"], keep="last")

    df = df.rename(
        columns={
            "Exchange Date": "data",
            "Close": f"{ativo}_Close"
        }
    )

    df["data"] = pd.to_datetime(df["data"], errors="coerce")

    return df


def carregar_fechamentos(ativos, price_folder):
    fechamentos = None

    for ativo in ativos:
        df_ativo = carregar_close_ativo(ativo, price_folder)

        if fechamentos is None:
            fechamentos = df_ativo
        else:
            fechamentos = fechamentos.merge(df_ativo, on="data", how="outer")

    if fechamentos is None:
        return pd.DataFrame(columns=["data"])

    return fechamentos.sort_values("data").reset_index(drop=True)


def montar_base_sinais(caminho_sinais, ativos, price_folder, tecnicas=None):
    # Monta a base final daquele arquivo de tecnica/target:
    # data + sinais + bin_aux + fechamentos usados no calculo dos pesos.
    nome_arquivo = os.path.basename(caminho_sinais)
    tecnica = extrair_tecnica(nome_arquivo, tecnicas=tecnicas)
    target = extrair_target(nome_arquivo)

    df = pd.read_csv(caminho_sinais)
    df["data"] = pd.to_datetime(df["data"], errors="coerce")

    base = df[["data"]].copy()

    # Converte as colunas binarias do ensemble para colunas simples:
    # PETR4_esmble_jan_tot -> PETR4_sinal.
    for ativo in ativos:
        coluna_sinal = f"{ativo}_{tecnica}"

        if coluna_sinal in df.columns:
            base[f"{ativo}_sinal"] = (
                pd.to_numeric(df[coluna_sinal], errors="coerce")
                .fillna(0)
                .astype(int)
            )
        else:
            base[f"{ativo}_sinal"] = 0

        base[f"{ativo}_bin_aux"] = calcular_bin_aux_ativo(
            df,
            ativo=ativo,
            tecnica=tecnica,
            target=target
        )

    fechamentos = carregar_fechamentos(ativos, price_folder)

    # Junta sinais e fechamentos na mesma data.
    base = base.merge(fechamentos, on="data", how="left")
    base = base.sort_values("data").reset_index(drop=True)

    colunas_saida = (
        ["data"] +
        [f"{ativo}_sinal" for ativo in ativos] +
        [f"{ativo}_bin_aux" for ativo in ativos] +
        [f"{ativo}_Close" for ativo in ativos]
    )

    return base[colunas_saida]


def run_base_mv_sharpe(
    input_folder="./data/ensemble/11_targets_por_tecnica/",
    output_folder="./data/ensemble/13_base_mv_sharpe/",
    price_folder="./data/pre_process/raw/refinitiv/",
    ativos=None,
    tecnicas=None
):
    # Executa a etapa 13 inteira para todos os arquivos da etapa 11.
    os.makedirs(output_folder, exist_ok=True)
    tecnicas = normalizar_tecnicas(tecnicas)

    arquivos = sorted([
        arquivo
        for arquivo in os.listdir(input_folder)
        if arquivo.endswith(".csv")
    ])

    if ativos is None:
        ativos = identificar_ativos_input(
            input_folder,
            price_folder,
            tecnicas=tecnicas
        )

    if not ativos:
        print("Nenhum ativo encontrado para montar as bases MV Sharpe.")
        return

    print(f"Usando {len(ativos)} ativos encontrados na pasta de input.")

    for arquivo in arquivos:
        caminho = os.path.join(input_folder, arquivo)

        print(f"Montando base MV Sharpe: {arquivo}")

        base = montar_base_sinais(
            caminho,
            ativos=ativos,
            price_folder=price_folder,
            tecnicas=tecnicas
        )

        output_path = os.path.join(output_folder, arquivo)
        base.to_csv(output_path, index=False)

        print(f"Salvo: {output_path}")

    print("\nBases intermediarias MV Sharpe concluidas.")


if __name__ == "__main__":
    run_base_mv_sharpe()
