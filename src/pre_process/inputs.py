import pandas as pd
import numpy as np
import warnings
import os
from tqdm import tqdm
from src.utils import *

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)      

targets = [1.01, 1.015, 1.02]
ibov = 'ibov_completo_tratado'
janela_temporal = 4

# Pasta onde estão os CSVs dos ativos
input_folder = "./data/pre_process/raw/refinitiv"


def listar_arquivos_ativos():
    arquivos = {}

    for nome_arquivo in sorted(os.listdir(input_folder)):
        if not nome_arquivo.lower().endswith(".csv"):
            continue

        ativo = os.path.splitext(nome_arquivo)[0]
        caminho = os.path.join(input_folder, nome_arquivo)

        arquivos[ativo] = caminho

    return arquivos


def main():

    ibov_status = carregar_ibov(ibov)
    arquivos_ativos = listar_arquivos_ativos()

    if "PETR4" not in arquivos_ativos:
        raise FileNotFoundError("Arquivo da PETR4 não encontrado em raw/refinitiv.")

    base_petr4 = carregar_dados(arquivos_ativos["PETR4"])

    os.makedirs("./data/pre_process/raw/ibov_status", exist_ok=True)
    os.makedirs("./data/pre_process/curated", exist_ok=True)

    print("\n Iniciando processamento e salvamento em curated...\n")

    ativos_pulados = []

    # Loop principal que percorre cada ativo
    for file, file_path in tqdm(arquivos_ativos.items(), desc="Ativos", unit="ativo"):

        if file not in ibov_status.columns:
            print(f"⚠️ Ativo {file} não encontrado no IBOV. Pulando.")
            ativos_pulados.append((file, "não encontrado no IBOV"))
            continue

        try:
            base_dados_original = carregar_dados(file_path)
            base_dados_original = padronizar_colunas(base_dados_original)
            base_dados_original = remover_linhas_invalidas(base_dados_original)
        except (ValueError, FileNotFoundError, pd.errors.EmptyDataError) as exc:
            print(f"⚠️ Ativo {file} ignorado: {exc}")
            ativos_pulados.append((file, str(exc)))
            continue

        if base_dados_original.empty:
            motivo = "sem linhas válidas após limpeza"
            print(f"⚠️ Ativo {file} ignorado: {motivo}")
            ativos_pulados.append((file, motivo))
            continue

        # Alinha os dados do ativo com o calendário da PETR4
        base_dados_original = pd.merge(
            base_petr4[['Exchange Date']],
            base_dados_original,
            on='Exchange Date',
            how='left'
        )

        # Marca linhas adicionadas
        base_dados_original['data_adicional'] = np.where(
            base_dados_original['Open'].isnull(), 1, 0
        )

        # Preenche valores faltantes
        base_dados_original = base_dados_original.ffill()
        base_dados_original = base_dados_original.dropna()

        if base_dados_original.empty:
            motivo = "sem dados após alinhamento com calendário da PETR4"
            print(f"⚠️ Ativo {file} ignorado: {motivo}")
            ativos_pulados.append((file, motivo))
            continue

        # Junta com IBOV
        base_dados_original = pd.merge(
            base_dados_original,
            ibov_status[['Exchange Date', file]],
            on='Exchange Date',
            how='left'
        )

        # Renomeia coluna de status
        base_dados_original.rename(columns={file: 'ibov_status'}, inplace=True)

        # Salva base intermediária
        base_dados_original.to_csv(
            f"./data/pre_process/raw/ibov_status/{file}.csv",
            index=False
        )

        # Loop de targets
        for target in tqdm(targets, desc=f"Targets de {file}", leave=False):

            base_dados = base_dados_original.copy()

            # Filtra período dentro do IBOV
            base_dados = base_dados[base_dados['ibov_status'] == 1].copy()

            if base_dados.empty:
                print(f"⚠️ {file} target {target}: sem dados no período do IBOV. Pulando target.")
                continue

            # Garante tipo numérico
            base_dados = garantir_numerico(
                base_dados,
                ['Open', 'High', 'Close']
            )

            # Variáveis derivadas
            base_dados = calcular_variaveis(base_dados)

            # Target
            base_dados = calcular_target(
                base_dados,
                target,
                janela_temporal
            )

            base_dados = base_dados.iloc[14:]

            # Salva
            curated_path = f"./data/pre_process/curated/{file}_target_{target}.csv"

            base_dados.to_csv(
                curated_path,
                index=False
            )

    if ativos_pulados:
        print("\n⚠️ Ativos pulados:")
        for ativo, motivo in ativos_pulados:
            print(f"  - {ativo}: {motivo}")

    print("\nProcessamento inicial finalizado! Dados salvos em ./data/pre_process/curated.")
