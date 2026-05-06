import pandas as pd
import os
import re

def processar_estrategia(
    input_path="./data/ensemble/11_targets_por_tecnica/",
    output_path="./data/ensemble/11_otimi/",
    tecnicas=['esmble_jan_tot', 'esmble_jan_par', 'in_precision'],
    capital_inicial=100
):

    def extrair_target(nome_arquivo):
        match = re.search(r'target_(\d+[_\.]\d+)', nome_arquivo)
        if match:
            return float(match.group(1).replace('_', '.'))
        else:
            raise ValueError(f"Target não encontrado em: {nome_arquivo}")

    os.makedirs(output_path, exist_ok=True)

    # =========================
    # LOOP NOS ARQUIVOS
    # =========================
    for file in os.listdir(input_path):

        if not file.endswith(".csv"):
            continue

        caminho = os.path.join(input_path, file)

        print(f"\n📂 Processando: {file}")

        df = pd.read_csv(caminho)

        # =========================
        # PREENCHER TODOS OS VALORES VAZIOS COM 0 (antes de tudo)
        # =========================
        df.fillna(0, inplace=True)

        tecnica = next((t for t in tecnicas if t in file), None)
        if tecnica is None:
            continue

        # =========================
        # IDENTIFICA ATIVOS DINAMICAMENTE
        # =========================
        ativos = sorted({
            col.split("_")[0]
            for col in df.columns
            if col.endswith(f"_{tecnica}")
        })

        # =========================
        # CRIAÇÃO DAS COLUNAS AUXILIARES
        # =========================
        target = extrair_target(file)
        threshold = target - 1

        df['total_dividir'] = 0.0
        df['total_n'] = 0.0
        df['N'] = 0.0

        for ativo in ativos:
            df[f'inicio_{ativo}_{tecnica}'] = 0.0
            df[f'fim_{ativo}_{tecnica}'] = 0.0
            df[f'retido_{ativo}'] = 0.0
            df[f'disponivel_{ativo}'] = 0.0
            df[f'bin_aux_{ativo}'] = 0.0

        # =========================
        # INICIALIZAÇÃO
        # =========================
        df.loc[0, 'total_dividir'] = capital_inicial
        valor_ini_disp = capital_inicial / len(ativos)

        for ativo in ativos:
            df.loc[0, f'disponivel_{ativo}'] = valor_ini_disp
            df.loc[0, f'inicio_{ativo}_{tecnica}'] = 0
            df.loc[0, f'fim_{ativo}_{tecnica}'] = 0
            df.loc[0, f'retido_{ativo}'] = 0
            df.loc[0, f'bin_aux_{ativo}'] = df.loc[0, f"{ativo}_{tecnica}"]

        df.loc[0, 'N'] = max(1, sum(df.loc[0, f'bin_aux_{a}'] for a in ativos))
        df.loc[0, 'total_n'] = df.loc[0, 'total_dividir'] / df.loc[0, 'N']
        df.loc[0, 'total_verdadeiro'] = sum(df.loc[0, f'disponivel_{a}'] for a in ativos)

        # =========================
        # LOOP PRINCIPAL
        # =========================
        for i in range(1, len(df)):

            for ativo in ativos:

                col_bin = f"{ativo}_{tecnica}"
                col_rend = f"{ativo}_rend_decisao_{tecnica}"
                col_rend_venda = f"{ativo}_rend_venda_{tecnica}"

                inicio_col = f'inicio_{ativo}_{tecnica}'
                fim_col = f'fim_{ativo}_{tecnica}'

                inicio_ant = df.loc[i-1, inicio_col]
                fim_ant = df.loc[i-1, fim_col]

                outros = [a for a in ativos if a != ativo]

                cond_outros = all(
                    (df.loc[i-1, f'inicio_{o}_{tecnica}'] == df.loc[i-1, f'fim_{o}_{tecnica}']) and
                    (df.loc[i-1, f'inicio_{o}_{tecnica}'] != 0)
                    for o in outros
                )

                if (inicio_ant == fim_ant) and (inicio_ant != 0):
                    inicio = inicio_ant
                elif cond_outros and df.loc[i-1, col_bin] == 1:
                    inicio = df.loc[i-1, 'total_dividir']
                elif df.loc[i-1, col_bin] == 1:
                    inicio = df.loc[i-1, 'total_n']
                else:
                    inicio = 0

                df.loc[i, inicio_col] = inicio

                rend = df.loc[i, col_rend]
                fim = inicio * target if rend >= threshold else inicio

                if i >= 3:
                    fim_1 = df.loc[i-1, fim_col]
                    fim_2 = df.loc[i-2, fim_col]
                    fim_3 = df.loc[i-3, fim_col]

                    if (
                        abs(fim_1 - fim_2) < 1e-9 and
                        abs(fim_1 - fim_3) < 1e-9
                    ):
                        rend_venda = df.loc[i, col_rend_venda]
                        fim = inicio * (1 + rend_venda)

                df.loc[i, fim_col] = fim

                df.loc[i, f'bin_aux_{ativo}'] = 0 if (inicio == fim and inicio != 0) else df.loc[i, col_bin]

            soma_bin = sum(df.loc[i, f'bin_aux_{a}'] for a in ativos)
            df.loc[i, 'N'] = 1 if soma_bin == 0 else soma_bin

            for ativo in ativos:
                inicio_atual = df.loc[i, f'inicio_{ativo}_{tecnica}']
                fim_atual = df.loc[i, f'fim_{ativo}_{tecnica}']
                df.loc[i, f'retido_{ativo}'] = inicio_atual if inicio_atual == fim_atual else 0

            for ativo in ativos:

                col_bin = f"{ativo}_{tecnica}"

                inicio_ant = df.loc[i-1, f'inicio_{ativo}_{tecnica}']
                fim_ant = df.loc[i-1, f'fim_{ativo}_{tecnica}']

                outros = [a for a in ativos if a != ativo]

                bin_ativo = df.loc[i-1, col_bin]
                bin_outros_zero = all(df.loc[i-1, f"{o}_{tecnica}"] == 0 for o in outros)

                outros_travados = all(
                    (df.loc[i-1, f'inicio_{o}_{tecnica}'] == df.loc[i-1, f'fim_{o}_{tecnica}']) and
                    (df.loc[i-1, f'inicio_{o}_{tecnica}'] != 0)
                    for o in outros
                )

                cond1 = (
                    (inicio_ant == fim_ant) and
                    (inicio_ant != 0) and
                    (bin_ativo == 1) and
                    (bin_outros_zero)
                )

                cond2 = (
                    (inicio_ant != fim_ant) and
                    (bin_ativo == 0) and
                    (outros_travados)
                )

                if cond1 or cond2:
                    df.loc[i, f'disponivel_{ativo}'] = df.loc[i-1, 'total_n']
                else:
                    df.loc[i, f'disponivel_{ativo}'] = 0

            total_fim = sum(df.loc[i, f'fim_{a}_{tecnica}'] for a in ativos)
            total_retido = sum(df.loc[i, f'retido_{a}'] for a in ativos)
            total_disp = sum(df.loc[i, f'disponivel_{a}'] for a in ativos)

            ontem_todos_zero = all(df.loc[i-1, f'bin_aux_{a}'] == 0 for a in ativos)

            if ontem_todos_zero:
                df.loc[i, 'total_dividir'] = total_fim - total_retido + df.loc[i-1, 'total_dividir']
                df.loc[i, 'total_verdadeiro'] = total_fim + df.loc[i-1, 'total_dividir']
            else:
                df.loc[i, 'total_dividir'] = total_fim - total_retido + total_disp
                df.loc[i, 'total_verdadeiro'] = total_fim + total_disp

            if total_fim == 0.0 and total_retido == 0.0 and total_disp == 0.0:
                df.loc[i, 'total_verdadeiro'] = df.loc[i-1, 'total_verdadeiro']
                df.loc[i, 'total_dividir'] = df.loc[i-1, 'total_verdadeiro']

            df.loc[i, 'total_n'] = df.loc[i, 'total_dividir'] / df.loc[i, 'N']

        df.to_csv(os.path.join(output_path, file), index=False)

    print("Processamento concluído 🚀")