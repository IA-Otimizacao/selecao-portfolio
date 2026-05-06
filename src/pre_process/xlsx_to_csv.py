import pandas as pd
import os

# Função principal responsável por converter arquivos Excel para CSV
def xlsx_to_csv():
    
    input_folder = "./data/pre_process/raw/xlsx"
    output_folder = "./data/pre_process/raw/refinitiv"
    os.makedirs(output_folder, exist_ok=True)

    # Lista das colunas que queremos manter no dataset final
    colunas_desejadas = ["Exchange Date", "Close", "Open", "Low", "High", "Volume"]

    meses_pt = {
        1: "jan.", 2: "fev.", 3: "mar.", 4: "abr.", 5: "mai.", 6: "jun.",
        7: "jul.", 8: "ago.", 9: "set.", 10: "out.", 11: "nov.", 12: "dez."
    }

    colunas_numericas = ["Close", "Open", "Low", "High"]

    # Função para formatar datas no padrão 
    # Exemplo: 2020-01-03 -> 3-jan.-2020
    def formatar_data(date):

        if pd.isna(date):
            return ""

        # Formata a data utilizando o dicionário de meses
        return f"{date.day}-{meses_pt[date.month]}-{date.year}"


    # Função para filtrar o dataframe por intervalo de datas
    def filtrar_intervalo(df, start_date, end_date):

        # Converte as datas fornecidas para formato datetime
        start = pd.to_datetime(start_date, dayfirst=True)
        end = pd.to_datetime(end_date, dayfirst=True)

        # Cria uma máscara lógica para selecionar apenas as datas no intervalo
        mask = (df["Exchange Date"] >= start) & (df["Exchange Date"] <= end)

        return df.loc[mask].copy()

    # Percorre todos os arquivos da pasta de entrada
    for file in os.listdir(input_folder):

        # Verifica se o arquivo é Excel (.xlsx ou .xlsm)
        if file.endswith((".xlsx", ".xlsm")):

            # Monta o caminho completo do arquivo
            file_path = os.path.join(input_folder, file)

            # Primeira leitura do Excel sem cabeçalho
            df_tmp = pd.read_excel(file_path, header=None)

            # Identifica a linha onde começa a tabela real
            start_row = df_tmp[
                df_tmp.iloc[:, 0].astype(str).str.startswith("Exchange Date", na=False)
            ].index[0]

            # Segunda leitura do Excel usando a linha encontrada como cabeçalho
            df = pd.read_excel(file_path, header=start_row)

            # Mantém apenas as colunas desejadas que existirem no dataframe
            df = df[[c for c in colunas_desejadas if c in df.columns]]

            # Tratamento da coluna de datas
            if "Exchange Date" in df.columns:

                # Converte a coluna para formato datetime
                df["Exchange Date"] = pd.to_datetime(df["Exchange Date"], errors="coerce")
                # inicio = "01-01-2020"
                # fim = "01-04-2020"
                # Filtra os dados
                # df = filtrar_intervalo(df, inicio, fim)

                # Formata as datas para o padrão
                df["Exchange Date"] = df["Exchange Date"].apply(formatar_data)


            # Tratamento das colunas numéricas
            for col in colunas_numericas:

                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(".", ",", regex=False)


            csv_name = os.path.splitext(file)[0] + ".csv"
            df.to_csv(
                os.path.join(output_folder, csv_name),
                index=False,          
                encoding="utf-8-sig"
            )
            print(f"Convertido: {file}")

    print("✅ XLSX → CSV concluído.")
    
