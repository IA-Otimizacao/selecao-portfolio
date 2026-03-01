import pandas as pd

# Função responsável por processar os dados do IBOV
def processar_ibov():

    # Caminho do arquivo Excel original contendo os dados completos do IBOV
    file_path = "./data/pre_process/ibov/ibov_completo.xlsx"
    df = pd.read_excel(file_path)
    colunas_desejadas = ["Data", "ITUB4", "PETR4", "VALE3"]
    df = df[colunas_desejadas]
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")

    # Criação de uma máscara para filtrar apenas o intervalo de datas
    mask = (df["Data"] >= "2020-01-01") & (df["Data"] <= "2023-01-01")
    df = df.loc[mask].copy()
    df.rename(columns={"Data": "Exchange Date"}, inplace=True)
    df.to_excel("./data/pre_process/ibov/ibov_filtrado.xlsx", index=False)
    df.to_csv("./data/pre_process/ibov/ibov_filtrado.csv", index=False)
    print("✅ IBOV processado e salvo.")