import pandas as pd

def processar_ibov():

    file_path = "./data/pre_process/ibov/ibov_completo.xlsx"
    df = pd.read_excel(file_path)

    # Garante que a coluna de data existe
    if "Data" not in df.columns:
        raise ValueError("Coluna 'Data' não encontrada no arquivo.")

    # Converte para datetime
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")

    # Renomeia a coluna de data
    df.rename(columns={"Data": "Exchange Date"}, inplace=True)

    # Salva com TODOS os ativos e TODAS as datas
    df.to_excel("./data/pre_process/ibov/ibov_completo_tratado.xlsx", index=False)
    df.to_csv("./data/pre_process/ibov/ibov_completo_tratado.csv", index=False)

    print("✅ IBOV completo processado e salvo.")

