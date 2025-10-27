import pandas as pd
import os

# Pastas de origem e destino
input_folder = "./data/raw/xlsx"
output_folder = "./data/raw/refinitiv"
os.makedirs(output_folder, exist_ok=True)

# Colunas que queremos manter
colunas_desejadas = ["Exchange Date", "Close", "Open", "Low", "High", "Volume"]

# Mapeamento dos meses para abreviação em português
meses_pt = {
    1: "jan.", 2: "fev.", 3: "mar.", 4: "abr.", 5: "mai.", 6: "jun.",
    7: "jul.", 8: "ago.", 9: "set.", 10: "out.", 11: "nov.", 12: "dez."
}

colunas_numericas = ["Close", "Open", "Low", "High"]

# Função para formatar data
def formatar_data(date):
    if pd.isna(date):
        return ""
    return f"{date.day}-{meses_pt[date.month]}-{date.year}"

# Nova função: retorna DataFrame filtrado pelo intervalo de datas
def filtrar_intervalo(df, start_date, end_date):
    """
    df: DataFrame com a coluna 'Exchange Date' em datetime
    start_date, end_date: strings 'dd-mm-yyyy' ou datetime
    """
    # Converte strings para datetime se necessário
    start = pd.to_datetime(start_date, dayfirst=True)
    end = pd.to_datetime(end_date, dayfirst=True)
    
    # Filtra as linhas dentro do intervalo
    mask = (df["Exchange Date"] >= start) & (df["Exchange Date"] <= end)
    return df.loc[mask].copy()

# Loop principal
for file in os.listdir(input_folder):
    if file.endswith((".xlsx", ".xlsm")):
        file_path = os.path.join(input_folder, file)
        
        # Lê o Excel sem cabeçalho
        df_tmp = pd.read_excel(file_path, header=None)
        start_row = df_tmp[df_tmp.iloc[:, 0].astype(str).str.startswith("Exchange Date", na=False)].index[0]
        
        # Reabre o arquivo usando essa linha como cabeçalho
        df = pd.read_excel(file_path, header=start_row)
        df = df[[col for col in colunas_desejadas if col in df.columns]]
        
        # Converte Exchange Date para datetime
        if "Exchange Date" in df.columns:
            df["Exchange Date"] = pd.to_datetime(df["Exchange Date"], errors='coerce')
            
            # Aplica filtro de intervalo de datas
            df = filtrar_intervalo(df, "01-01-2020", "01-01-2025")
            
            # Formata novamente para '23-jan.-2005'
            df["Exchange Date"] = df["Exchange Date"].apply(formatar_data)
        
        # Substitui ponto por vírgula nas colunas numéricas
        for col in colunas_numericas:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('.', ',', regex=False)
        
        # Salva CSV
        csv_name = os.path.splitext(file)[0] + ".csv"
        csv_path = os.path.join(output_folder, csv_name)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        
        print(f"Convertido: {file} -> {csv_name} (intervalo de datas aplicado)")

    print("✅ Conversão concluída!")