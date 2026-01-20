import pandas as pd

# Caminho do arquivo
file_path = "./data/ibov/ibov_completo.xlsx"

# Carrega o Excel
df = pd.read_excel(file_path)

# Mantém apenas as colunas desejadas
colunas_desejadas = ["Data", "ITUB4", "PETR4", "VALE3"]
df = df[colunas_desejadas]

# Converte coluna Data para datetime
df["Data"] = pd.to_datetime(df["Data"], errors='coerce')

# Filtra intervalo de datas
start_date = "2020-01-01"
end_date = "2023-01-01"
mask = (df["Data"] >= start_date) & (df["Data"] <= end_date)
df_filtrado = df.loc[mask].copy()

# Renomeia coluna Data para Exchange Date
df_filtrado.rename(columns={"Data": "Exchange Date"}, inplace=True)

# Mostra as primeiras linhas
print(df_filtrado.head())

# Opcional: salvar em CSV
# Salvar em Excel
df_filtrado.to_excel("./data/ibov/ibov_filtrado.xlsx", index=False)
df_filtrado.to_csv("./data/ibov/ibov_filtrado.csv", index=False)