import pandas as pd
import os

def xlsx_to_csv():
    input_folder = "./data/pre_process/raw/xlsx"
    output_folder = "./data/pre_process/raw/refinitiv"
    os.makedirs(output_folder, exist_ok=True)

    colunas_desejadas = ["Exchange Date", "Close", "Open", "Low", "High", "Volume"]

    meses_pt = {
        1: "jan.", 2: "fev.", 3: "mar.", 4: "abr.", 5: "mai.", 6: "jun.",
        7: "jul.", 8: "ago.", 9: "set.", 10: "out.", 11: "nov.", 12: "dez."
    }

    colunas_numericas = ["Close", "Open", "Low", "High"]

    def formatar_data(date):
        if pd.isna(date):
            return ""
        return f"{date.day}-{meses_pt[date.month]}-{date.year}"

    def filtrar_intervalo(df, start_date, end_date):
        start = pd.to_datetime(start_date, dayfirst=True)
        end = pd.to_datetime(end_date, dayfirst=True)
        mask = (df["Exchange Date"] >= start) & (df["Exchange Date"] <= end)
        return df.loc[mask].copy()

    for file in os.listdir(input_folder):
        if file.endswith((".xlsx", ".xlsm")):
            file_path = os.path.join(input_folder, file)

            df_tmp = pd.read_excel(file_path, header=None)
            start_row = df_tmp[
                df_tmp.iloc[:, 0].astype(str).str.startswith("Exchange Date", na=False)
            ].index[0]

            df = pd.read_excel(file_path, header=start_row)
            df = df[[c for c in colunas_desejadas if c in df.columns]]

            if "Exchange Date" in df.columns:
                df["Exchange Date"] = pd.to_datetime(df["Exchange Date"], errors="coerce")
                df = filtrar_intervalo(df, "01-01-2020", "01-01-2023")
                df["Exchange Date"] = df["Exchange Date"].apply(formatar_data)

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
