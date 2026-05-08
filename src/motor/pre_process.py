from src.pre_process.ibov import processar_ibov
from src.pre_process.inputs import main as processar_inputs

def run_pre_process():
    print("\n🚀 Iniciando pipeline de pré-processamento\n")

    print("1️⃣ Lendo CSVs brutos da Refinitiv em ./data/pre_process/raw/refinitiv")

    print("\n2️⃣ Processando IBOV...")
    processar_ibov()

    print("\n3️⃣ Gerando dados curated...")
    processar_inputs()

    print("\n✅ Pipeline completo finalizado com sucesso!")

if __name__ == "__main__":
    run_pre_process()
