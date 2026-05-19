import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.motor.pre_process import run_pre_process
from src.motor.train import main as run_train
from src.motor.ensemble import run_ensemble
from src.plot import main as run_plot


# ================= CONFIGURACAO DO PIPELINE =================
# Use estas variaveis para rodar o fluxo completo ou testar apenas partes.
RUN_PRE_PROCESS = False
RUN_TRAIN = False
RUN_ENSEMBLE = True
RUN_PLOT = True

# None roda todos os ativos. Exemplo para testar apenas PETR4:
# ATIVOS_TREINO = ["PETR4"]
ATIVOS_TREINO = None


def main():
    print("\n🔥 INICIANDO PIPELINE COMPLETO 🔥\n")

    # -----------------------------
    # 1️⃣ Pré-processamento
    # -----------------------------
    if RUN_PRE_PROCESS:
        print("\n[MAIN] 1/4 - Pre-processamento")
        run_pre_process()
    else:
        print("\n[MAIN] 1/4 - Pre-processamento pulado")

    # -----------------------------
    # 2️⃣ Treinamento
    # -----------------------------
    if RUN_TRAIN:
        print("\n[MAIN] 2/4 - Treinamento")
        run_train(ativos_especificos=ATIVOS_TREINO)
    else:
        print("\n[MAIN] 2/4 - Treinamento pulado")

    # -----------------------------
    # 3️⃣ Ensemble completo
    # -----------------------------
    if RUN_ENSEMBLE:
        print("\n[MAIN] 3/4 - Ensemble completo")
        run_ensemble()
    else:
        print("\n[MAIN] 3/4 - Ensemble pulado")

    # -----------------------------
    # 4️⃣ Visualização
    # -----------------------------
    if RUN_PLOT:
        print("\n[MAIN] 4/4 - Gerando plots")
        run_plot()
    else:
        print("\n[MAIN] 4/4 - Plot pulado")

    print("\n🏁 PIPELINE FINALIZADO COM SUCESSO 🏁")


if __name__ == "__main__":
    main()
