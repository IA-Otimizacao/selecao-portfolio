from src.motor.pre_process import run_pre_process
from src.motor.train import main as run_train
from src.motor.ensemble import run_ensemble


def main():
    print("\n🔥 INICIANDO PIPELINE COMPLETO 🔥\n")

    # -----------------------------
    # 1️⃣ Pré-processamento
    # -----------------------------
    # run_pre_process()

    # -----------------------------
    # 2️⃣ Treinamento
    # -----------------------------
    run_train()

    # -----------------------------
    # 3️⃣ Ensemble + monetário
    # -----------------------------
    # run_ensemble()

    print("\n🏁 PIPELINE FINALIZADO COM SUCESSO 🏁")


if __name__ == "__main__":
    main()
