import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ensemble.aleatorio_18 import TECNICA_ALEATORIA
from src.ensemble.base_MV_sharpe_13 import run_base_mv_sharpe
from src.ensemble.otimizacao_MV_min_variancia_15 import (
    run_otimizacao_mv_min_variancia,
)


def base_mv_pronta(base_folder):
    if not os.path.isdir(base_folder):
        return False

    return any(
        arquivo.endswith(".csv")
        for arquivo in os.listdir(base_folder)
    )


def run_otimizacao_mv_min_variancia_aleatorio(
    targets_folder="./data/ensemble/18_targets_aleatorios/",
    base_folder="./data/ensemble/19_base_mv_aleatorio/",
    output_folder="./data/ensemble/20_mv_min_variancia_aleatorio/",
    price_folder="./data/pre_process/raw/refinitiv/",
    janela=60
):
    if base_mv_pronta(base_folder):
        print(f"Reutilizando base MV aleatoria existente: {base_folder}")
    else:
        run_base_mv_sharpe(
            input_folder=targets_folder,
            output_folder=base_folder,
            price_folder=price_folder,
            tecnicas=[TECNICA_ALEATORIA]
        )
    run_otimizacao_mv_min_variancia(
        input_folder=base_folder,
        output_folder=output_folder,
        janela=janela
    )

    print("\nPesos MV minima variancia aleatorio concluidos.")


if __name__ == "__main__":
    run_otimizacao_mv_min_variancia_aleatorio()
