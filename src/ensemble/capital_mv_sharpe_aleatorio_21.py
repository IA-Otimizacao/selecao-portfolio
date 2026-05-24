import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ensemble.capital_mv_aleatorio_fast import run_capital_mv_aleatorio


def run_capital_mv_sharpe_aleatorio(
    weights_folder="./data/ensemble/19_mv_sharpe_aleatorio/",
    targets_folder="./data/ensemble/18_targets_aleatorios/",
    output_folder="./data/ensemble/21_capital_mv_sharpe_aleatorio/",
    capital_inicial=100.0,
    dias_max_posicao=4
):
    run_capital_mv_aleatorio(
        weights_folder=weights_folder,
        targets_folder=targets_folder,
        output_folder=output_folder,
        estrategia="mv_sharpe",
        capital_inicial=capital_inicial,
        dias_max_posicao=dias_max_posicao
    )

    print("\nCapital MV Sharpe aleatorio concluido.")


if __name__ == "__main__":
    run_capital_mv_sharpe_aleatorio()
