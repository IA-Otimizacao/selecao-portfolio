import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tecnicas.capital_mv_tecnicas import run_capital_mv_tecnicas


def run_capital_mv_sharpe_tecnicas(
    weights_folder="./data/tecnicas/mv_sharpe_tecnicas_8/",
    targets_folder="./data/tecnicas/targets_por_tecnica_tecnicas_6/",
    output_folder="./data/tecnicas/capital_mv_sharpe_tecnicas_10/",
    capital_inicial=100.0,
    dias_max_posicao=4,
):
    run_capital_mv_tecnicas(
        weights_folder=weights_folder,
        targets_folder=targets_folder,
        output_folder=output_folder,
        metodo="mv_sharpe",
        capital_inicial=capital_inicial,
        dias_max_posicao=dias_max_posicao,
    )

    print("\nDistribuicao de capital MV Sharpe por tecnicas concluida.")


if __name__ == "__main__":
    run_capital_mv_sharpe_tecnicas()
