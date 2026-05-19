import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ensemble.comparison_tec_1 import run_comparison_tec
from src.ensemble.comparison_jan_2 import run_comparison_jan
from src.ensemble.comparison_prec_3 import run_comparison_precision
from src.ensemble.comparison_esmbs_4 import run_comparison_esmbs
from src.ensemble.acuracia_precision_5 import run_acuracia_precision
from src.ensemble.completo_6 import gerar_comparacao_completa
from src.ensemble.juncao_intrumentos_7 import run_join_intr
from src.ensemble.calculo_invest_8 import run_invest
from src.ensemble.capital_9 import run_calculo_capital
from src.ensemble.juncao_ativos_10 import run_df_por_target
from src.ensemble.target_tecnica_11 import separar_por_tecnica
from src.ensemble.n_12 import processar_estrategia
from src.ensemble.base_MV_sharpe_13 import run_base_mv_sharpe
from src.ensemble.otimizacao_MV_sharpe_14 import run_otimizacao_mv_sharpe
from src.ensemble.otimizacao_MV_min_variancia_15 import run_otimizacao_mv_min_variancia
from src.ensemble.capital_MV_sharpe_16 import run_capital_mv_sharpe
from src.ensemble.capital_MV_min_variancia_17 import run_capital_mv_min_variancia


def run_ensemble():
    print("\n🚀 Iniciando pipeline de ENSEMBLE\n")

    print("\n[1/17] Comparacao por tecnica")
    run_comparison_tec()

    print("\n[2/17] Comparacao por janela")
    run_comparison_jan()

    print("\n[3/17] Comparacao de precision")
    run_comparison_precision()

    print("\n[4/17] Melhor precision por valor")
    run_comparison_esmbs()

    print("\n[5/17] Acuracia precision")
    run_acuracia_precision()

    print("\n[6/17] Comparacao completa")
    gerar_comparacao_completa()

    print("\n[7/17] Juncao intraday")
    run_join_intr()

    print("\n[8/17] Calculo monetario")
    run_invest()

    print("\n[9/17] Calculo de capital")
    run_calculo_capital()

    print("\n[10/17] Juncao de ativos por target")
    run_df_por_target()

    print("\n[11/17] Separacao por tecnica")
    separar_por_tecnica()

    print("\n[12/17] Estrategia 1/n")
    processar_estrategia()

    print("\n[13/17] Base MV Sharpe")
    run_base_mv_sharpe()

    print("\n[14/17] Otimizacao MV Sharpe")
    run_otimizacao_mv_sharpe()

    print("\n[15/17] Otimizacao MV Min Variancia")
    run_otimizacao_mv_min_variancia()

    print("\n[16/17] Capital MV Sharpe")
    run_capital_mv_sharpe()

    print("\n[17/17] Capital MV Min Variancia")
    run_capital_mv_min_variancia()

    print("\n✅ Pipeline de ensemble finalizado com sucesso!")

if __name__ == "__main__":
    run_ensemble()
