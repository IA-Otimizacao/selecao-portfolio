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
from src.ensemble.n import processar_estrategia


def run_ensemble():
    print("\n🚀 Iniciando pipeline de ENSEMBLE\n")

    run_comparison_tec()
    run_comparison_jan()
    run_comparison_precision()
    run_comparison_esmbs()
    run_acuracia_precision()
    gerar_comparacao_completa()
    run_join_intr()
    run_invest()
    run_calculo_capital()
    run_df_por_target()
    separar_por_tecnica()
    # processar_estrategia()



    print("\n✅ Pipeline de ensemble finalizado com sucesso!")

if __name__ == "__main__":
    run_ensemble()
