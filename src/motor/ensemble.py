from src.ensemble.comparison_tec_1 import run_comparison_tec
from src.ensemble.comparison_jan_2 import run_comparison_jan
from src.ensemble.comparison_prec_3 import run_comparison_precision
from src.ensemble.comparison_esmbs_4 import run_comparison_esmbs
from src.ensemble.acuracia_precision_5 import run_acuracia_precision
from src.ensemble.completo_6 import gerar_comparacao_completa
from src.ensemble.monetary_7 import run_monetary
from src.plot import gerar_graficos_capital

def run_ensemble():
    print("\n🚀 Iniciando pipeline de ENSEMBLE\n")

    run_comparison_tec()
    run_comparison_jan()
    run_comparison_precision()
    run_comparison_esmbs()
    run_acuracia_precision()
    gerar_comparacao_completa()
    run_monetary()
    gerar_graficos_capital()

    print("\n✅ Pipeline de ensemble finalizado com sucesso!")

if __name__ == "__main__":
    run_ensemble()
