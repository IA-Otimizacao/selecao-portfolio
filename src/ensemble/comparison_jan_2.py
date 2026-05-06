import pandas as pd
from tqdm import tqdm
import os
import re


def run_comparison_jan():

    os.makedirs("./data/ensemble/2_tot_par", exist_ok=True)

    # =========================
    # PEGA TODOS OS ATIVOS DA PASTA
    # =========================
    pasta_input = "./data/ensemble/1_comparison/"

    arquivos = [
        f for f in os.listdir(pasta_input)
        if f.endswith("_comparison.csv")
    ]

    todos = [
        re.search(r"(.*)_comparison\.csv", f).group(1)
        for f in arquivos
    ]

    # =========================
    # FUNÇÃO DE PROCESSAMENTO
    # =========================
    def carregar_e_criar_ensembles(file):

        base = pd.read_csv(f'{pasta_input}{file}_comparison.csv')

        # Converte data para datetime para ordenação correta
        if 'data' in base.columns:
            base['data'] = pd.to_datetime(base['data'], errors='coerce')

        tabela = (
            base
            .pivot_table(
                index=['ativo', 'target', 'data', 'target_real', 'resultado_real'],
                columns='janela',
                values=['RNA', 'Random Forest', 'SVC'],
                aggfunc='first'
            )
        )

        # flatten colunas
        tabela.columns = [f"{c[0]}_{c[1]}" for c in tabela.columns]
        tabela = tabela.reset_index()

        # padroniza nome
        tabela.columns = [
            c.replace("Random Forest", "RandomForest")
            for c in tabela.columns
        ]

        # identifica colunas de previsão
        tecnicas_cols = [
            c for c in tabela.columns
            if any(t in c for t in ['RNA', 'RandomForest', 'SVC'])
        ]

        # remove linhas incompletas
        tabela = tabela.dropna(subset=tecnicas_cols)

        # =========================
        # ENSEMBLE TOTAL (DINÂMICO)
        # =========================
        tabela['esmble_jan_tot'] = (
            tabela[tecnicas_cols].sum(axis=1) == len(tecnicas_cols)
        ).astype(int)

        # =========================
        # ENSEMBLE PARCIAL (DINÂMICO)
        # =========================
        rna_cols = [c for c in tecnicas_cols if c.startswith('RNA_')]
        rf_cols = [c for c in tecnicas_cols if c.startswith('RandomForest_')]
        svc_cols = [c for c in tecnicas_cols if c.startswith('SVC_')]

        tabela['esmble_jan_par'] = (
            (
                (tabela[tecnicas_cols].sum(axis=1) >= int(len(tecnicas_cols) * 0.55))
                &
                (tabela[rna_cols].sum(axis=1) >= 1)
                &
                (tabela[rf_cols].sum(axis=1) >= 1)
                &
                (tabela[svc_cols].sum(axis=1) >= 1)
            )
        ).astype(int)

        # 🔽 ORDENAÇÃO: primeiro pelo target, depois pela data
        tabela = tabela.sort_values(['target', 'data'])

        return tabela, tecnicas_cols

    # =========================
    # LOOP PRINCIPAL
    # =========================
    for file in tqdm(todos, desc="Ativos"):

        df, cols = carregar_e_criar_ensembles(file)

        df[
            ['ativo', 'target', 'data', 'target_real', 'resultado_real']
            + cols
            + ['esmble_jan_tot', 'esmble_jan_par']
        ].to_csv(
            f'./data/ensemble/2_tot_par/{file}_ensemble_jan_tot_e_parcial.csv',
            index=False
        )

    print("✅ comparison_jan concluído")


if __name__ == "__main__":
    run_comparison_jan()