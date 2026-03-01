import pandas as pd
from tqdm import tqdm
import os


def run_comparison_jan():

    todos = ['PETR4', 'ITUB4', 'VALE3']
    os.makedirs("./data/ensemble/2_tot_par", exist_ok=True)

    # FUNÇÃO RESPONSÁVEL POR CARREGAR OS DADOS E CRIAR ENSEMBLES
    def carregar_e_criar_ensembles(file):

        # CARREGAMENTO DOS DADOS DA ETAPA ANTERIOR
        base = pd.read_csv(f'./data/ensemble/1_comparison/{file}_comparison.csv')

        # REORGANIZAÇÃO DOS DADOS (PIVOT TABLE)
        # O objetivo aqui é reorganizar os dados para que cada combinação "técnica + janela" vire uma coluna.
        # Assim será possível calcular ensembles entre janelas.
        tabela = (
            base
            .pivot_table(
                index=['ativo', 'target', 'data', 'target_real', 'resultado_real'],
                columns='janela',              # cada janela vira uma coluna
                values=['RNA', 'Random Forest', 'SVC'],  # previsões das técnicas
                aggfunc='first'
            )
        )

        # AJUSTE DOS NOMES DAS COLUNAS. Ex: RNA_60
        tabela.columns = [f"{c[0]}_{c[1]}" for c in tabela.columns]

        # Reset do índice para voltar ao formato de DataFrame comum
        tabela = tabela.reset_index()

        # Remove o espaço no nome "Random Forest" para evitar problemas ao acessar as colunas
        tabela.columns = [
            c.replace("Random Forest", "RandomForest")
            for c in tabela.columns
        ]

        # IDENTIFICAÇÃO DAS COLUNAS DE PREVISÃO
        tecnicas_cols = [
            c for c in tabela.columns
            if any(t in c for t in ['RNA', 'RandomForest', 'SVC'])
        ]

        # REMOÇÃO DE LINHAS COM VALORES AUSENTES
        tabela = tabela.dropna(subset=tecnicas_cols)

        # ENSEMBLE TOTAL ENTRE JANELAS

        # Existem 9 previsões, se todas as previsões forem 1: soma = 9 então o ensemble total é ativado.
        tabela['esmble_jan_tot'] = (
            tabela[tecnicas_cols].sum(axis=1) == 9
        ).astype(int)

        # ENSEMBLE PARCIAL ENTRE JANELAS
        #
        # Critérios: Pelo menos 5 previsões positivas no total e pelo menos uma previsão de cada tecnica.
        tabela['esmble_jan_par'] = (
            (
                (tabela[tecnicas_cols].sum(axis=1) >= 5)
                &
                (tabela[['RNA_60', 'RNA_75', 'RNA_90']].sum(axis=1) >= 1)
                &
                (tabela[['RandomForest_60', 'RandomForest_75', 'RandomForest_90']].sum(axis=1) >= 1)
                &
                (tabela[['SVC_60', 'SVC_75', 'SVC_90']].sum(axis=1) >= 1)
            )
        ).astype(int)

        return tabela, tecnicas_cols


    # LOOP PRINCIPAL PARA PROCESSAR CADA ATIVO
    for file in tqdm(todos):

        # Carrega os dados e calcula os ensembles
        df, cols = carregar_e_criar_ensembles(file)


        # ------------------------------------------------------
        # SALVAMENTO DOS RESULTADOS
        # ------------------------------------------------------
        # O arquivo final conterá: identificadores da observação, previsões individuais dos modelos, ensemble total e parcial.
        df[
            ['ativo', 'target', 'data', 'target_real', 'resultado_real']
            + cols
            + ['esmble_jan_tot', 'esmble_jan_par']
        ].to_csv(
            f'./data/ensemble/2_tot_par/{file}_ensemble_jan_tot_e_parcial.csv',
            index=False
        )


    print("✅ comparison_jan concluído")