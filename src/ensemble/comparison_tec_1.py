import pandas as pd
from tqdm import tqdm
import os


def run_comparison_tec():
    os.makedirs("./data/ensemble/comparison", exist_ok=True)
    todos = ['PETR4', 'ITUB4', 'VALE3']

    # FUNÇÃO RESPONSÁVEL POR CARREGAR E PROCESSAR OS DADOS
    def carregar_dados_comparacao(file):

        # Carrega os resultados gerados na etapa de modelagem
        base_dados = pd.read_csv(f'./data/train/outputs/target_previsto_{file}.csv')

        # Conversão da coluna de data
        if 'data' in base_dados.columns:
            base_dados['data'] = pd.to_datetime(base_dados['data'], errors='coerce')

        # Definição das colunas que identificarão cada observação
        index_cols = ['ativo', 'target', 'janela', 'data', 'target_real']

        # Se o resultado financeiro real existir, adiciona ao índice
        if 'resultado_real' in base_dados.columns:
            index_cols.append('resultado_real')

        # Transformação da base para formato de comparação
        tabela = (
            base_dados
            .pivot_table(
                index=index_cols,      # identificação da observação
                columns='tecnica',     # cada técnica vira uma coluna
                values='target_pred',  # valor previsto
                aggfunc='first'        # garante um valor único
            )
            .reset_index()
        )

        # Remove o nome do índice de colunas criado pelo pivot
        tabela.columns.name = None  

        # CRIAÇÃO DO ENSEMBLE DE TÉCNICAS
        if all(col in tabela.columns for col in ['RNA', 'Random Forest', 'SVC']):

            # Soma as previsões dos três modelos
            soma = tabela['RNA'] + tabela['Random Forest'] + tabela['SVC']

            # Filtro de consenso entre modelos.
            tabela['ensemble_tecnicas'] = soma.apply(
                lambda x: 1 if x in [0, 3] else 0
            )

        else:
            # Caso alguma técnica esteja ausente o ensemble não pode ser calculado
            tabela['ensemble_tecnicas'] = None

        return tabela


    # LOOP PRINCIPAL DE PROCESSAMENTO
    for file in tqdm(todos, desc="Ativos"):

        df = carregar_dados_comparacao(file)

        # Salva o resultado final em CSV
        df.to_csv(
            f'./data/ensemble/1_comparison/{file}_comparison.csv',
            index=False
        )

    print("✅ comparison_tec concluído")