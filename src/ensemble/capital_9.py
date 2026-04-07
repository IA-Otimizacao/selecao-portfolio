import os
import pandas as pd
import numpy as np


def calcula_capital(
    input_folder="./data/ensemble/8_monetario/",  # 📥 Entrada: já com compra/venda definidos
    output_folder="./data/ensemble/9_capital/"    # 📤 Saída: evolução do capital
):

    # Garante existência da pasta de saída
    os.makedirs(output_folder, exist_ok=True)

    # 🤖 Algoritmos avaliados
    algoritmos = ['esmble_jan_tot', 'esmble_jan_par', 'in_precision']

    # 🔁 Loop nos arquivos
    for fname in os.listdir(input_folder):

        # Ignora não-CSV
        if not fname.lower().endswith(".csv"):
            continue

        path = os.path.join(input_folder, fname)

        print(f"\n📂 Processando arquivo: {fname}")

        # 📥 Leitura
        df = pd.read_csv(path, sep="|")
        df['data'] = pd.to_datetime(df['data'])

        # Ordenação temporal
        df = df.sort_values(['ativo', 'target', 'data'])

        results = []

        # 🔁 Loop por ativo e target
        for (ativo, target), g in df.groupby(['ativo', 'target'], sort=False):

            print(f"  ▶ ativo={ativo} | target={target}")

            g = g.reset_index(drop=True)

            # 🔁 Loop por algoritmo (simulação independente)
            for alg in algoritmos:

                # =========================
                # 📦 VETORES DE SAÍDA
                # =========================
                dinheiro = np.zeros(len(g))   # 💵 caixa disponível
                investido = np.zeros(len(g))  # 📈 capital alocado
                qtde = np.zeros(len(g))       # 📊 quantidade de ativos

                # =========================
                # 📊 ESTADO INICIAL
                # =========================
                capital = 1000          # 💰 capital inicial fixo
                posicao_aberta = False  # indica se está comprado
                qtde_ativos = 0         # quantidade de ativos na posição

                # 🔁 LOOP TEMPORAL
                for i in range(len(g)):

                    # 📥 Sinais de compra/venda já calculados anteriormente
                    compra = g[f'compra_{alg}'].iloc[i]
                    venda = g[f'venda_{alg}'].iloc[i]

                    # =========================
                    # 📌 SEM POSIÇÃO (CAIXA)
                    # =========================
                    if not posicao_aberta:

                        if compra > 0:
                            # 💸 ENTRA NA OPERAÇÃO (USA TODO CAPITAL)

                            investido[i] = capital              # todo capital vira posição
                            qtde_ativos = capital / compra      # compra ativos
                            qtde[i] = qtde_ativos
                            dinheiro[i] = 0                     # fica sem caixa

                            posicao_aberta = True

                        else:
                            # 💵 Continua fora do mercado
                            dinheiro[i] = capital
                            investido[i] = 0
                            qtde[i] = 0

                    # =========================
                    # 📌 COM POSIÇÃO ABERTA
                    # =========================
                    else:
                        # 📈 Mantém posição (mark-to-market simplificado)
                        investido[i] = capital   # capital "travado"
                        qtde[i] = qtde_ativos
                        dinheiro[i] = 0

                    # =========================
                    # 📌 EVENTO DE VENDA
                    # =========================
                    if posicao_aberta and venda > 0:

                        # 💰 Realiza capital (encerra operação)
                        capital = qtde_ativos * venda

                        # 🔄 Zera posição
                        posicao_aberta = False
                        qtde_ativos = 0

                # =========================
                # 📊 SALVA COLUNAS
                # =========================
                g[f'dinheiro_{alg}'] = dinheiro
                g[f'investido_{alg}'] = investido
                g[f'qtde_ativos_{alg}'] = qtde

                # =========================
                # 📌 CAPITAL TEÓRICO (EQUITY)
                # =========================
                # Aqui você constrói sua equity curve
                g[f'capital_teorico_{alg}'] = (
                    g[f'dinheiro_{alg}'] + g[f'investido_{alg}']
                )

            # Guarda resultado desse ativo/target
            results.append(g)

        # 🔗 Junta tudo
        if results:
            final_df = pd.concat(results, ignore_index=True)

            # 📤 Caminho de saída
            out_path = os.path.join(
                output_folder,
                fname.replace('.csv', '_capital.csv')
            )

            # 💾 Salva
            final_df.to_csv(out_path, index=False, sep="|")

            print(f"✅ Salvo em: {out_path}")
        else:
            print("⚠️ Nenhum dado processado.")


# 🚀 Runner
def run_calculo_capital():
    calcula_capital()


# ▶ Execução direta
if __name__ == "__main__":
    run_calculo_capital()