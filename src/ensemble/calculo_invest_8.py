import os
import pandas as pd
import numpy as np


def calcula_investimento(
    input_folder="./data/ensemble/7_intraday_join/",  # 📥 Dados já com intraday + labels
    output_folder="./data/ensemble/8_monetario/"      # 📤 Saída com simulação financeira
):

    # Garante que a pasta de saída existe
    os.makedirs(output_folder, exist_ok=True)

    # 🤖 Lista dos algoritmos/sinais que serão testados
    algoritmos = ['esmble_jan_tot', 'esmble_jan_par', 'in_precision']

    # 🔁 Loop nos arquivos de entrada
    for fname in os.listdir(input_folder):

        # Ignora arquivos que não são CSV
        if not fname.lower().endswith(".csv"):
            continue

        path = os.path.join(input_folder, fname)

        print(f"\n📂 Processando arquivo: {fname}")

        # 📥 Leitura dos dados
        df = pd.read_csv(path, sep="|")

        # 🔄 Ajuste de tipo e ordenação temporal
        df['data'] = pd.to_datetime(df['data'])
        df = df.sort_values(['ativo', 'target', 'data'])

        results = []  # Acumulador final

        # 🔁 Loop principal por ativo e target
        for (ativo, target), g in df.groupby(['ativo', 'target'], sort=False):

            print(f"  ▶ Processando | ativo={ativo} | target={target}")

            g = g.reset_index(drop=True)

            # 🔁 Loop por algoritmo (cada estratégia independente)
            for alg in algoritmos:

                # =========================
                # 📦 VETORES DE CONTROLE
                # =========================
                compra = np.zeros(len(g))  # preço de entrada da operação
                rend = np.zeros(len(g))    # retorno atual da operação
                dias = np.zeros(len(g))    # dias em holding

                # =========================
                # 📊 ESTADO DA OPERAÇÃO
                # =========================
                preco_compra = 0  # preço fixo da compra atual
                contador = 0      # controla duração da operação

                # 🔁 SHIFT DO SINAL (COMPRA D+1)
                # Compra só acontece se ontem deu sinal
                sinal_shift = g[alg].shift(1)

                # 🔁 LOOP TEMPORAL (dia a dia)
                for i in range(len(g)):

                    # =========================
                    # 📌 INÍCIO DE NOVA COMPRA
                    # =========================
                    if contador == 0:  # não estou posicionado

                        if sinal_shift.iloc[i] == 1:  # sinal positivo no dia anterior
                            preco_compra = g['Open'].iloc[i]  # compra na abertura
                            compra[i] = preco_compra
                            contador = 1  # inicia holding

                        else:
                            # sem sinal → não entra na operação
                            compra[i] = 0
                            dias[i] = 0
                            continue  # pula resto do loop

                    # =========================
                    # 📌 CONTINUAÇÃO DA POSIÇÃO
                    # =========================
                    else:
                        # mantém o preço de compra fixo
                        compra[i] = preco_compra

                    # =========================
                    # 📌 CÁLCULO DO RETORNO
                    # =========================
                    # retorno baseado no HIGH (melhor cenário intraday)
                    rend[i] = (g['High'].iloc[i] - preco_compra) / preco_compra

                    # =========================
                    # 📌 REGRA DE SAÍDA (TARGET)
                    # =========================
                    atingiu = rend[i] >= (target - 1)

                    if atingiu:
                        # 🎯 bateu target → encerra operação
                        dias[i] = 0
                        contador = 0
                        preco_compra = 0

                    else:
                        # ⏳ continua operação
                        dias[i] = contador
                        contador += 1

                        # ⛔ STOP DE TEMPO (máx 4 dias)
                        if contador > 4:
                            contador = 0
                            preco_compra = 0

                # =========================
                # 📊 SALVA RESULTADOS DO ALGORITMO
                # =========================
                g[f'compra_{alg}'] = compra
                g[f'rend_decisao_{alg}'] = rend
                g[f'dias_{alg}'] = dias

                # =========================
                # 📌 REGRA DE VENDA (MONETIZAÇÃO)
                # =========================
                venda = np.where(
                    # 🎯 Se zerou dias → vendeu no target
                    g[f'dias_{alg}'] == 0,
                    g[f'compra_{alg}'] * target,

                    # ⏳ Se chegou no último dia (4) → vende no fechamento
                    np.where(
                        g[f'dias_{alg}'] == 4,
                        g['Close'],
                        0  # caso contrário, ainda em holding
                    )
                )

                g[f'venda_{alg}'] = venda

            # Guarda resultado desse ativo/target
            results.append(g)

        # 🔗 Junta todos os resultados do arquivo
        if results:
            final_df = pd.concat(results, ignore_index=True)

            # 📤 Define caminho de saída
            out_path = os.path.join(
                output_folder,
                fname.replace('.csv', '_monetario.csv')
            )

            # 💾 Salva
            final_df.to_csv(out_path, index=False, sep="|")

            print(f"✅ Salvo em: {out_path}")
        else:
            print("⚠️ Nenhum dado processado.")


# 🚀 Runner
def run_invest():
    calcula_investimento()


# ▶ Execução direta
if __name__ == "__main__":
    run_invest()