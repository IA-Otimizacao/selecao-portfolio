import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_capital(
    input_folder="./data/ensemble/9_capital/",
    output_folder="./src/visualization"
):

    os.makedirs(output_folder, exist_ok=True)

    algoritmos = ['esmble_jan_tot', 'esmble_jan_par', 'in_precision']

    # 🎨 cores por target (fixas)
    cores = ['blue', 'green', 'red']

    # 📈 estilo por algoritmo
    estilos = {
        'esmble_jan_tot': '-',
        'esmble_jan_par': '--',
        'in_precision': ':'
    }

    for fname in os.listdir(input_folder):

        if not fname.lower().endswith(".csv"):
            continue

        path = os.path.join(input_folder, fname)

        print(f"\n📂 Gerando gráfico: {fname}")

        df = pd.read_csv(path, sep="|")
        df['data'] = pd.to_datetime(df['data'])

        plt.figure(figsize=(14, 7))

        targets = sorted(df['target'].unique())

        for i, target in enumerate(targets):

            cor = cores[i % len(cores)]

            df_target = df[df['target'] == target]

            for alg in algoritmos:

                plt.plot(
                    df_target['data'],
                    df_target[f'capital_teorico_{alg}'],
                    linestyle=estilos[alg],
                    color=cor,
                    label=f"{alg} | target={target}"
                )

        plt.title(f"Capital Teórico ao Longo do Tempo\n{fname}")
        plt.xlabel("Data")
        plt.ylabel("Capital")
        plt.legend()
        plt.grid(True)

        out_path = os.path.join(
            output_folder,
            fname.replace('.csv', '_capital_plot.png')
        )

        plt.savefig(out_path)
        plt.close()

        print(f"✅ Salvo em: {out_path}")


def run_plot():
    plot_capital()


if __name__ == "__main__":
    run_plot()