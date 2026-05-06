import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configurações
pasta = Path("./data/ensemble/11_otimi")
tecnicas = ['esmble_jan_tot', 'esmble_jan_par', 'in_precision']
alvos = ['1_01', '1_015', '1_02']

# Mapeamentos visuais
cores = {
    '1_01': 'blue',
    '1_015': 'green',
    '1_02': 'red'
}
estilos_linha = {
    'esmble_jan_tot': '-',      # linha sólida
    'esmble_jan_par': '--',     # tracejada
    'in_precision': ':'         # pontilhada
}

# Leitura dos dados
dados = []
for tecnica in tecnicas:
    for alvo in alvos:
        arquivo = pasta / f"{tecnica}_target_{alvo}.csv"
        if not arquivo.exists():
            print(f"Aviso: arquivo não encontrado - {arquivo}")
            continue
        df = pd.read_csv(arquivo)
        # Converte a coluna 'data' para datetime (ajuste o formato se necessário)
        df['data'] = pd.to_datetime(df['data'])
        df['tecnica'] = tecnica
        df['alvo'] = alvo
        dados.append(df)

if not dados:
    raise FileNotFoundError("Nenhum arquivo encontrado. Verifique o caminho e os nomes.")

df_all = pd.concat(dados, ignore_index=True)
df_all.sort_values('data', inplace=True)

# Criação do gráfico
plt.figure(figsize=(12, 6))
for (tecnica, alvo), grupo in df_all.groupby(['tecnica', 'alvo']):
    cor = cores[alvo]
    estilo = estilos_linha[tecnica]
    label = f"{alvo} ({tecnica})"
    plt.plot(grupo['data'], grupo['total_verdadeiro'],
             color=cor, linestyle=estilo, linewidth=2,
             label=label)
plt.xlabel("Data")
plt.ylabel("Total Verdadeiro")
plt.title("Comparação: Alvos (cores) × Técnicas (tracejados)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

# Salvar a imagem
imagem_saida = pasta / "comparacao_alvos_tecnicas.png"
plt.savefig(imagem_saida, dpi=150)
print(f"Gráfico salvo em: {imagem_saida}")

# Opcional: mostrar na tela
# plt.show()