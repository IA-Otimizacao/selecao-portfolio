# Projeto de Simulação de Investimentos

Este repositório contém scripts e dados para tratamento, modelagem, simulação e visualização de estratégias de investimento na B3, incluindo análise de performance de modelos de classificação e simulação de capital ao longo do tempo.

---

## 📂 Estrutura de Pastas

### 1. Scripts (`./src/`)

| Script | Descrição | Ordem de execução |
|--------|-----------|-----------------|
| `utils.py` | Funções utilitárias usadas em todo o projeto | 1 |
| `train.py` | Tratamento de dados e treinamento dos modelos de classificação | 2 |
| `monetary.py` | Implementa simulação de capital e janelas de investimento com base nos resultados do modelo | 3 |
| `plot_monetario.py` | Gera gráficos da evolução do valor obtido ao longo do tempo, separando por ativo, target, janela e técnica | 4 |

> ⚠️ É importante executar os scripts na ordem acima para garantir que os dados e resultados necessários estejam disponíveis para os próximos passos.

---

### 2. Dados (`./data/`)

| Pasta | Descrição |
|-------|-----------|
| `ibov` | Ativos presentes no Ibovespa ao longo do tempo |
| `raw/refinitiv` | Dados brutos retirados diretamente do software Refinitiv |
| `raw/ibov_status` | Dados brutos com alinhamento de série temporal e inclusão do status de presença no Ibovespa |
| `curated` | Dados tratados e preparados com variáveis de treinamento e auxiliares |
| `train_out` | Dados pós-treinamento com a coluna binária resultante da classificação |
| `monetario` | Dados simulados relativos ao capital de investimento e janelas |
| `analytics/results` | Resultados de métricas de acurácia, precisão e outras análises dos modelos treinados |

---

### 3. Gráficos (`./src/visualization/`)

| Tipo | Descrição |
|------|-----------|
| Plot do saldo simulado | Evolução do valor obtido em relação ao capital inicial ao longo do tempo, separados por ativo e target, com diferentes linhas para cada combinação de janela e técnica |

---

## 🚀 Ordem sugerida de execução

1. Garantir que os dados brutos estejam em `./data/raw/`  
2. Executar `train.py` → gera dados tratados (`curated`) e saída do modelo (`train_out`)  
3. Executar `monetary.py` → gera simulações de capital em `./data/monetario/`  
4. Executar `plot_monetario.py` → gera gráficos em `./src/visualization/`  

---

## 📌 Observações

- Os gráficos salvos em `visualization` são organizados por **ativo e target**, permitindo fácil comparação entre diferentes estratégias e técnicas de modelagem.  
- Certifique-se de que os caminhos das pastas estejam corretos antes de executar os scripts.  

---

## ⚙️ Requisitos

- Python 3.12 ou superior
- Bibliotecas: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `ta-lib`, `tqdm`, `feature_engine`, `openpyxl`
- Instalação recomendada via `pip install -r requirements.txt` (arquivo de requisitos incluído no repositório)
