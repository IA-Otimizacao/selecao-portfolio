# 📈 Portfolio Selection & Investment Strategy Simulator

Pipeline completo para **modelagem preditiva**, **ensemble de classificadores** e **simulação monetária de estratégias de investimento** no mercado brasileiro (B3).

O projeto transforma dados financeiros brutos em **decisões de investimento simuladas**, permitindo comparar diferentes estratégias a partir da **evolução do capital ao longo do tempo**.

---

## 🚀 O que esse projeto faz

✔️ Processa dados financeiros históricos  
✔️ Treina modelos de machine learning para previsão de alvos de retorno  
✔️ Combina modelos usando diferentes técnicas de ensemble  
✔️ Simula estratégias de investimento com janelas temporais fixas  
✔️ Gera curvas de capital para comparação entre estratégias  

Tudo de forma **modular**, **reprodutível** e **automatizada**.

---

## 🧠 Ideia central

A cada janela de tempo, o sistema decide:

- **Encerrar a operação antecipadamente**, caso um sinal binário seja ativado, ou
- **Manter até o final da janela**, usando o retorno real observado

O capital é atualizado de forma **multiplicativa**, permitindo analisar:
- crescimento acumulado  
- estabilidade da estratégia  
- impacto de diferentes sinais preditivos  

---

## 🗂️ Estrutura do Projeto

```text
src/
├── main.py                  # Orquestrador principal do pipeline
│
├── motor/                   # Pipelines de alto nível
│   ├── pre_process.py       # Pré-processamento de dados
│   ├── train.py             # Treinamento e avaliação de modelos
│   └── ensemble.py          # Ensemble + simulação monetária
│
├── pre_process/             # Tratamento e preparação dos dados
│   ├── xlsx_to_csv.py
│   ├── ibov.py
│   └── inputs.py
│
├── ensemble/                # Técnicas de ensemble e análises
│   ├── comparison_tec_1.py
│   ├── comparison_jan_2.py
│   ├── comparison_prec_3.py
│   ├── comparison_esmbs_4.py
│   ├── acuracia_precision_5.py
│   ├── completo_6.py
│   └── monetary_7.py
│
├── utils.py                 # Funções utilitárias
└── plot.py                  # Geração de gráficos


---

## 📁 Estrutura de Diretórios – Dados

data/
├── pre_process/                    # Dados brutos, sem qualquer tratamento
│   ├── ibov/
│   ├── raw/
│   └── curated/
│
├── train/              # Dados tratados e prontos para uso
│   ├── analytics/
│   ├── inputs/
│   └── outputs/
│
└──ensemble/               # Resultados e análises dos modelos ensemble
    ├── 1_comparison/
    ├── 2_tot_par/
    ├── 3_precision/
    ├── 4_melhor_precision_valor/
    ├── 5__6_completo/
    └── 7_monetario/


---

## ▶️ Como executar
🔹 Execução completa (recomendada)
python -m src.main

🔹 Execução por etapa
python -m src.motor.pre_process
python -m src.motor.train
python -m src.motor.ensemble


---

## 📌 Observações

- Os gráficos salvos em `visualization` são organizados por **ativo e target**, permitindo fácil comparação entre diferentes estratégias e técnicas de modelagem.  
- Certifique-se de que os caminhos das pastas estejam corretos antes de executar os scripts.  

---

## ⚙️ Requisitos

- Python 3.12 ou superior
- Bibliotecas: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `ta-lib`, `tqdm`, `feature_engine`, `openpyxl`
- Instalação recomendada via `pip install -r requirements.txt` (arquivo de requisitos incluído no repositório)
