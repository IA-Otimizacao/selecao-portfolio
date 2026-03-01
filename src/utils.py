import pandas as pd
import numpy as np
import talib
import warnings
import time
import os
import locale

# Modelos e ferramentas de ML
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Métricas de avaliação
from sklearn.metrics import accuracy_score, precision_score

# Validação de hiperparâmetros
from sklearn.model_selection import GridSearchCV

# Seleção de features
from sklearn.feature_selection import VarianceThreshold
from feature_engine.selection import DropCorrelatedFeatures


# ======================================================
# DECORATOR PARA MEDIR TEMPO DE EXECUÇÃO
# ======================================================
def medir_tempo(func):
    """
    Decorator que mede quanto tempo uma função leva para executar.
    Muito útil para identificar gargalos no pipeline.
    """
    def wrapper(*args, **kwargs):

        # Marca o início da execução
        inicio = time.time()

        # Executa a função original
        resultado = func(*args, **kwargs)

        # Marca o final da execução
        fim = time.time()

        # Calcula duração
        duracao = fim - inicio

        # Exibe tempo no terminal
        print(f"\033[92m⏱ Função '{func.__name__}' executada em {duracao:.2f} segundos.\033[0m")

        return resultado

    return wrapper


# ======================================================
# FUNÇÃO PARA CARREGAR DADOS DOS ATIVOS
# ======================================================
def carregar_dados(file):
    try:
        locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
    except:
        pass

    # Lê arquivo CSV do Refinitiv
    base_dados = pd.read_csv(f'./data/raw/refinitiv/{file}.csv')

    # Remove pontos e espaços da coluna de data
    base_dados['Exchange Date'] = base_dados['Exchange Date'].str.replace('.', '', regex=False).str.strip()

    # Converte coluna de data para datetime
    base_dados['Exchange Date'] = pd.to_datetime(
        base_dados['Exchange Date'],
        format='%d-%b-%Y',
        errors='coerce'
    )

    return base_dados


# ======================================================
# CARREGAR STATUS DO IBOVESPA
# ======================================================
def carregar_ibov(ibov):

    # Lê planilha contendo status dos ativos no índice
    ibov_status = pd.read_excel(f'./data/ibov/{ibov}.xlsx')

    # Converte datas
    ibov_status['Exchange Date'] = pd.to_datetime(ibov_status['Exchange Date'])

    return ibov_status


# ======================================================
# PADRONIZAÇÃO DAS COLUNAS NUMÉRICAS
# ======================================================
def padronizar_colunas(base_dados):

    for col in base_dados.columns:

        # Se for string, padroniza separadores
        if base_dados[col].dtype == 'object':

            base_dados[col] = base_dados[col].str.replace('.', '')
            base_dados[col] = base_dados[col].str.replace(',', '.')

        # Converte para número (exceto data)
        if col != 'Exchange Date':

            base_dados[col] = pd.to_numeric(base_dados[col], errors='coerce')

    return base_dados


# ======================================================
# REMOVER LINHAS INVÁLIDAS
# ======================================================
def remover_linhas_invalidas(base_dados):

    colunas_verificacao = ['Close', 'Open', 'Low', 'High', 'Volume']

    # Remove linhas com valores nulos
    base_dados = base_dados.dropna(subset=colunas_verificacao)

    # Remove valores negativos ou zero
    for col in colunas_verificacao:

        base_dados = base_dados.loc[base_dados[col] > 0]

    return base_dados


# ======================================================
# GARANTIR QUE COLUNAS SÃO NUMÉRICAS
# ======================================================
def garantir_numerico(base_dados, colunas):

    for col in colunas:

        if col in base_dados.columns:

            base_dados[col] = pd.to_numeric(base_dados[col], errors='coerce')

        else:

            print(f"Aviso: coluna '{col}' não encontrada no DataFrame.")

    return base_dados


# ======================================================
# CRIAÇÃO DAS VARIÁVEIS (FEATURE ENGINEERING)
# ======================================================
def calcular_variaveis(base_dados):

    # =========================
    # RETORNOS LOGARÍTMICOS
    # =========================

    base_dados['r1'] = np.log(base_dados['Close'] / base_dados['Close'].shift(1))
    base_dados['r2'] = np.log(base_dados['Close'].shift(1) / base_dados['Close'].shift(2))
    base_dados['r3'] = np.log(base_dados['Close'].shift(2) / base_dados['Close'].shift(3))
    base_dados['r4'] = np.log(base_dados['Close'].shift(3) / base_dados['Close'].shift(4))

    # Relações entre preços de abertura e máximos
    base_dados['r5'] = np.log(base_dados['High'] / base_dados['Open'])
    base_dados['r6'] = np.log(base_dados['High'] / base_dados['Open'].shift(1))
    base_dados['r7'] = np.log(base_dados['High'] / base_dados['Open'].shift(2))
    base_dados['r8'] = np.log(base_dados['High'] / base_dados['Open'].shift(3))

    # Relações entre máximos e abertura
    base_dados['r9'] = np.log(base_dados['High'].shift(1) / base_dados['Open'].shift(1))
    base_dados['r10'] = np.log(base_dados['High'].shift(2) / base_dados['Open'].shift(2))
    base_dados['r11'] = np.log(base_dados['High'].shift(3) / base_dados['Open'].shift(3))

    # Relações com mínimas
    base_dados['r12'] = np.log(base_dados['Low'] / base_dados['Open'])
    base_dados['r13'] = np.log(base_dados['Low'].shift(1) / base_dados['Open'].shift(1))
    base_dados['r14'] = np.log(base_dados['Low'].shift(2) / base_dados['Open'].shift(2))
    base_dados['r15'] = np.log(base_dados['Low'].shift(3) / base_dados['Open'].shift(3))

    # =========================
    # INDICADORES TÉCNICOS
    # =========================

    base_dados['Momentum'] = talib.MOM(base_dados['Close'], timeperiod=10)

    base_dados['RSI'] = talib.RSI(base_dados['Close'], timeperiod=14)

    base_dados['Parabolic_SAR'] = talib.SAR(
        base_dados['High'],
        base_dados['Low'],
        acceleration=0,
        maximum=0
    )

    base_dados['ATR'] = talib.ATR(
        base_dados['High'],
        base_dados['Low'],
        base_dados['Close'],
        timeperiod=14
    )

    base_dados['True_Range'] = talib.TRANGE(
        base_dados['High'],
        base_dados['Low'],
        base_dados['Close']
    )

    base_dados['Chaikin_AD'] = talib.AD(
        base_dados['High'],
        base_dados['Low'],
        base_dados['Close'],
        base_dados['Volume']
    )

    base_dados['OBV'] = talib.OBV(
        base_dados['Close'],
        base_dados['Volume']
    )

    return base_dados


# ======================================================
# SEPARAÇÃO DAS FEATURES
# ======================================================
def x_split(base_dados):

    colunas_de_interesse = [
        'r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11',
        'r12','r13','r14','r15',
        'Momentum','RSI','Parabolic_SAR','ATR','True_Range','Chaikin_AD','OBV'
    ]

    return base_dados[colunas_de_interesse].copy()


# ======================================================
# SEPARAÇÃO DO TARGET
# ======================================================
def y_split(base_dados):

    return base_dados['target'].values


# ======================================================
# VARIÁVEIS AUXILIARES (DATA E RESULTADO REAL)
# ======================================================
def z_split(base_dados):

    return base_dados[['Exchange Date','resultado_real','target']].copy()


# ======================================================
# SELEÇÃO DE FEATURES
# ======================================================
@medir_tempo
def features_selection(x_dados, correlation_threshold=0.8):

    colunas_de_interesse = [
        'r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11',
        'r12','r13','r14','r15',
        'Momentum','RSI','Parabolic_SAR','ATR','True_Range','Chaikin_AD','OBV'
    ]

    # Remove features com variância muito baixa
    fs_qconst = VarianceThreshold(threshold=0.01)

    x_dados_filtrado = fs_qconst.fit_transform(x_dados)

    # Recupera nomes das colunas selecionadas
    colunas_selecionadas = np.array(colunas_de_interesse)[fs_qconst.get_support()]

    x_dados_filtrado = pd.DataFrame(
        x_dados_filtrado,
        columns=colunas_selecionadas,
        index=x_dados.index
    )

    # Remove features altamente correlacionadas
    sel = DropCorrelatedFeatures(
        threshold=correlation_threshold,
        method='pearson',
        missing_values='ignore'
    )

    return sel.fit_transform(x_dados_filtrado)


# ======================================================
# CÁLCULO DO TARGET
# ======================================================
def calcular_target(base_dados, target, janela_temporal):

    base_dados = base_dados.reset_index(drop=True)

    # Preço de compra no dia seguinte
    base_dados['compra'] = base_dados['Open'].shift(-1)

    # Máximo dentro da janela
    base_dados['maximo'] = base_dados['High'].rolling(
        window=janela_temporal
    ).max().shift(-janela_temporal)

    # Preço de venda no final da janela
    base_dados['venda'] = base_dados['Close'].shift(-janela_temporal)

    base_dados['target'] = 0

    # Verifica se o target foi atingido
    for i in range(len(base_dados) - janela_temporal):

        compra = base_dados.at[i,'compra']

        for j in range(1, janela_temporal+1):
             # Verifica se o preço máximo (High) daquele dia atingiu o valor necessário para gerar o lucro desejado.
            if base_dados.at[i+j,'High'] >= compra * target:
                # Se o target foi atingido, assumimos que a venda foi executada exatamente no preço alvo.
                base_dados.at[i,'venda'] = compra * target
                # Define o target como atingido
                base_dados.at[i,'target'] = 1
                break

    # Retorno real da operação
    base_dados['resultado_real'] = (base_dados['venda'] / base_dados['compra']) - 1

    return base_dados


# ======================================================
# TREINO E AVALIAÇÃO DOS MODELOS
# ======================================================
@medir_tempo
def treinar_e_avaliar(file, x_treino, y_treino, x_teste, y_teste_real,
                      parametros_rna, parametros_rf, parametros_svc):

    """
    Esta função treina e avalia três algoritmos de Machine Learning
    utilizados para prever se uma operação de trade atingirá ou não
    o target de lucro definido.

    Modelos utilizados:
    • Rede Neural (MLPClassifier)
    • Random Forest
    • Support Vector Classifier (SVC)

    A função executa as seguintes etapas:

    1) Normalização das variáveis explicativas
    2) Busca dos melhores hiperparâmetros com GridSearchCV
    3) Treinamento do melhor modelo encontrado
    4) Avaliação IN-SAMPLE (dados de treino)
    5) Previsão OUT-OF-SAMPLE (dados de teste)

    ----------------------------------------------------------------

    Parâmetros de entrada:

    file
        Nome do ativo sendo analisado (PETR4, ITUB4, VALE3).

    x_treino
        Variáveis explicativas do conjunto de treinamento.

    y_treino
        Variável alvo (target) do conjunto de treinamento.

    x_teste
        Observação futura utilizada para previsão.

    y_teste_real
        Resultado real observado no período de teste.

    parametros_rna / parametros_rf / parametros_svc
        Dicionários contendo os hiperparâmetros utilizados
        na busca via GridSearchCV.

    ----------------------------------------------------------------

    Retorno:

    Um dicionário contendo:

    • previsões out-of-sample
    • valores reais
    • previsões in-sample
    • valores reais do treino
    • precision in-sample
    """

    # ESTRUTURA PARA ARMAZENAR RESULTADOS
    resultados = {
        'RNA': {'previsoes': [], 'y_real': [], 'y_treino': [], 'y_in': [], 'precision_in': None},
        'Random Forest': {'previsoes': [], 'y_real': [], 'y_treino': [], 'y_in': [], 'precision_in': None},
        'SVC': {'previsoes': [], 'y_real': [], 'y_treino': [], 'y_in': [], 'precision_in': None}
    }

    # Cada modelo armazenará:
    #
    # previsoes → previsão feita para o dado de teste
    # y_real → valor real observado
    # y_in → previsões feitas nos dados de treino
    # y_treino → valores reais do treino
    # precision_in → precisão do modelo no treino

    # Normalização dos dados
    scaler_dados = StandardScaler()

    # Ajusta o scaler usando apenas os dados de treino
    x_treino_normalizado = scaler_dados.fit_transform(x_treino)

    # Aplica a mesma transformação nos dados de teste
    x_teste_normalizado = scaler_dados.transform(x_teste)

    # ==================================================
    # MODELO 1 - REDE NEURAL
    # ==================================================

    if len(np.unique(y_treino)) < 2:

        print("⚠️ RNA ignorada: apenas uma classe presente nos dados.")

    else:

        rna = MLPClassifier(max_iter=300)

        # Busca dos melhores hiperparâmetros
        grid_rna = GridSearchCV(
            rna,
            parametros_rna,
            cv=3,
            scoring='accuracy'
        )

        # Treina todos os modelos do grid
        grid_rna.fit(x_treino_normalizado, y_treino)

        # Seleciona o melhor modelo encontrado
        melhor_rna = grid_rna.best_estimator_

        # Treina novamente o melhor modelo
        melhor_rna.fit(x_treino_normalizado, y_treino)

        # AVALIAÇÃO IN-SAMPLE
        # Previsões feitas sobre os próprios dados de treino
        y_in_rna = melhor_rna.predict(x_treino_normalizado)

        # Precision mede: entre todas as previsões positivas feitas pelo modelo, quantas realmente estavam corretas.
        precision_in_rna = precision_score(y_treino, y_in_rna, zero_division=0)

        resultados['RNA']['y_in'] = list(y_in_rna)
        resultados['RNA']['y_treino'] = list(y_treino)
        resultados['RNA']['precision_in'] = precision_in_rna

        # PREVISÃO OUT-OF-SAMPLE

        # O modelo recebe uma observação nova e prevê se o target será atingido ou não.
        y_previsao_rna = melhor_rna.predict(x_teste_normalizado)

        resultados['RNA']['previsoes'].append(int(y_previsao_rna[0]))
        resultados['RNA']['y_real'].append(int(y_teste_real))

    # ==================================================
    # MODELO 2 - RANDOM FOREST
    # ==================================================

    if len(np.unique(y_treino)) < 2:

        print("⚠️ Random Forest ignorada.")

    else:

        rf = RandomForestClassifier(random_state=42)

        grid_rf = GridSearchCV(
            rf,
            parametros_rf,
            cv=3,
            scoring='accuracy'
        )

        grid_rf.fit(x_treino_normalizado, y_treino)

        melhor_rf = grid_rf.best_estimator_

        melhor_rf.fit(x_treino_normalizado, y_treino)

        y_in_rf = melhor_rf.predict(x_treino_normalizado)

        precision_in_rf = precision_score(y_treino, y_in_rf, zero_division=0)

        resultados['Random Forest']['y_in'] = list(y_in_rf)
        resultados['Random Forest']['y_treino'] = list(y_treino)
        resultados['Random Forest']['precision_in'] = precision_in_rf

        y_previsao_rf = melhor_rf.predict(x_teste_normalizado)

        resultados['Random Forest']['previsoes'].append(int(y_previsao_rf[0]))
        resultados['Random Forest']['y_real'].append(int(y_teste_real))

    # ==================================================
    # MODELO 3 - SVC
    # ==================================================

    if len(np.unique(y_treino)) < 2:

        print("⚠️ SVC ignorado.")

    else:

        svc = SVC(random_state=42)

        grid_svc = GridSearchCV(
            svc,
            parametros_svc,
            cv=3,
            scoring='accuracy'
        )

        grid_svc.fit(x_treino_normalizado, y_treino)

        melhor_svc = grid_svc.best_estimator_

        melhor_svc.fit(x_treino_normalizado, y_treino)

        y_in_svc = melhor_svc.predict(x_treino_normalizado)

        precision_in_svc = precision_score(y_treino, y_in_svc, zero_division=0)

        resultados['SVC']['y_in'] = list(y_in_svc)
        resultados['SVC']['y_treino'] = list(y_treino)
        resultados['SVC']['precision_in'] = precision_in_svc

        y_previsao_svc = melhor_svc.predict(x_teste_normalizado)

        resultados['SVC']['previsoes'].append(int(y_previsao_svc[0]))
        resultados['SVC']['y_real'].append(int(y_teste_real))

    return resultados