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

# ================= CACHE GRIDSEARCH =================
melhores_params_cache = {}
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
def carregar_dados(file, base_path="./data/pre_process/raw/refinitiv"):
    
    try:
        locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
    except:
        pass

    # Se já vier com .csv, não adiciona
    if file.endswith(".csv"):
        path = file
    else:
        path = os.path.join(base_path, f"{file}.csv")

    base_dados = pd.read_csv(path)

    base_dados['Exchange Date'] = base_dados['Exchange Date'] \
        .str.replace('.', '', regex=False).str.strip()

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
    ibov_status = pd.read_excel(f'./data/pre_process/ibov/{ibov}.xlsx')

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
                      parametros_rna, parametros_rf, parametros_svc,
                      ano_atual=None, janela=None):

    resultados = {
        'RNA': {'previsoes': [], 'y_real': [], 'y_treino': [], 'y_in': [], 'precision_in': None},
        'Random Forest': {'previsoes': [], 'y_real': [], 'y_treino': [], 'y_in': [], 'precision_in': None},
        'SVC': {'previsoes': [], 'y_real': [], 'y_treino': [], 'y_in': [], 'precision_in': None}
    }

    scaler = StandardScaler()
    x_treino = scaler.fit_transform(x_treino)
    x_teste = scaler.transform(x_teste)

    # ================= RNA =================
    chave = (file, 'RNA', ano_atual, janela)

    if len(np.unique(y_treino)) >= 2:

        if chave not in melhores_params_cache:

            print(f"🔍 GridSearch RNA ({ano_atual})")

            model = MLPClassifier(max_iter=300)

            grid = GridSearchCV(model, parametros_rna, cv=3, scoring='accuracy')
            grid.fit(x_treino, y_treino)

            melhores_params_cache[chave] = grid.best_estimator_.get_params()

        model = MLPClassifier(**melhores_params_cache[chave])
        model.fit(x_treino, y_treino)

        y_in = model.predict(x_treino)
        resultados['RNA']['y_in'] = list(y_in)
        resultados['RNA']['y_treino'] = list(y_treino)
        resultados['RNA']['precision_in'] = precision_score(y_treino, y_in, zero_division=0)

        pred = model.predict(x_teste)
        resultados['RNA']['previsoes'].append(int(pred[0]))
        resultados['RNA']['y_real'].append(int(y_teste_real))

    # ================= RF =================
    chave = (file, 'RF', ano_atual, janela)

    if len(np.unique(y_treino)) >= 2:

        if chave not in melhores_params_cache:

            print(f"🔍 GridSearch RF ({ano_atual})")

            model = RandomForestClassifier(random_state=42)

            grid = GridSearchCV(model, parametros_rf, cv=3, scoring='accuracy')
            grid.fit(x_treino, y_treino)

            melhores_params_cache[chave] = grid.best_estimator_.get_params()

        model = RandomForestClassifier(**melhores_params_cache[chave])
        model.fit(x_treino, y_treino)

        y_in = model.predict(x_treino)
        resultados['Random Forest']['y_in'] = list(y_in)
        resultados['Random Forest']['y_treino'] = list(y_treino)
        resultados['Random Forest']['precision_in'] = precision_score(y_treino, y_in, zero_division=0)

        pred = model.predict(x_teste)
        resultados['Random Forest']['previsoes'].append(int(pred[0]))
        resultados['Random Forest']['y_real'].append(int(y_teste_real))

    # ================= SVC =================
    chave = (file, 'SVC', ano_atual, janela)

    if len(np.unique(y_treino)) >= 2:

        if chave not in melhores_params_cache:

            print(f"🔍 GridSearch SVC ({ano_atual})")

            model = SVC()

            grid = GridSearchCV(model, parametros_svc, cv=3, scoring='accuracy')
            grid.fit(x_treino, y_treino)

            melhores_params_cache[chave] = grid.best_estimator_.get_params()

        model = SVC(**melhores_params_cache[chave])
        model.fit(x_treino, y_treino)

        y_in = model.predict(x_treino)
        resultados['SVC']['y_in'] = list(y_in)
        resultados['SVC']['y_treino'] = list(y_treino)
        resultados['SVC']['precision_in'] = precision_score(y_treino, y_in, zero_division=0)

        pred = model.predict(x_teste)
        resultados['SVC']['previsoes'].append(int(pred[0]))
        resultados['SVC']['y_real'].append(int(y_teste_real))

    return resultados