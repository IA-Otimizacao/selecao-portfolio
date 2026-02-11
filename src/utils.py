import pandas as pd
import numpy as np
import talib
import warnings
import time
import os
import locale
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from feature_engine.selection import DropCorrelatedFeatures


# ==============================
# DECORATOR DE CRONOMETRAGEM
# ==============================
def medir_tempo(func):
    """Decorator para medir o tempo de execução de funções."""
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = func(*args, **kwargs)
        fim = time.time()
        duracao = fim - inicio
        print(f"\033[92m⏱ Função '{func.__name__}' executada em {duracao:.2f} segundos.\033[0m")
        return resultado
    return wrapper


# ==============================
# FUNÇÕES PRINCIPAIS
# ==============================

def carregar_dados(file):
    try:
        locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
    except:
        pass
    base_dados = pd.read_csv(f'./data/raw/refinitiv/{file}.csv')
    base_dados['Exchange Date'] = base_dados['Exchange Date'].str.replace('.', '', regex=False).str.strip()
    base_dados['Exchange Date'] = pd.to_datetime(
        base_dados['Exchange Date'],
        format='%d-%b-%Y',
        errors='coerce'
    )
    return base_dados


def carregar_ibov(ibov):
    ibov_status = pd.read_excel(f'./data/ibov/{ibov}.xlsx')
    ibov_status['Exchange Date'] = pd.to_datetime(ibov_status['Exchange Date'])
    return ibov_status


def padronizar_colunas(base_dados):
    for col in base_dados.columns:
        if base_dados[col].dtype == 'object':
            base_dados[col] = base_dados[col].str.replace('.', '')
            base_dados[col] = base_dados[col].str.replace(',', '.')
        if col != 'Exchange Date':
            base_dados[col] = pd.to_numeric(base_dados[col], errors='coerce')
    return base_dados


def remover_linhas_invalidas(base_dados):
    colunas_verificacao = ['Close', 'Open', 'Low', 'High', 'Volume']
    base_dados = base_dados.dropna(subset=colunas_verificacao)
    for col in colunas_verificacao:
        base_dados = base_dados.loc[base_dados[col] > 0]
    return base_dados

def garantir_numerico(base_dados, colunas):
    for col in colunas:
        if col in base_dados.columns:
            base_dados[col] = pd.to_numeric(base_dados[col], errors='coerce')
        else:
            print(f"Aviso: coluna '{col}' não encontrada no DataFrame.")
    return base_dados

def calcular_variaveis(base_dados):
    base_dados['r1'] = np.log(base_dados['Close'] / base_dados['Close'].shift(1))
    base_dados['r2'] = np.log(base_dados['Close'].shift(1) / base_dados['Close'].shift(2))
    base_dados['r3'] = np.log(base_dados['Close'].shift(2) / base_dados['Close'].shift(3))
    base_dados['r4'] = np.log(base_dados['Close'].shift(3) / base_dados['Close'].shift(4))
    base_dados['r5'] = np.log(base_dados['High'] / base_dados['Open'])
    base_dados['r6'] = np.log(base_dados['High'] / base_dados['Open'].shift(1))
    base_dados['r7'] = np.log(base_dados['High'] / base_dados['Open'].shift(2))
    base_dados['r8'] = np.log(base_dados['High'] / base_dados['Open'].shift(3))
    base_dados['r9'] = np.log(base_dados['High'].shift(1) / base_dados['Open'].shift(1))
    base_dados['r10'] = np.log(base_dados['High'].shift(2) / base_dados['Open'].shift(2))
    base_dados['r11'] = np.log(base_dados['High'].shift(3) / base_dados['Open'].shift(3))
    base_dados['r12'] = np.log(base_dados['Low'] / base_dados['Open'])
    base_dados['r13'] = np.log(base_dados['Low'].shift(1) / base_dados['Open'].shift(1))
    base_dados['r14'] = np.log(base_dados['Low'].shift(2) / base_dados['Open'].shift(2))
    base_dados['r15'] = np.log(base_dados['Low'].shift(3) / base_dados['Open'].shift(3))
    base_dados['Momentum'] = talib.MOM(base_dados['Close'], timeperiod=10)
    base_dados['RSI'] = talib.RSI(base_dados['Close'], timeperiod=14)
    base_dados['Parabolic_SAR'] = talib.SAR(base_dados['High'], base_dados['Low'], acceleration=0, maximum=0)
    base_dados['ATR'] = talib.ATR(base_dados['High'], base_dados['Low'], base_dados['Close'], timeperiod=14)
    base_dados['True_Range'] = talib.TRANGE(base_dados['High'], base_dados['Low'], base_dados['Close'])
    base_dados['Chaikin_AD'] = talib.AD(base_dados['High'], base_dados['Low'], base_dados['Close'], base_dados['Volume'])
    base_dados['OBV'] = talib.OBV(base_dados['Close'], base_dados['Volume'])
    return base_dados


def x_split(base_dados):
    colunas_de_interesse = [
        'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12',
        'r13', 'r14', 'r15', 'Momentum', 'RSI', 'Parabolic_SAR', 'ATR', 'True_Range',
        'Chaikin_AD', 'OBV'
    ]
    return base_dados[colunas_de_interesse].copy()


def y_split(base_dados):
    return base_dados['target'].values


def z_split(base_dados):
    return base_dados[['Exchange Date', 'resultado_real', 'target']].copy()


@medir_tempo
def features_selection(x_dados, correlation_threshold=0.8):
    colunas_de_interesse = [
        'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12',
        'r13', 'r14', 'r15', 'Momentum', 'RSI', 'Parabolic_SAR', 'ATR', 'True_Range',
        'Chaikin_AD', 'OBV'
    ]
    fs_qconst = VarianceThreshold(threshold=0.01)
    x_dados_filtrado = fs_qconst.fit_transform(x_dados)
    colunas_selecionadas = np.array(colunas_de_interesse)[fs_qconst.get_support()]
    x_dados_filtrado = pd.DataFrame(x_dados_filtrado, columns=colunas_selecionadas, index=x_dados.index)

    sel = DropCorrelatedFeatures(threshold=correlation_threshold, method='pearson', missing_values='ignore')
    return sel.fit_transform(x_dados_filtrado)



def calcular_target(base_dados, target, janela_temporal):
    base_dados = base_dados.reset_index(drop=True)
    base_dados['compra'] = base_dados['Open'].shift(-1)
    base_dados['maximo'] = base_dados['High'].rolling(window=janela_temporal).max().shift(-janela_temporal)
    base_dados['venda'] = base_dados['Close'].shift(-janela_temporal)
    base_dados['target'] = 0
    for i in range(len(base_dados) - janela_temporal):
        compra = base_dados.at[i, 'compra']
        for j in range(1, janela_temporal + 1):
            if base_dados.at[i + j, 'High'] >= compra * target:
                base_dados.at[i, 'venda'] = compra * target
                base_dados.at[i, 'target'] = 1
                break
    base_dados['resultado_real'] = (base_dados['venda'] / base_dados['compra']) - 1
    return base_dados


@medir_tempo
def treinar_e_avaliar(file, x_treino, y_treino, x_teste, y_teste_real,
                      parametros_rna, parametros_rf, parametros_svc):
    """
    Função principal de treino e avaliação dos modelos RNA, Random Forest e SVC.
    Retorna previsões out-of-sample, além de resultados in-sample (y_in, y_treino e precision_in).
    """

    resultados = {
        'RNA': {'previsoes': [], 'y_real': [],'y_treino': [], 'y_in': [], 'precision_in': None},
        'Random Forest': {'previsoes': [], 'y_real': [], 'y_treino': [],'y_in': [],  'precision_in': None},
        'SVC': {'previsoes': [], 'y_real': [], 'y_treino': [], 'y_in': [], 'precision_in': None}
    }

    scaler_dados = StandardScaler()
    x_treino_normalizado = scaler_dados.fit_transform(x_treino)
    x_teste_normalizado = scaler_dados.transform(x_teste)

    # ========================
    # RNA
    # ========================
    if len(np.unique(y_treino)) < 2:
        print("⚠️ RNA ignorada: apenas uma classe presente nos dados de treino.")
    else:
        print(f"\n🔹 Iniciando GridSearchCV para RNA - {file}...")
        inicio_grid = time.time()
        rna = MLPClassifier(max_iter=300, verbose=False, tol=0.001)
        grid_rna = GridSearchCV(rna, parametros_rna, cv=3, scoring='accuracy')
        grid_rna.fit(x_treino_normalizado, y_treino)
        fim_grid = time.time()
        print(f"⏱ Tempo total do GridSearchCV (RNA - {file}): {fim_grid - inicio_grid:.2f} segundos")

        melhor_rna = grid_rna.best_estimator_
        print(f"🚀 Iniciando treino do melhor modelo RNA - {file}...")
        inicio_treino = time.time()
        melhor_rna.fit(x_treino_normalizado, y_treino)
        fim_treino = time.time()
        print(f"⏱ Tempo de treino final (RNA- {file}): {fim_treino - inicio_treino:.2f} segundos")

        # ---- In-sample (RNA)
        y_in_rna = melhor_rna.predict(x_treino_normalizado)
        precision_in_rna = precision_score(y_treino, y_in_rna, zero_division=0)

        resultados['RNA']['y_in'] = list(y_in_rna)
        resultados['RNA']['y_treino'] = list(y_treino)
        resultados['RNA']['precision_in'] = precision_in_rna

        # ---- Out-of-sample
        y_previsao_rna = melhor_rna.predict(x_teste_normalizado)
        resultados['RNA']['previsoes'].append(int(y_previsao_rna[0]))
        resultados['RNA']['y_real'].append(int(y_teste_real))

    # ========================
    # Random Forest
    # ========================
    if len(np.unique(y_treino)) < 2:
        print("⚠️ Random Forest ignorada: apenas uma classe presente nos dados de treino.")
    else:
        print(f"\n🌲 Iniciando GridSearchCV para Random Forest - {file}...")
        inicio_grid = time.time()
        rf = RandomForestClassifier(random_state=42)
        grid_rf = GridSearchCV(rf, parametros_rf, cv=3, scoring='accuracy')
        grid_rf.fit(x_treino_normalizado, y_treino)
        fim_grid = time.time()
        print(f"⏱ Tempo total do GridSearchCV (Random Forest - {file}): {fim_grid - inicio_grid:.2f} segundos")

        melhor_rf = grid_rf.best_estimator_
        print(f"🚀 Iniciando treino do melhor modelo Random Forest - {file}...")
        inicio_treino = time.time()
        melhor_rf.fit(x_treino_normalizado, y_treino)
        fim_treino = time.time()
        print(f"⏱ Tempo de treino final (Random Forest): {fim_treino - inicio_treino:.2f} segundos")

        # ---- In-sample (RF)
        y_in_rf = melhor_rf.predict(x_treino_normalizado)
        precision_in_rf = precision_score(y_treino, y_in_rf, zero_division=0)

        resultados['Random Forest']['y_in'] = list(y_in_rf)
        resultados['Random Forest']['y_treino'] = list(y_treino)
        resultados['Random Forest']['precision_in'] = precision_in_rf

        # ---- Out-of-sample
        y_previsao_rf = melhor_rf.predict(x_teste_normalizado)
        resultados['Random Forest']['previsoes'].append(int(y_previsao_rf[0]))
        resultados['Random Forest']['y_real'].append(int(y_teste_real))

    # ========================
    # SVC
    # ========================
    if len(np.unique(y_treino)) < 2:
        print("⚠️ SVC ignorado: apenas uma classe presente nos dados de treino.")
    else:
        print(f"\n⚙️ Iniciando GridSearchCV para SVC - {file}...")
        inicio_grid = time.time()
        svc = SVC(probability=False, random_state=42)
        grid_svc = GridSearchCV(svc, parametros_svc, cv=3, scoring='accuracy')
        grid_svc.fit(x_treino_normalizado, y_treino)
        fim_grid = time.time()
        print(f"⏱ Tempo total do GridSearchCV (SVC - {file}): {fim_grid - inicio_grid:.2f} segundos")

        melhor_svc = grid_svc.best_estimator_
        print(f"🚀 Iniciando treino do melhor modelo SVC - {file}...")
        inicio_treino = time.time()
        melhor_svc.fit(x_treino_normalizado, y_treino)
        fim_treino = time.time()
        print(f"⏱ Tempo de treino final (SVC - {file}): {fim_treino - inicio_treino:.2f} segundos")

        # ---- In-sample (SVC)
        y_in_svc = melhor_svc.predict(x_treino_normalizado)
        precision_in_svc = precision_score(y_treino, y_in_svc, zero_division=0)

        resultados['SVC']['y_in'] = list(y_in_svc)
        resultados['SVC']['y_treino'] = list(y_treino)
        resultados['SVC']['precision_in'] = precision_in_svc

        # ---- Out-of-sample
        y_previsao_svc = melhor_svc.predict(x_teste_normalizado)
        resultados['SVC']['previsoes'].append(int(y_previsao_svc[0]))
        resultados['SVC']['y_real'].append(int(y_teste_real))

    return resultados
