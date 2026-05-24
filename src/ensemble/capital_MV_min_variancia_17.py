import os
import re

import pandas as pd


# PASSO A PASSO DO SCRIPT
# 1. Le os pesos diarios gerados pela otimizacao MV minima variancia da etapa 15.
# 2. Le o arquivo equivalente da etapa 11 para obter sinais e retornos por ativo.
# 3. Junta pesos e retornos pela coluna data.
# 4. Em cada data, identifica os ativos livres, ou seja, sem posicao aberta.
# 5. Usa os pesos de minima variancia apenas entre os ativos livres e renormaliza
#    a soma.
# 6. Aloca o capital disponivel nas novas posicoes.
# 7. Atualiza posicoes abertas: fecha com target se bateu o retorno alvo.
# 8. Se nao bateu o target ate o limite de dias, vende pelo retorno de fechamento.
# 9. Soma capital disponivel + capital retido para calcular o patrimonio diario.
# 10. Salva o historico de capital e os detalhes por ativo.

# Tecnicas que aparecem no nome dos arquivos vindos da etapa 11.
TECNICAS = ["esmble_jan_tot", "esmble_jan_par", "in_precision", "aleatorio"]

# Margem pequena para comparar valores muito proximos de zero.
EPSILON = 1e-12


def extrair_target(nome_arquivo):
    # Exemplo: esmble_jan_par_target_1_01.csv -> 1.01
    match = re.search(r"target_(\d+[_\.]\d+)", nome_arquivo)

    if not match:
        raise ValueError(f"Target nao encontrado em: {nome_arquivo}")

    return float(match.group(1).replace("_", "."))


def extrair_tecnica(nome_arquivo):
    # Identifica qual tecnica de ensemble esta sendo processada pelo nome.
    tecnica = next((valor for valor in TECNICAS if valor in nome_arquivo), None)

    if tecnica is None:
        raise ValueError(f"Tecnica nao encontrada em: {nome_arquivo}")

    return tecnica


def identificar_ativos_pesos(df):
    # Procura colunas como peso_PETR4_min_variancia e devolve PETR4.
    ativos = []

    for col in df.columns:
        match = re.match(r"peso_(.+)_min_variancia$", col)

        if match:
            ativos.append(match.group(1))

    return ativos


def valor_float(valor, padrao=0.0):
    # Padroniza valores numericos e troca NaN/erro por zero.
    if pd.isna(valor):
        return padrao

    try:
        return float(valor)
    except (TypeError, ValueError):
        return padrao


def normalizar_pesos(pesos):
    # Garante que os pesos usados na alocacao sejam positivos e somem 1.
    # Isso e necessario porque ativos ja retidos sao removidos antes da compra.
    pesos_limpos = {
        ativo: max(0.0, valor_float(peso))
        for ativo, peso in pesos.items()
    }
    soma = sum(pesos_limpos.values())

    if soma <= EPSILON:
        return {}

    return {
        ativo: peso / soma
        for ativo, peso in pesos_limpos.items()
        if peso > EPSILON
    }


def preparar_dados(caminho_pesos, caminho_targets):
    # Le os pesos do MV Min Variancia e os retornos/targets da tecnica correspondente.
    df_pesos = pd.read_csv(caminho_pesos)
    df_targets = pd.read_csv(caminho_targets)

    # Converte a data para permitir merge correto entre as duas bases.
    df_pesos["data"] = pd.to_datetime(df_pesos["data"], errors="coerce")
    df_targets["data"] = pd.to_datetime(df_targets["data"], errors="coerce")

    # Remove linhas sem data valida, porque elas nao conseguem ser alinhadas.
    df_pesos = df_pesos.dropna(subset=["data"])
    df_targets = df_targets.dropna(subset=["data"])

    # Junta pesos e retornos na mesma linha de data.
    df = df_pesos.merge(df_targets, on="data", how="left")
    df = df.sort_values("data").reset_index(drop=True)

    return df


def processar_arquivo(
    caminho_pesos,
    targets_folder,
    output_folder,
    capital_inicial=100.0,
    dias_max_posicao=4
):
    # Simula a carteira de um arquivo especifico de tecnica/target.
    # Cada arquivo representa uma tecnica + target, por exemplo:
    # esmble_jan_par_target_1_01.csv.
    nome_arquivo = os.path.basename(caminho_pesos)
    caminho_targets = os.path.join(targets_folder, nome_arquivo)

    # O arquivo de pesos precisa ter um arquivo equivalente de retornos.
    if not os.path.exists(caminho_targets):
        print(f"Pulando {nome_arquivo}: arquivo de targets nao encontrado.")
        return

    print(f"\nDistribuindo capital MV Min Variancia: {nome_arquivo}")

    # Target vira o limite de ganho da operacao.
    # Exemplo: target 1.01 -> threshold 0.01.
    target = extrair_target(nome_arquivo)
    threshold = target - 1
    tecnica = extrair_tecnica(nome_arquivo)

    # Base final de trabalho: data + pesos + retornos por ativo.
    df = preparar_dados(caminho_pesos, caminho_targets)
    ativos_pesos = identificar_ativos_pesos(df)

    # So da para simular capital nos ativos que possuem as colunas de retorno.
    ativos_com_retorno = [
        ativo
        for ativo in ativos_pesos
        if (
            f"{ativo}_rend_decisao_{tecnica}" in df.columns
            and f"{ativo}_rend_venda_{tecnica}" in df.columns
        )
    ]

    # Mantem o aviso quando existe peso, mas falta retorno para calcular o fim.
    ativos_ignorados = sorted(set(ativos_pesos) - set(ativos_com_retorno))
    if ativos_ignorados:
        print(
            "  Ativos ignorados sem colunas de retorno: "
            f"{', '.join(ativos_ignorados)}"
        )

    # DataFrame que sera salvo no final.
    # Ele comeca com data e metadados dos pesos calculados na etapa 15.
    saida = df[["data"]].copy()

    if "status_min_variancia" in df.columns:
        saida["status_min_variancia"] = df["status_min_variancia"]

    if "ativos_min_variancia" in df.columns:
        saida["ativos_min_variancia"] = df["ativos_min_variancia"]

    # Colunas gerais da carteira.
    # capital_disponivel_*: dinheiro livre para novas entradas.
    # capital_retido: dinheiro preso em posicoes ainda abertas.
    # capital_total: disponivel + retido.
    saida["capital_disponivel_inicio_min_variancia"] = 0.0
    saida["capital_alocado_novo_min_variancia"] = 0.0
    saida["capital_disponivel_fim_min_variancia"] = 0.0
    saida["capital_retido_min_variancia"] = 0.0
    saida["capital_total_min_variancia"] = 0.0
    saida["retorno_total_min_variancia"] = 0.0
    saida["soma_pesos_original_min_variancia"] = 0.0
    saida["soma_pesos_alocacao_min_variancia"] = 0.0

    # Cria colunas individuais para cada ativo que veio na base de pesos.
    for ativo in ativos_pesos:
        # peso_*: peso original calculado pelo Markowitz.
        # peso_alocacao_*: peso recalculado so entre ativos livres para compra.
        saida[f"peso_{ativo}_min_variancia"] = 0.0
        saida[f"peso_alocacao_{ativo}_min_variancia"] = 0.0

        # inicio/fim: capital daquela posicao no inicio e no fim do dia.
        # retido: capital que continua preso para o proximo dia.
        # retorno: retorno aplicado quando a posicao fecha.
        saida[f"inicio_{ativo}_min_variancia"] = 0.0
        saida[f"fim_{ativo}_min_variancia"] = 0.0
        saida[f"retido_{ativo}_min_variancia"] = 0.0
        saida[f"retorno_{ativo}_min_variancia"] = 0.0
        saida[f"dias_{ativo}_min_variancia"] = 0

        # Ativos sem coluna de retorno ficam no output, mas nao entram na conta.
        if ativo in ativos_com_retorno:
            saida[f"status_{ativo}_min_variancia"] = "sem_posicao"
        else:
            saida[f"status_{ativo}_min_variancia"] = "sem_retorno"

    # Estado das posicoes abertas ao longo do tempo.
    # capital: valor ainda investido naquele ativo.
    # dias: ha quantos dias a posicao esta aberta.
    posicoes = {
        ativo: {
            "capital": 0.0,
            "dias": 0
        }
        for ativo in ativos_com_retorno
    }

    capital_disponivel = float(capital_inicial)
    capital_total_anterior = float(capital_inicial)

    # Loop principal: cada linha representa um dia de simulacao.
    for i, row in df.iterrows():
        # Guarda quanto dinheiro livre existia no comeco do dia.
        capital_inicio = capital_disponivel

        # Le os pesos originais da etapa de otimizacao MV Min Variancia.
        pesos_originais = {
            ativo: valor_float(row.get(f"peso_{ativo}_min_variancia", 0.0))
            for ativo in ativos_pesos
        }
        soma_pesos_original = sum(
            max(0.0, peso)
            for peso in pesos_originais.values()
        )

        # So pode receber alocacao nova o ativo que nao tem posicao aberta.
        ativos_livres = [
            ativo
            for ativo in ativos_com_retorno
            if posicoes[ativo]["capital"] <= EPSILON
        ]

        # Se algum ativo ja esta retido, os pesos sao renormalizados apenas
        # entre os ativos livres. Assim nao compramos duas vezes o mesmo ativo.
        pesos_para_alocar = normalizar_pesos({
            ativo: pesos_originais[ativo]
            for ativo in ativos_livres
        })

        capital_alocado_novo = 0.0

        # Abre novas posicoes usando o capital disponivel no inicio do dia.
        for ativo, peso_alocacao in pesos_para_alocar.items():
            valor_alocado = capital_inicio * peso_alocacao

            posicoes[ativo]["capital"] = valor_alocado
            posicoes[ativo]["dias"] = 1
            capital_alocado_novo += valor_alocado

            saida.loc[i, f"peso_alocacao_{ativo}_min_variancia"] = peso_alocacao

        # O dinheiro usado em novas compras deixa de estar disponivel.
        capital_disponivel -= capital_alocado_novo

        # Atualiza cada posicao aberta, verificando se fecha ou continua retida.
        for ativo in ativos_pesos:
            peso_original = max(0.0, pesos_originais[ativo])
            saida.loc[i, f"peso_{ativo}_min_variancia"] = peso_original

            # Sem retorno nao da para calcular venda, entao nao participa.
            if ativo not in ativos_com_retorno:
                continue

            capital_posicao = posicoes[ativo]["capital"]

            # Se nao ha posicao aberta no ativo, nada a atualizar.
            if capital_posicao <= EPSILON:
                continue

            dias_posicao = posicoes[ativo]["dias"]
            col_rend_decisao = f"{ativo}_rend_decisao_{tecnica}"
            col_rend_venda = f"{ativo}_rend_venda_{tecnica}"

            # rend_decisao verifica se bateu o target no dia.
            # rend_venda e usado quando nao bate target e precisa vender no limite.
            rend_decisao = valor_float(row.get(col_rend_decisao, 0.0))
            rend_venda = valor_float(row.get(col_rend_venda, 0.0))

            # Por padrao, a posicao continua aberta sem retorno realizado.
            capital_fim = capital_posicao
            retorno_ativo = 0.0
            status_posicao = "mantido"
            manter_posicao = True

            # Se bateu o target, fecha a posicao com retorno alvo.
            if rend_decisao >= threshold:
                retorno_ativo = threshold
                capital_fim = capital_posicao * target
                status_posicao = "target"
                manter_posicao = False

            # Se nao bateu o target ate o limite de dias, vende pelo retorno
            # de fechamento calculado na etapa anterior do ensemble.
            elif dias_posicao >= dias_max_posicao:
                retorno_ativo = rend_venda
                capital_fim = capital_posicao * (1 + rend_venda)
                status_posicao = "venda_limite_dias"
                manter_posicao = False

            # Registra no output o estado do ativo naquele dia.
            saida.loc[i, f"inicio_{ativo}_min_variancia"] = capital_posicao
            saida.loc[i, f"fim_{ativo}_min_variancia"] = capital_fim
            saida.loc[i, f"retorno_{ativo}_min_variancia"] = retorno_ativo
            saida.loc[i, f"dias_{ativo}_min_variancia"] = dias_posicao
            saida.loc[i, f"status_{ativo}_min_variancia"] = status_posicao

            if manter_posicao:
                # Posicao segue aberta: o capital fica retido para o proximo dia.
                saida.loc[i, f"retido_{ativo}_min_variancia"] = capital_fim
                posicoes[ativo]["capital"] = capital_fim
                posicoes[ativo]["dias"] = dias_posicao + 1
            else:
                # Posicao fechou: capital volta a ficar disponivel.
                capital_disponivel += capital_fim
                posicoes[ativo]["capital"] = 0.0
                posicoes[ativo]["dias"] = 0

        # Soma quanto capital ainda esta preso em posicoes abertas.
        capital_retido = sum(
            posicao["capital"]
            for posicao in posicoes.values()
        )

        # Patrimonio total da carteira ao fim do dia.
        capital_total = capital_disponivel + capital_retido

        # Retorno diario da carteira em relacao ao patrimonio do dia anterior.
        if capital_total_anterior > EPSILON:
            retorno_total = (capital_total / capital_total_anterior) - 1
        else:
            retorno_total = 0.0

        # Salva os indicadores gerais da carteira no dia atual.
        saida.loc[i, "capital_disponivel_inicio_min_variancia"] = capital_inicio
        saida.loc[i, "capital_alocado_novo_min_variancia"] = capital_alocado_novo
        saida.loc[i, "capital_disponivel_fim_min_variancia"] = capital_disponivel
        saida.loc[i, "capital_retido_min_variancia"] = capital_retido
        saida.loc[i, "capital_total_min_variancia"] = capital_total
        saida.loc[i, "retorno_total_min_variancia"] = retorno_total
        saida.loc[i, "soma_pesos_original_min_variancia"] = soma_pesos_original
        saida.loc[i, "soma_pesos_alocacao_min_variancia"] = sum(
            pesos_para_alocar.values()
        )

        # Atualiza a referencia para calcular o retorno do proximo dia.
        capital_total_anterior = capital_total

        # Log simples para acompanhar arquivos longos.
        if i % 250 == 0 or i == len(df) - 1:
            print(
                f"  andamento {nome_arquivo}: {i + 1}/{len(df)} | "
                f"capital {capital_total:.2f}"
            )

    # Salva o CSV final da distribuicao de capital para esta tecnica/target.
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, nome_arquivo)
    saida.to_csv(output_path, index=False)

    print(f"Salvo: {output_path}")


def run_capital_mv_min_variancia(
    weights_folder="./data/ensemble/15_mv_min_variancia/",
    targets_folder="./data/ensemble/11_targets_por_tecnica/",
    output_folder="./data/ensemble/17_capital_mv_min_variancia/",
    capital_inicial=100.0,
    dias_max_posicao=4
):
    # Executa a etapa 17 para todos os arquivos de pesos de minima variancia.
    # Pasta de saida fica separada para nao mexer na metodologia 1/n.
    os.makedirs(output_folder, exist_ok=True)

    # Processa todos os CSVs de pesos gerados pelo MV Min Variancia.
    arquivos = sorted([
        arquivo
        for arquivo in os.listdir(weights_folder)
        if arquivo.endswith(".csv")
    ])

    for arquivo in arquivos:
        # Cada arquivo da pasta 15 precisa de um arquivo equivalente na pasta 11.
        processar_arquivo(
            os.path.join(weights_folder, arquivo),
            targets_folder=targets_folder,
            output_folder=output_folder,
            capital_inicial=capital_inicial,
            dias_max_posicao=dias_max_posicao
        )

    print("\nDistribuicao de capital MV Min Variancia concluida.")


if __name__ == "__main__":
    run_capital_mv_min_variancia()
