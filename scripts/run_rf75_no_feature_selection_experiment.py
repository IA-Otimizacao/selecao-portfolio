import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import x_split, y_split, z_split  # noqa: E402


TARGET = 1.02
JANELA = 75
TECNICA_TRAIN = "Random Forest"
COLUNA_TECNICA = "RandomForest_75"
EXPERIMENTO_NOME = "randomforest_75_target_1_02_sem_fs"

PARAMETROS_RF = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}


def garantir_pasta(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def listar_ativos(curated_dir, ativos_especificos=None):
    ativos = sorted(
        {
            caminho.name.split("_target_")[0]
            for caminho in curated_dir.glob("*_target_*.csv")
        }
    )

    if not ativos_especificos:
        return ativos

    ativos_normalizados = {ativo.upper() for ativo in ativos_especificos}
    return [ativo for ativo in ativos if ativo.upper() in ativos_normalizados]


def carregar_json(path, padrao):
    if not path.exists():
        return padrao

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def salvar_json(path, conteudo):
    garantir_pasta(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(conteudo, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def params_key(ativo, ano_atual, janela):
    return f"{ativo}|RF_NO_FS|{ano_atual}|{janela}"


def datas_esperadas_para_ativo(base_dados, janela):
    datas = pd.to_datetime(
        base_dados["Exchange Date"].iloc[janela + 1 :],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    return set(datas.dropna())


def combinacao_rf_sem_fs_completa(caminho_saida, base_dados, target, janela):
    if not caminho_saida.exists():
        return False

    esperado = datas_esperadas_para_ativo(base_dados, janela)
    if not esperado:
        return True

    df = pd.read_csv(caminho_saida)
    if df.empty:
        return False

    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df["janela"] = pd.to_numeric(df["janela"], errors="coerce")
    df = df[
        (df["target"].round(3) == round(target, 3))
        & (df["janela"] == janela)
        & (df["tecnica"] == TECNICA_TRAIN)
    ].copy()

    if df.empty:
        return False

    df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.strftime("%Y-%m-%d")
    presentes = set(df["data"].dropna())
    return esperado.issubset(presentes)


def treinar_random_forest_sem_fs(
    ativo,
    base_dados,
    janela,
    target,
    params_cache_path,
):
    x_dados = x_split(base_dados)
    y_dados = y_split(base_dados)
    z_dados = z_split(base_dados)

    params_cache = carregar_json(params_cache_path, {})
    registros_out = []
    registros_in = []

    qtd_treinamentos = len(base_dados) - janela - 1
    if qtd_treinamentos <= 0:
        return registros_out, registros_in

    for i in range(qtd_treinamentos):
        x_janela_atual = x_dados[i : i + janela].copy()
        y_janela_atual = y_dados[i : i + janela]
        x_teste_df = x_dados.iloc[i + janela : i + janela + 1].copy()

        data_inicio_janela = z_dados.at[i, "Exchange Date"]
        data_final_janela = z_dados.at[i + janela - 1, "Exchange Date"]
        data_atual = z_dados.at[i + janela + 1, "Exchange Date"]
        ano_atual = data_atual.year

        resultado_real = z_dados.at[i + janela + 1, "resultado_real"]
        target_real = z_dados.at[i + janela + 1, "target"]
        y_real_atual = int(y_dados[i + janela + 1])

        x_treino = x_janela_atual.values
        y_treino = y_janela_atual
        x_teste = x_teste_df[x_janela_atual.columns].values

        classes_treino = np.unique(y_treino)
        if len(classes_treino) < 2:
            pred_constante = int(classes_treino[0])
            y_in = [pred_constante] * len(y_treino)
            precision_in = precision_score(y_treino, y_in, zero_division=0)
            pred = pred_constante
        else:
            scaler = StandardScaler()
            x_treino_scaled = scaler.fit_transform(x_treino)
            x_teste_scaled = scaler.transform(x_teste)

            chave = params_key(ativo, ano_atual, janela)
            if chave not in params_cache:
                print(
                    f"🔍 GridSearch RF sem FS | ativo {ativo} | "
                    f"ano {ano_atual} | janela {janela}"
                )
                try:
                    grid = GridSearchCV(
                        RandomForestClassifier(random_state=42),
                        PARAMETROS_RF,
                        cv=3,
                        scoring="accuracy",
                        n_jobs=-1,
                    )
                    grid.fit(x_treino_scaled, y_treino)
                    params_cache[chave] = grid.best_estimator_.get_params()
                except ValueError as exc:
                    print(
                        f"  Aviso: GridSearch falhou para {ativo} | ano {ano_atual} "
                        f"| janela {janela}: {exc}. Usando parametros padrao."
                    )
                    params_cache[chave] = RandomForestClassifier(
                        random_state=42
                    ).get_params()
                salvar_json(params_cache_path, params_cache)

            model = RandomForestClassifier(**params_cache[chave])
            model.fit(x_treino_scaled, y_treino)

            y_in = model.predict(x_treino_scaled)
            precision_in = precision_score(y_treino, y_in, zero_division=0)
            pred = int(model.predict(x_teste_scaled)[0])

        registros_out.append(
            {
                "ativo": ativo,
                "target": target,
                "janela": janela,
                "tecnica": TECNICA_TRAIN,
                "data": data_atual,
                "target_real": target_real,
                "target_pred": pred,
                "resultado_real": resultado_real,
            }
        )
        registros_in.append(
            {
                "ativo": ativo,
                "target": target,
                "janela": janela,
                "data_inicio_janela": data_inicio_janela,
                "data_final_janela": data_final_janela,
                "tecnica": TECNICA_TRAIN,
                "y_treino": list(y_treino),
                "y_in": list(y_in),
                "precision": precision_in,
            }
        )

        if i == 0 or (i + 1) == qtd_treinamentos or (i + 1) % max(250, qtd_treinamentos // 10) == 0:
            percentual = ((i + 1) / qtd_treinamentos) * 100
            print(
                f"   andamento {ativo} | target {target} | janela {janela}: "
                f"{i + 1}/{qtd_treinamentos} ({percentual:.1f}%) | "
                f"data {pd.to_datetime(data_atual).date()}"
            )

    return registros_out, registros_in


def salvar_registros_experimento(caminho, registros):
    if not registros:
        return

    df = pd.DataFrame(registros)
    for coluna in df.columns:
        if "data" in coluna.lower():
            df[coluna] = pd.to_datetime(df[coluna], errors="coerce").dt.strftime("%Y-%m-%d")
    df.to_csv(caminho, index=False)


def executar_treino_experimento(exp_root, ativos_especificos=None):
    curated_dir = PROJECT_ROOT / "data" / "pre_process" / "curated"
    output_dir = garantir_pasta(exp_root / "train" / "outputs")
    input_dir = garantir_pasta(exp_root / "train" / "inputs")
    params_cache_path = exp_root / "train" / "params_cache_rf_no_fs.json"

    ativos = listar_ativos(curated_dir, ativos_especificos=ativos_especificos)
    if not ativos:
        raise ValueError("Nenhum ativo encontrado para o experimento.")

    print(f"\nTreinando RF sem feature selection para {len(ativos)} ativos...")

    for idx, ativo in enumerate(ativos, start=1):
        print(f"\n🔹 Ativo {idx}/{len(ativos)}: {ativo}")
        caminho_curated = curated_dir / f"{ativo}_target_{TARGET}.csv"

        if not caminho_curated.exists():
            print(f"  Pulando {ativo}: curated ausente em {caminho_curated}")
            continue

        base_dados = pd.read_csv(caminho_curated)
        base_dados["Exchange Date"] = pd.to_datetime(base_dados["Exchange Date"], errors="coerce")

        caminho_out = output_dir / f"target_previsto_{ativo}.csv"
        caminho_in = input_dir / f"target_in_{ativo}.csv"

        if combinacao_rf_sem_fs_completa(caminho_out, base_dados, TARGET, JANELA):
            print(f"  ✅ Ja completo no experimento: {ativo}")
            continue

        registros_out, registros_in = treinar_random_forest_sem_fs(
            ativo=ativo,
            base_dados=base_dados,
            janela=JANELA,
            target=TARGET,
            params_cache_path=params_cache_path,
        )

        salvar_registros_experimento(caminho_out, registros_out)
        salvar_registros_experimento(caminho_in, registros_in)
        print(f"  ✅ Saidas salvas para {ativo}")


def construir_2_tot_par_experimento(exp_root, ativos_especificos=None):
    original_dir = PROJECT_ROOT / "data" / "ensemble" / "2_tot_par"
    output_dir = garantir_pasta(exp_root / "ensemble" / "2_tot_par")
    exp_output_dir = exp_root / "train" / "outputs"

    ativos_filtrados = None
    if ativos_especificos:
        ativos_filtrados = {ativo.upper() for ativo in ativos_especificos}

    arquivos = sorted(original_dir.glob("*_ensemble_jan_tot_e_parcial.csv"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {original_dir}")

    for caminho in arquivos:
        ativo = caminho.name.replace("_ensemble_jan_tot_e_parcial.csv", "")
        if ativos_filtrados and ativo.upper() not in ativos_filtrados:
            continue

        df_base = pd.read_csv(caminho)
        caminho_exp = exp_output_dir / f"target_previsto_{ativo}.csv"

        if caminho_exp.exists():
            df_novo = pd.read_csv(caminho_exp)
            df_novo["target"] = pd.to_numeric(df_novo["target"], errors="coerce")
            df_novo["janela"] = pd.to_numeric(df_novo["janela"], errors="coerce")
            df_novo = df_novo[
                (df_novo["target"].round(3) == round(TARGET, 3))
                & (df_novo["janela"] == JANELA)
                & (df_novo["tecnica"] == TECNICA_TRAIN)
            ][["ativo", "target", "data", "target_real", "resultado_real", "target_pred"]].copy()

            if not df_novo.empty:
                df_base["target"] = pd.to_numeric(df_base["target"], errors="coerce")
                df_base["data"] = pd.to_datetime(df_base["data"], errors="coerce").dt.strftime("%Y-%m-%d")
                df_novo["data"] = pd.to_datetime(df_novo["data"], errors="coerce").dt.strftime("%Y-%m-%d")
                df_base["target_real"] = pd.to_numeric(df_base["target_real"], errors="coerce")
                df_base["resultado_real"] = pd.to_numeric(df_base["resultado_real"], errors="coerce")
                df_novo["target_real"] = pd.to_numeric(df_novo["target_real"], errors="coerce")
                df_novo["resultado_real"] = pd.to_numeric(df_novo["resultado_real"], errors="coerce")

                merged = df_base.merge(
                    df_novo,
                    on=["ativo", "target", "data", "target_real", "resultado_real"],
                    how="left",
                )
                mascara = merged["target"].round(3) == round(TARGET, 3)
                substituidos = merged.loc[mascara, "target_pred"].notna().sum()
                merged.loc[mascara & merged["target_pred"].notna(), COLUNA_TECNICA] = (
                    merged.loc[mascara & merged["target_pred"].notna(), "target_pred"]
                )
                df_base = merged.drop(columns=["target_pred"])
                print(f"  {ativo}: coluna {COLUNA_TECNICA} atualizada em {substituidos} linhas")

        df_base.to_csv(output_dir / caminho.name, index=False)


def copiar_filtrando(input_folder, output_folder, trecho_nome):
    input_folder = Path(input_folder)
    output_folder = garantir_pasta(Path(output_folder))

    copiados = 0
    for caminho in sorted(input_folder.glob("*.csv")):
        if trecho_nome in caminho.name:
            shutil.copy2(caminho, output_folder / caminho.name)
            copiados += 1

    if copiados == 0:
        raise FileNotFoundError(
            f"Nenhum arquivo com '{trecho_nome}' encontrado em {input_folder}"
        )

    return output_folder


def executar_pipeline_pos_treino(exp_root):
    from src.tecnicas.juncao_tecnicas_intraday import run_juncao_tecnicas_intraday
    from src.tecnicas.calculo_invest_tecnicas_2 import calcula_investimento_tecnicas
    from src.tecnicas.separacao_tecnicas_3 import run_separacao_tecnicas
    from src.tecnicas.capital_tecnicas_4 import calcula_capital_tecnicas
    from src.tecnicas.juncao_ativos_tecnicas_5 import run_juncao_ativos_tecnicas
    from src.tecnicas.target_tecnica_tecnicas_6 import separar_targets_por_tecnica_tecnicas
    from src.tecnicas.base_mv_sharpe_tecnicas_7 import run_base_mv_sharpe_tecnicas

    try:
        from src.tecnicas.otimizacao_mv_min_variancia_tecnicas_9 import run_otimizacao_mv_min_variancia_tecnicas
        from src.tecnicas.capital_mv_min_variancia_tecnicas_11 import run_capital_mv_min_variancia_tecnicas
    except ModuleNotFoundError as exc:
        if exc.name == "pypfopt":
            raise ModuleNotFoundError(
                "A etapa de Min Variancia exige a dependencia 'pypfopt'. "
                "Instale-a no ambiente antes de rodar a pos-pipeline do experimento."
            ) from exc
        raise

    intraday_dir = PROJECT_ROOT / "data" / "ensemble" / "7_intraday_join"
    ensemble_2_dir = exp_root / "ensemble" / "2_tot_par"

    juncao_dir = exp_root / "tecnicas" / "juncao_tecnicas_1"
    monetario_dir = exp_root / "tecnicas" / "monetario_tecnicas_2"
    separacao_dir = exp_root / "tecnicas" / "separacao_tecnicas_3"
    separacao_rf_dir = exp_root / "tecnicas" / "separacao_randomforest"
    capital_dir = exp_root / "tecnicas" / "capital_tecnicas_4"
    targets_alinhados_dir = exp_root / "tecnicas" / "targets_alinhados_tecnicas_5"
    targets_target102_dir = exp_root / "tecnicas" / "targets_alinhados_target_1_02"
    targets_por_tecnica_dir = exp_root / "tecnicas" / "targets_por_tecnica_tecnicas_6"
    targets_rf75_dir = exp_root / "tecnicas" / "targets_por_tecnica_rf75_target_1_02"
    base_mv_dir = exp_root / "tecnicas" / "base_mv_sharpe_tecnicas_7"
    pesos_mv_dir = exp_root / "tecnicas" / "mv_min_variancia_tecnicas_9"
    capital_mv_dir = exp_root / "tecnicas" / "capital_mv_min_variancia_tecnicas_11"

    print("\n[1/9] Juncao tecnicas + intraday")
    run_juncao_tecnicas_intraday(
        input_intraday=str(intraday_dir),
        input_tecnicas=str(ensemble_2_dir),
        output_folder=str(juncao_dir),
    )

    print("\n[2/9] Calculo monetario")
    calcula_investimento_tecnicas(
        input_folder=str(juncao_dir),
        output_folder=str(monetario_dir),
    )

    print("\n[3/9] Separacao por familia")
    run_separacao_tecnicas(
        input_folder=str(monetario_dir),
        output_folder=str(separacao_dir),
    )

    copiar_filtrando(separacao_dir, separacao_rf_dir, "_RandomForest_")

    print("\n[4/9] Capital por ativo para RandomForest")
    calcula_capital_tecnicas(
        input_folder=str(separacao_rf_dir),
        output_folder=str(capital_dir),
        capital_inicial=1000,
    )

    print("\n[5/9] Juncao de ativos por target")
    run_juncao_ativos_tecnicas(
        input_folder=str(capital_dir),
        output_folder=str(targets_alinhados_dir),
    )

    copiar_filtrando(targets_alinhados_dir, targets_target102_dir, "RandomForest_target_1_02")

    print("\n[6/9] Separacao por tecnica + janela")
    separar_targets_por_tecnica_tecnicas(
        input_folder=str(targets_target102_dir),
        output_folder=str(targets_por_tecnica_dir),
    )

    copiar_filtrando(targets_por_tecnica_dir, targets_rf75_dir, "RandomForest_75_target_1_02")

    print("\n[7/9] Base MV para o metodo filtrado")
    run_base_mv_sharpe_tecnicas(
        input_folder=str(targets_rf75_dir),
        output_folder=str(base_mv_dir),
        price_folder=str(PROJECT_ROOT / "data" / "pre_process" / "raw" / "refinitiv"),
    )

    print("\n[8/9] Otimizacao MV Min Variancia")
    run_otimizacao_mv_min_variancia_tecnicas(
        input_folder=str(base_mv_dir),
        output_folder=str(pesos_mv_dir),
    )

    print("\n[9/9] Capital MV Min Variancia")
    run_capital_mv_min_variancia_tecnicas(
        weights_folder=str(pesos_mv_dir),
        targets_folder=str(targets_rf75_dir),
        output_folder=str(capital_mv_dir),
        capital_inicial=100.0,
        dias_max_posicao=4,
    )


def gerar_relatorio_comparacao(exp_root):
    reports_dir = garantir_pasta(exp_root / "reports")
    arquivo_original = PROJECT_ROOT / "data" / "tecnicas" / "capital_mv_min_variancia_tecnicas_11" / "RandomForest_75_target_1_02.csv"
    arquivo_experimento = exp_root / "tecnicas" / "capital_mv_min_variancia_tecnicas_11" / "RandomForest_75_target_1_02.csv"

    if not arquivo_original.exists():
        raise FileNotFoundError(f"Arquivo original nao encontrado: {arquivo_original}")
    if not arquivo_experimento.exists():
        raise FileNotFoundError(f"Arquivo do experimento nao encontrado: {arquivo_experimento}")

    df_original = pd.read_csv(arquivo_original)
    df_experimento = pd.read_csv(arquivo_experimento)

    col_capital = "capital_total_min_variancia"

    original_final = float(pd.to_numeric(df_original[col_capital], errors="coerce").dropna().iloc[-1])
    experimento_final = float(pd.to_numeric(df_experimento[col_capital], errors="coerce").dropna().iloc[-1])

    resumo = pd.DataFrame(
        [
            {
                "cenario": "original_com_feature_selection",
                "arquivo": str(arquivo_original),
                "linhas": len(df_original),
                "data_inicial": pd.to_datetime(df_original["data"], errors="coerce").min(),
                "data_final": pd.to_datetime(df_original["data"], errors="coerce").max(),
                "capital_final": original_final,
            },
            {
                "cenario": "randomforest_75_target_1_02_sem_fs",
                "arquivo": str(arquivo_experimento),
                "linhas": len(df_experimento),
                "data_inicial": pd.to_datetime(df_experimento["data"], errors="coerce").min(),
                "data_final": pd.to_datetime(df_experimento["data"], errors="coerce").max(),
                "capital_final": experimento_final,
            },
            {
                "cenario": "diferenca_experimento_menos_original",
                "arquivo": "",
                "linhas": len(df_experimento) - len(df_original),
                "data_inicial": pd.NaT,
                "data_final": pd.NaT,
                "capital_final": experimento_final - original_final,
            },
        ]
    )

    caminho_saida = reports_dir / "comparacao_capital_randomforest_75_target_1_02_min_variancia.csv"
    resumo.to_csv(caminho_saida, index=False)
    print(f"\nRelatorio salvo em: {caminho_saida}")
    print(resumo.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Roda um experimento isolado para RandomForest | janela 75 | "
            "target 1.02 sem feature selection e propaga ate o capital "
            "Min Variancia em pasta separada."
        )
    )
    parser.add_argument(
        "--ativos",
        nargs="+",
        help="Opcional: limita o experimento aos ativos informados, ex.: --ativos PETR4 VALE3",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Pula a etapa de treino e reaproveita as saidas ja geradas no experimento.",
    )
    parser.add_argument(
        "--skip-pos-pipeline",
        action="store_true",
        help="Pula as etapas posteriores ao treino.",
    )
    args = parser.parse_args()

    exp_root = garantir_pasta(PROJECT_ROOT / "data" / "experimentos" / EXPERIMENTO_NOME)

    print(f"\nExperimento: {EXPERIMENTO_NOME}")
    print(f"Pasta raiz: {exp_root}")

    if not args.skip_train:
        executar_treino_experimento(exp_root, ativos_especificos=args.ativos)

    construir_2_tot_par_experimento(exp_root, ativos_especificos=args.ativos)

    if not args.skip_pos_pipeline:
        executar_pipeline_pos_treino(exp_root)
        gerar_relatorio_comparacao(exp_root)

    print("\nExperimento concluido.")


if __name__ == "__main__":
    main()
