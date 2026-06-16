"""
Orquestrador da camada de Machine Learning do AgroNexusSpace.

Para cada uma das tres variaveis-alvo geradas pela camada de dados
(data/processed/train.csv e test.csv), treina e compara modelos, gera os
graficos exigidos (matriz de confusao, feature importance, comparacao de
modelos, curvas de treinamento da RNA, correlacao e distribuicao de risco)
e salva os modelos treinados + um resumo de metricas.

Uso:
    python src/machine_learning/treinar_modelos.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import data_loader
import avaliacao as av
import modelos_classicos as mc
from rede_neural_numpy import MLPClassifier

import joblib
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def preparar_dados():
    treino, teste = data_loader.carregar_treino_teste()
    features = data_loader.colunas_features(treino)
    return treino, teste, features

# EDA
def secao_eda(treino, teste):
    print("\n=== EDA: correlacao e distribuicao das variaveis de risco ===")
    colunas_corr = [
        "ndvi", "evi", "temperatura_superficie_c", "umidade_solo_satelite_pct",
        "precipitacao_mm", "temperatura_ar_c", "umidade_ar_pct",
        "umidade_solo_sensor_pct", "luminosidade_lux", "ndvi_media_7d",
        "ndvi_tendencia_7d", "amplitude_termica", "indice_estresse_climatico",
        "razao_umidade_sensor_satelite", "necessidade_irrigacao",
        "produtividade_estimada_ton_ha",
    ]
    av.plot_correlacao(treino, colunas_corr)
    av.plot_distribuicao_risco(treino, teste)

# 1) status_saude (multiclasse: saudavel / atencao / critico)
def secao_status_saude(treino, teste, features, metricas):
    print("\n=== status_saude (classificacao multiclasse - risco/saude da lavoura) ===")
    classes = config.STATUS_SAUDE_CLASSES
    indice_classe = {nome: i for i, nome in enumerate(classes)}

    X_train, y_train = data_loader.separar_x_y(treino, config.COL_TARGET_SAUDE, features)
    X_test, y_test = data_loader.separar_x_y(teste, config.COL_TARGET_SAUDE, features)

    resultados = {}

    # Random Forest
    rf = mc.treinar_random_forest_classificador(X_train, y_train)
    pred_rf = rf.predict(X_test)
    resultados["random_forest"] = av.avaliar_classificacao(y_test, pred_rf, labels=classes)
    av.plot_matriz_confusao(
        y_test, pred_rf, classes,
        "Matriz de Confusao - Saude da Lavoura (Random Forest)",
        "matriz_confusao_status_saude_random_forest.png",
    )
    av.plot_feature_importance(
        rf.feature_importances_, features,
        "Feature Importance - Saude da Lavoura (Random Forest)",
        "feature_importance_status_saude_random_forest.png",
    )
    joblib.dump(rf, config.MODELS_DIR / "status_saude_random_forest.joblib")

    # Gradient Boosting
    gb = mc.treinar_gradient_boosting_classificador(X_train, y_train)
    pred_gb = gb.predict(X_test)
    resultados["gradient_boosting"] = av.avaliar_classificacao(y_test, pred_gb, labels=classes)
    av.plot_matriz_confusao(
        y_test, pred_gb, classes,
        "Matriz de Confusao - Saude da Lavoura (Gradient Boosting)",
        "matriz_confusao_status_saude_gradient_boosting.png",
    )
    joblib.dump(gb, config.MODELS_DIR / "status_saude_gradient_boosting.joblib")

    # RNA (NumPy)
    y_train_idx = y_train.map(indice_classe).to_numpy()
    y_test_idx = y_test.map(indice_classe).to_numpy()

    scaler = data_loader.ajustar_scaler_rna(X_train)
    X_train_rna = scaler.transform(X_train)
    X_test_rna = scaler.transform(X_test)

    pesos_classe = compute_class_weight(
        "balanced", classes=np.arange(len(classes)), y=y_train_idx
    )

    print("Treinando RNA (NumPy) para status_saude...")
    rna = MLPClassifier(
        tamanho_entrada=X_train_rna.shape[1],
        tamanhos_ocultos=[32, 16],
        n_classes=len(classes),
        taxa_aprendizado=0.02,
        l2=1e-3,
        seed=config.RANDOM_STATE,
    )
    historico = rna.treinar(
        X_train_rna, y_train_idx, X_test_rna, y_test_idx,
        epochs=150, batch_size=64, pesos_classe=pesos_classe,
    )

    pred_rna_idx = rna.prever(X_test_rna)
    pred_rna = np.array(classes)[pred_rna_idx]
    resultados["rna_numpy"] = av.avaliar_classificacao(y_test, pred_rna, labels=classes)
    av.plot_matriz_confusao(
        y_test, pred_rna, classes,
        "Matriz de Confusao - Saude da Lavoura (RNA NumPy)",
        "matriz_confusao_status_saude_rna_numpy.png",
    )
    av.plot_curvas_treinamento(
        historico, "RNA (NumPy) - status_saude (saude/risco da lavoura)",
        "curvas_treinamento_rna.png",
    )
    rna.salvar(config.MODELS_DIR / "status_saude_rna_numpy.npz")
    joblib.dump(scaler, config.MODELS_DIR / "status_saude_rna_scaler.joblib")

    av.plot_comparacao_modelos(
        resultados, "f1_macro",
        "Comparacao de Modelos - Saude da Lavoura (F1 macro)",
        "comparacao_modelos_status_saude.png",
    )

    metricas["status_saude"] = resultados
    for nome, m in resultados.items():
        print(f"  {nome}: {m}")

# 2) necessidade_irrigacao (binaria)
def secao_irrigacao(treino, teste, features, metricas):
    print("\n=== necessidade_irrigacao (classificacao binaria - risco hidrico) ===")
    X_train, y_train = data_loader.separar_x_y(treino, config.COL_TARGET_IRRIGACAO, features)
    X_test, y_test = data_loader.separar_x_y(teste, config.COL_TARGET_IRRIGACAO, features)
    labels = [0, 1]

    resultados = {}

    rf = mc.treinar_random_forest_classificador(X_train, y_train)
    pred_rf = rf.predict(X_test)
    resultados["random_forest"] = av.avaliar_classificacao(y_test, pred_rf, labels=labels)
    av.plot_matriz_confusao(
        y_test, pred_rf, labels,
        "Matriz de Confusao - Necessidade de Irrigacao (Random Forest)",
        "matriz_confusao_irrigacao_random_forest.png",
    )
    av.plot_feature_importance(
        rf.feature_importances_, features,
        "Feature Importance - Necessidade de Irrigacao (Random Forest)",
        "feature_importance_irrigacao_random_forest.png",
    )
    joblib.dump(rf, config.MODELS_DIR / "irrigacao_random_forest.joblib")

    gb = mc.treinar_gradient_boosting_classificador(X_train, y_train)
    pred_gb = gb.predict(X_test)
    resultados["gradient_boosting"] = av.avaliar_classificacao(y_test, pred_gb, labels=labels)
    av.plot_matriz_confusao(
        y_test, pred_gb, labels,
        "Matriz de Confusao - Necessidade de Irrigacao (Gradient Boosting)",
        "matriz_confusao_irrigacao_gradient_boosting.png",
    )
    joblib.dump(gb, config.MODELS_DIR / "irrigacao_gradient_boosting.joblib")

    av.plot_comparacao_modelos(
        resultados, "f1_macro",
        "Comparacao de Modelos - Necessidade de Irrigacao (F1 macro)",
        "comparacao_modelos_irrigacao.png",
    )

    metricas["necessidade_irrigacao"] = resultados
    for nome, m in resultados.items():
        print(f"  {nome}: {m}")

# 3) produtividade_estimada_ton_ha (regressao)
def secao_produtividade(treino, teste, features, metricas):
    print("\n=== produtividade_estimada_ton_ha (regressao) ===")
    X_train, y_train = data_loader.separar_x_y(treino, config.COL_TARGET_PRODUTIVIDADE, features)
    X_test, y_test = data_loader.separar_x_y(teste, config.COL_TARGET_PRODUTIVIDADE, features)

    resultados = {}

    rf = mc.treinar_random_forest_regressor(X_train, y_train)
    pred_rf = rf.predict(X_test)
    resultados["random_forest"] = av.avaliar_regressao(y_test, pred_rf)
    av.plot_feature_importance(
        rf.feature_importances_, features,
        "Feature Importance - Produtividade (Random Forest)",
        "feature_importance_produtividade_random_forest.png",
    )
    joblib.dump(rf, config.MODELS_DIR / "produtividade_random_forest.joblib")

    boosting = mc.treinar_regressor_boosting(X_train, y_train)
    nome_boosting = "xgboost" if mc.TEM_XGBOOST else "gradient_boosting"
    pred_boosting = boosting.predict(X_test)
    resultados[nome_boosting] = av.avaliar_regressao(y_test, pred_boosting)
    joblib.dump(boosting, config.MODELS_DIR / f"produtividade_{nome_boosting}.joblib")

    av.plot_comparacao_modelos(
        resultados, "r2",
        "Comparacao de Modelos - Produtividade (R2)",
        "comparacao_modelos_produtividade.png",
    )

    metricas["produtividade_estimada_ton_ha"] = resultados
    for nome, m in resultados.items():
        print(f"  {nome}: {m}")

# Main
def main():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

    treino, teste, features = preparar_dados()
    print(f"Treino: {treino.shape} | Teste: {teste.shape} | Features de entrada: {len(features)}")

    with open(config.MODELS_DIR / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    metricas = {}

    secao_eda(treino, teste)
    secao_status_saude(treino, teste, features, metricas)
    secao_irrigacao(treino, teste, features, metricas)
    secao_produtividade(treino, teste, features, metricas)

    with open(config.METRICAS_PATH, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    print(f"\nMetricas salvas em: {config.METRICAS_PATH.relative_to(config.BASE_DIR)}")
    print("Treinamento concluido com sucesso.")

if __name__ == "__main__":
    main()
