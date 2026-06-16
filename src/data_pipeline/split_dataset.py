"""
Split treino/teste + ajuste do scaler.

Split por GRUPO (talhao_id), nao por linha: o dataset e um painel temporal
(varios dias por talhao), entao um split por linha colocaria dias quase
iguais do mesmo talhao em treino e teste (data leakage). Separando por
talhao, o teste avalia generalizacao para talhoes nunca vistos.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# Colunas numericas continuas que se beneficiam de padronizacao (media 0,
# desvio 1) para uso por modelos baseados em redes neurais ou regressao
# linear/logistica. Identificadores, datas, variaveis-alvo e colunas
# binarias/one-hot ficam de fora.
COLUNAS_NUMERICAS_PARA_ESCALA = [
    "area_ha", "latitude", "longitude", "dia_juliano", "dias_desde_plantio",
    "ndvi", "evi", "temperatura_superficie_c", "umidade_solo_satelite_pct",
    "precipitacao_mm", "temperatura_ar_c", "umidade_ar_pct",
    "umidade_solo_sensor_pct", "luminosidade_lux",
    "dias_desde_plantio_pct", "ndvi_media_7d", "ndvi_tendencia_7d",
    "precipitacao_acum_7d", "umidade_solo_sensor_lag1", "umidade_solo_sensor_delta1",
    "amplitude_termica", "indice_estresse_climatico", "razao_umidade_sensor_satelite",
]


def dividir_treino_teste(df, test_size=None, seed=None):
    test_size = test_size if test_size is not None else config.TEST_SIZE
    seed = seed if seed is not None else config.SEED

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_treino, idx_teste = next(splitter.split(df, groups=df["talhao_id"]))

    treino = df.iloc[idx_treino].reset_index(drop=True)
    teste = df.iloc[idx_teste].reset_index(drop=True)
    return treino, teste


def ajustar_scaler(treino):
    """Ajusta o StandardScaler apenas com dados de treino (evita vazamento
    de informacao do conjunto de teste para os parametros de normalizacao).
    """
    scaler = StandardScaler()
    scaler.fit(treino[COLUNAS_NUMERICAS_PARA_ESCALA])
    return scaler


def relatorio_balanceamento(treino, teste):
    for nome, df in [("Treino", treino), ("Teste", teste)]:
        print(f"\n[{nome}] talhoes={df['talhao_id'].nunique()} | registros={len(df)}")
        print(f"[{nome}] necessidade_irrigacao:")
        print(df["necessidade_irrigacao"].value_counts(normalize=True).round(3).to_string())
        print(f"[{nome}] status_saude:")
        print(df["status_saude"].value_counts(normalize=True).round(3).to_string())


def salvar_datasets(treino, teste, scaler):
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    treino.to_csv(config.TRAIN_DATASET_PATH, index=False)
    teste.to_csv(config.TEST_DATASET_PATH, index=False)
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"\nTreino salvo em {config.TRAIN_DATASET_PATH} -> {treino.shape}")
    print(f"Teste  salvo em {config.TEST_DATASET_PATH} -> {teste.shape}")
    print(f"Scaler salvo em {config.SCALER_PATH}")


def processar_split(df):
    treino, teste = dividir_treino_teste(df)
    scaler = ajustar_scaler(treino)
    relatorio_balanceamento(treino, teste)
    salvar_datasets(treino, teste, scaler)
    return treino, teste, scaler


if __name__ == "__main__":
    from preprocessing import preprocessar
    from feature_engineering import aplicar_feature_engineering

    df = preprocessar()
    df = aplicar_feature_engineering(df)
    processar_split(df)
