"""
Carregamento dos datasets processados (data/processed/) e separacao em
features (X) e alvos (y) para a camada de Machine Learning.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

import pandas as pd
from sklearn.preprocessing import StandardScaler


def carregar_treino_teste():
    """Carrega train.csv e test.csv gerados por src/data_pipeline/."""
    treino = pd.read_csv(config.TRAIN_PATH, parse_dates=["data"])
    teste = pd.read_csv(config.TEST_PATH, parse_dates=["data"])
    return treino, teste


def colunas_features(df):
    """Lista de colunas de entrada dos modelos: todas exceto identificadores
    e as tres variaveis-alvo."""
    excluir = set(config.COLUNAS_ID + config.COLUNAS_TARGET)
    return [c for c in df.columns if c not in excluir]


def separar_x_y(df, alvo, features):
    """Retorna (X, y) para o alvo informado, descartando linhas sem rotulo."""
    dados = df.dropna(subset=[alvo])
    return dados[features], dados[alvo]


def ajustar_scaler_rna(X_train):
    """StandardScaler proprio para as 31 features de entrada da RNA (ajustado
    so no treino) -- diferente do scaler.joblib do data_pipeline, que cobre
    apenas as colunas continuas."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler
