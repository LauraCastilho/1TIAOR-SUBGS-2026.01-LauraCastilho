"""
Modelos de arvore (RandomForest, GradientBoosting, XGBoost) para os tres
alvos. Treinam direto sobre os CSVs de data/processed/ (sem scaler, arvores
nao precisam disso). O desbalanceamento de status_saude (~85/12/3%) e
tratado com sample_weight balanceado, igual ao esquema de pesos da RNA.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBRegressor

    TEM_XGBOOST = True
except ImportError:  # pragma: no cover - fallback caso xgboost nao esteja instalado
    TEM_XGBOOST = False


def treinar_random_forest_classificador(X, y, balancear=True):
    pesos = compute_sample_weight("balanced", y) if balancear else None
    modelo = RandomForestClassifier(
        n_estimators=200, max_depth=14, random_state=config.RANDOM_STATE, n_jobs=-1
    )
    modelo.fit(X, y, sample_weight=pesos)
    return modelo


def treinar_gradient_boosting_classificador(X, y, balancear=True):
    pesos = compute_sample_weight("balanced", y) if balancear else None
    modelo = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1, random_state=config.RANDOM_STATE
    )
    modelo.fit(X, y, sample_weight=pesos)
    return modelo


def treinar_random_forest_regressor(X, y):
    modelo = RandomForestRegressor(
        n_estimators=200, max_depth=14, random_state=config.RANDOM_STATE, n_jobs=-1
    )
    modelo.fit(X, y)
    return modelo


def treinar_regressor_boosting(X, y):
    """XGBoost quando disponivel; caso contrario, GradientBoostingRegressor
    do scikit-learn como alternativa equivalente (pedida no enunciado como
    'XGBoost ou Gradient Boosting')."""
    if TEM_XGBOOST:
        modelo = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=config.RANDOM_STATE,
        )
    else:
        modelo = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=config.RANDOM_STATE
        )
    modelo.fit(X, y)
    return modelo
