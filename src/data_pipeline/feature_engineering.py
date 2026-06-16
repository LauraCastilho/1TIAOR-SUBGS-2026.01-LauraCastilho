"""
Feature engineering do dataset do AgroNexusSpace.

Cria variaveis derivadas (temporais, fenologicas, de tendencia e de
interacao climatica) a partir do dataset limpo, e codifica as variaveis
categoricas de entrada (cultura, estacao do ano). O resultado e o conjunto
de features consumido pelos modelos de ML descritos na arquitetura:
necessidade de irrigacao, status de saude da vegetacao e produtividade.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

import numpy as np
import pandas as pd

# Codificacao ordinal do estagio fenologico, na ordem natural do ciclo
# produtivo (definida em config.ESTAGIOS_FENOLOGICOS).
ESTAGIO_ORDEM = [nome for nome, _, _ in config.ESTAGIOS_FENOLOGICOS]
ESTAGIO_COD = {nome: i for i, nome in enumerate(ESTAGIO_ORDEM)}

# Duracao total do ciclo de cada cultura, usada para normalizar
# "dias_desde_plantio" em uma escala comparavel entre culturas (0 a 1).
CICLO_TOTAL = {cultura: params["ciclo_dias"] for cultura, params in config.CULTURAS.items()}

ESTACAO_POR_MES = {
    12: "verao", 1: "verao", 2: "verao",
    3: "outono", 4: "outono", 5: "outono",
    6: "inverno", 7: "inverno", 8: "inverno",
    9: "primavera", 10: "primavera", 11: "primavera",
}


def adicionar_features_temporais(df):
    """Mes e estacao do ano (sazonalidade climatica do Cerrado)."""
    df["mes"] = df["data"].dt.month
    df["estacao"] = df["mes"].map(ESTACAO_POR_MES)
    return df


def adicionar_features_fenologicas(df):
    """Progresso do ciclo (0-1, comparavel entre culturas de duracoes
    diferentes) e codificacao ordinal do estagio fenologico."""
    df["ciclo_total_dias"] = df["cultura"].astype(str).map(CICLO_TOTAL)
    df["dias_desde_plantio_pct"] = (df["dias_desde_plantio"] / df["ciclo_total_dias"]).round(4)
    df["estagio_fenologico_cod"] = (
        df["estagio_fenologico"].astype(str).map(ESTAGIO_COD).astype("int8")
    )
    return df


def adicionar_features_temporais_rolantes(df):
    """Features de tendencia/memoria por talhao-safra.

    Todas usam groupby(...).transform(), que preserva o indice original do
    DataFrame e garante que a janela (rolling/lag) nunca atravesse a
    fronteira entre talhoes ou entre safras diferentes do mesmo talhao.
    """
    df = df.sort_values(["talhao_id", "safra", "data"]).reset_index(drop=True)
    janela = config.JANELA_ROLLING_DIAS
    grupo = df.groupby(["talhao_id", "safra"])

    df["ndvi_media_7d"] = grupo["ndvi"].transform(
        lambda s: s.rolling(janela, min_periods=1).mean()
    ).round(4)

    ndvi_lag_7d = grupo["ndvi"].transform(lambda s: s.shift(janela))
    df["ndvi_tendencia_7d"] = (df["ndvi"] - ndvi_lag_7d).fillna(0).round(4)

    df["precipitacao_acum_7d"] = grupo["precipitacao_mm"].transform(
        lambda s: s.rolling(janela, min_periods=1).sum()
    ).round(2)

    lag1 = grupo["umidade_solo_sensor_pct"].transform(lambda s: s.shift(1))
    df["umidade_solo_sensor_lag1"] = lag1.fillna(df["umidade_solo_sensor_pct"]).round(2)
    df["umidade_solo_sensor_delta1"] = (
        df["umidade_solo_sensor_pct"] - df["umidade_solo_sensor_lag1"]
    ).round(2)

    return df


def adicionar_features_climaticas(df):
    """Interacoes simples entre variaveis climaticas/de sensor que ajudam a
    sinalizar estresse hidrico e termico."""
    df["amplitude_termica"] = (df["temperatura_superficie_c"] - df["temperatura_ar_c"]).round(2)

    df["indice_estresse_climatico"] = (
        df["temperatura_ar_c"] * (100 - df["umidade_ar_pct"]) / 100
    ).round(2)

    razao = df["umidade_solo_sensor_pct"] / df["umidade_solo_satelite_pct"].replace(0, np.nan)
    df["razao_umidade_sensor_satelite"] = razao.fillna(1.0).round(4)

    return df


def codificar_categoricas(df):
    """One-hot encoding de cultura e estacao (drop_first=True).

    drop_first evita a "dummy variable trap" (multicolinearidade perfeita)
    para modelos lineares/redes neurais; modelos baseados em arvore nao sao
    afetados por essa escolha. A categoria removida (a primeira em ordem
    alfabetica entre as categorias presentes) passa a ser a "baseline"
    implicita:
      - cultura:  Algodao  (dummies: cultura_Feijao, cultura_Milho, cultura_Soja)
      - estacao:  outono   (dummies: estacao_primavera, estacao_verao;
                  'inverno' nunca ocorre, pois o ciclo agricola vai de
                  set/out a mar/abr)
    """
    colunas_antes = set(df.columns)
    df = pd.get_dummies(
        df, columns=["cultura", "estacao"], prefix=["cultura", "estacao"], drop_first=True
    )
    novas_colunas = [c for c in df.columns if c not in colunas_antes]
    df[novas_colunas] = df[novas_colunas].astype(int)
    return df


def aplicar_feature_engineering(df):
    df = adicionar_features_temporais(df)
    df = adicionar_features_fenologicas(df)
    df = adicionar_features_temporais_rolantes(df)
    df = adicionar_features_climaticas(df)
    df = codificar_categoricas(df)
    return df


if __name__ == "__main__":
    from preprocessing import preprocessar

    df = preprocessar()
    df = aplicar_feature_engineering(df)
    print(f"Shape final: {df.shape}")
    print(df.columns.tolist())
