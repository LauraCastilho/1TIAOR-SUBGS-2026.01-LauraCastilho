"""
Limpeza do dataset sintetico: pega data/raw/dados_agricolas_sinteticos.csv e
devolve um DataFrame pronto para o feature engineering (remove duplicatas,
trata NaN dos sensores ESP32, valida faixas fisicas e ajusta tipos).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

import pandas as pd

# Colunas de sensores ESP32 sujeitas a falhas de transmissao (ver
# generate_synthetic_data.aplicar_falhas_sensores)
COLUNAS_SENSOR = [
    "temperatura_ar_c",
    "umidade_ar_pct",
    "umidade_solo_sensor_pct",
    "luminosidade_lux",
]

# Colunas com limites fisicos conhecidos, usadas para validacao defensiva
COLUNAS_PERCENTUAL = ["umidade_solo_satelite_pct", "umidade_solo_sensor_pct", "umidade_ar_pct"]
COLUNAS_INDICE = ["ndvi", "evi"]


def carregar_dataset_bruto(caminho=None):
    caminho = caminho or config.RAW_DATASET_PATH
    return pd.read_csv(caminho, parse_dates=["data"])


def remover_duplicatas(df):
    """Remove registros duplicados para o mesmo talhao/safra/data (chave
    natural de uma leitura diaria por talhao)."""
    antes = len(df)
    df = df.drop_duplicates(subset=["talhao_id", "safra", "data"]).reset_index(drop=True)
    removidos = antes - len(df)
    if removidos:
        print(f"Removidas {removidos} linha(s) duplicada(s) (talhao_id, safra, data).")
    return df


def tratar_valores_ausentes(df):
    """Imputa lacunas de sensores ESP32: forward-fill / backward-fill dentro
    de cada serie (talhao_id, safra) ordenada por data e, se ainda sobrar
    NaN (serie toda ausente), preenche com a mediana global da coluna."""
    df = df.sort_values(["talhao_id", "safra", "data"]).reset_index(drop=True)

    n_antes = int(df[COLUNAS_SENSOR].isna().sum().sum())

    grupo = df.groupby(["talhao_id", "safra"], group_keys=False)
    df[COLUNAS_SENSOR] = grupo[COLUNAS_SENSOR].apply(lambda g: g.ffill().bfill())

    for col in COLUNAS_SENSOR:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    n_depois = int(df[COLUNAS_SENSOR].isna().sum().sum())
    print(f"Valores ausentes em colunas de sensores: {n_antes} -> {n_depois}")
    return df


def validar_faixas_fisicas(df):
    """Clipa variaveis para limites fisicamente plausiveis (% entre 0-100,
    indices entre 0-1, chuva/luminosidade >= 0) -- protege contra leituras
    fora da faixa que viriam de sensores reais."""
    for col in COLUNAS_PERCENTUAL:
        df[col] = df[col].clip(0, 100)
    for col in COLUNAS_INDICE:
        df[col] = df[col].clip(0, 1)
    df["precipitacao_mm"] = df["precipitacao_mm"].clip(lower=0)
    df["luminosidade_lux"] = df["luminosidade_lux"].clip(lower=0)
    return df


def ajustar_tipos(df):
    """Define tipos de dados eficientes e semanticamente corretos."""
    df["cultura"] = df["cultura"].astype("category")
    df["estagio_fenologico"] = df["estagio_fenologico"].astype("category")
    df["status_saude"] = df["status_saude"].astype("category")
    df["necessidade_irrigacao"] = df["necessidade_irrigacao"].astype("int8")
    return df


def preprocessar(df=None):
    if df is None:
        df = carregar_dataset_bruto()
    df = remover_duplicatas(df)
    df = tratar_valores_ausentes(df)
    df = validar_faixas_fisicas(df)
    df = ajustar_tipos(df)
    return df


if __name__ == "__main__":
    df = preprocessar()
    print(df.info())
    print(df.head())
