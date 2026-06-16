"""
Gerador de dados sinteticos do AgroNexusSpace.

Para cada talhao/safra, monta uma serie diaria com dados "espaciais"
(NDVI, EVI, LST, umidade do solo via satelite, chuva), dados de sensores
ESP32 (temperatura/umidade do ar, umidade do solo, luminosidade) e os tres
alvos de ML (irrigacao, status de saude, produtividade). Usa config.SEED,
entao roda duas vezes = mesmo dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

import numpy as np
import pandas as pd

# Talhoes
def gerar_talhoes(rng):
    """Cria 1 talhao por cultura em cada fazenda (4 fazendas x 4 culturas)."""
    registros = []
    for fazenda in config.FAZENDAS:
        lat_centro = rng.uniform(config.LAT_MIN, config.LAT_MAX)
        lon_centro = rng.uniform(config.LON_MIN, config.LON_MAX)
        for cultura in config.CULTURAS:
            talhao_id = f"{fazenda}-{cultura[:3].upper()}"
            lat = lat_centro + rng.uniform(-0.05, 0.05)
            lon = lon_centro + rng.uniform(-0.05, 0.05)
            area_ha = round(rng.uniform(40, 280), 1)
            registros.append(
                dict(
                    talhao_id=talhao_id,
                    fazenda_id=fazenda,
                    cultura=cultura,
                    area_ha=area_ha,
                    latitude=round(lat, 5),
                    longitude=round(lon, 5),
                )
            )
    return pd.DataFrame(registros)

def data_plantio(safra, cultura, rng):
    base = pd.Timestamp(config.SAFRA_INICIO[safra]) + pd.Timedelta(
        days=config.CULTURA_OFFSET_PLANTIO_DIAS[cultura]
    )
    jitter = int(rng.integers(-config.JITTER_PLANTIO_DIAS, config.JITTER_PLANTIO_DIAS + 1))
    return base + pd.Timedelta(days=jitter)

# Curva fenologica (NDVI esperado) e eventos de seca
def curva_ndvi_esperada(t, params):
    """Curva dupla-logistica de NDVI: cresce de ndvi_min a ndvi_max em torno
    de t1 (vegetativo/floracao) e volta a ndvi_min em torno de t2 (senescencia)."""
    ciclo = params["ciclo_dias"]
    t1 = params["t1_frac"] * ciclo
    t2 = params["t2_frac"] * ciclo
    crescimento = 1.0 / (1.0 + np.exp(-params["k1"] * (t - t1)))
    senescencia = 1.0 / (1.0 + np.exp(-params["k2"] * (t - t2)))
    return params["ndvi_min"] + (params["ndvi_max"] - params["ndvi_min"]) * (
        crescimento - senescencia
    )

def gerar_intensidade_seca(ciclo, rng):
    """Com probabilidade PROB_EVENTO_SECA, sorteia um periodo de estresse
    hidrico no ciclo. Retorna um array de intensidade [0, 1] usado para
    reduzir o NDVI esperado e a probabilidade de chuva na janela afetada."""
    intensidade = np.zeros(ciclo)
    if rng.random() >= config.PROB_EVENTO_SECA:
        return intensidade

    duracao = int(rng.integers(config.SECA_DURACAO_MIN, config.SECA_DURACAO_MAX + 1))
    inicio_min = int(ciclo * 0.15)
    inicio_max = max(inicio_min + 1, ciclo - duracao - config.SECA_DIAS_RECUPERACAO)
    if inicio_max <= inicio_min:
        return intensidade
    inicio = int(rng.integers(inicio_min, inicio_max))
    fim = inicio + duracao

    for idx in range(ciclo):
        if inicio <= idx <= fim:
            intensidade[idx] = config.SECA_REDUCAO_NDVI_MAX * (idx - inicio) / duracao
        elif fim < idx <= fim + config.SECA_DIAS_RECUPERACAO:
            decaimento = 1 - (idx - fim) / config.SECA_DIAS_RECUPERACAO
            intensidade[idx] = config.SECA_REDUCAO_NDVI_MAX * max(decaimento, 0)
    return intensidade

# Umidade do solo: modelo "balde com vazamento"
def simular_umidade_solo(precipitacao, temperatura_ar, ndvi_observado, rng):
    """Umidade do solo como um "balde com vazamento": recebe infiltracao da
    chuva e perde agua por evapotranspiracao. O sensor de superficie tem
    memoria mais curta e reage de forma mais forte/ruidosa a cada chuva do
    que a estimativa de satelite (zona de raizes)."""
    n = len(precipitacao)
    satelite = np.zeros(n)
    sensor = np.zeros(n)
    satelite[0] = config.UMIDADE_INICIAL_PCT
    sensor[0] = config.UMIDADE_INICIAL_PCT + rng.normal(0, 2)

    for t in range(1, n):
        et = (
            config.EVAPOTRANSPIRACAO_BASE
            + config.EVAPOTRANSPIRACAO_TEMP * max(temperatura_ar[t] - 20, 0)
            - 1.5 * ndvi_observado[t]
        )
        et = max(et, 0.2)

        satelite[t] = (
            satelite[t - 1] * config.UMIDADE_DECAIMENTO_SATELITE
            + config.UMIDADE_INFILTRACAO_SATELITE * precipitacao[t]
            - et
        )
        sensor[t] = (
            sensor[t - 1] * config.UMIDADE_DECAIMENTO_SENSOR
            + config.UMIDADE_INFILTRACAO_SENSOR * precipitacao[t]
            - 1.3 * et
            + rng.normal(0, 1.5)
        )

    satelite = np.clip(satelite, 8, 95)
    sensor = np.clip(sensor, 5, 100)
    return satelite, sensor

# Estagio fenologico
def nome_estagio(frac_ciclo):
    for nome, ini, fim in config.ESTAGIOS_FENOLOGICOS:
        if ini <= frac_ciclo < fim:
            return nome
    return config.ESTAGIOS_FENOLOGICOS[-1][0]

# Variaveis-alvo
def calcular_necessidade_irrigacao(umidade_sensor, precipitacao, estagio, params, rng):
    """Solo abaixo do limiar critico da cultura + sem chuva recente + fora de
    maturacao/colheita => precisa irrigar. Com pequeno ruido de rotulo
    (PROB_RUIDO_LABEL_IRRIGACAO) pra nao ficar perfeitamente determinista."""
    limiar = params["umidade_critica_pct"]
    nao_colheita = ~np.isin(estagio, ["maturacao", "colheita"])
    base = ((umidade_sensor < limiar) & (precipitacao < 3) & nao_colheita).astype(int)
    ruido = rng.random(len(base)) < config.PROB_RUIDO_LABEL_IRRIGACAO
    return np.where(ruido, 1 - base, base)

def calcular_status_saude(ndvi_observado, ndvi_esperado):
    """Razao ndvi_observado/ndvi_esperado como indice de anomalia: proximo
    de 1 = vegetacao normal para o estagio; bem abaixo de 1 = estresse."""
    razao = ndvi_observado / np.clip(ndvi_esperado, 1e-3, None)
    condicoes = [razao >= 0.92, razao >= 0.75]
    escolhas = ["saudavel", "atencao"]
    return np.select(condicoes, escolhas, default="critico")

def calcular_produtividade(ndvi_observado, params, rng):
    """Produtividade final da safra (1 valor por talhao/safra, repetido em
    todas as linhas), proporcional ao NDVI medio sobre o NDVI maximo da
    cultura, com +/-10% de variacao aleatoria (manejo, semente etc.)."""
    ndvi_medio_norm = ndvi_observado.mean() / params["ndvi_max"]
    fator = rng.uniform(0.9, 1.1)
    valor = round(params["produtividade_base_ton_ha"] * (0.5 + 0.5 * ndvi_medio_norm) * fator, 2)
    return np.full(len(ndvi_observado), valor)

# Falhas de sensores (dados ausentes)
COLUNAS_SENSOR = [
    "temperatura_ar_c",
    "umidade_ar_pct",
    "umidade_solo_sensor_pct",
    "luminosidade_lux",
]

def aplicar_falhas_sensores(df, rng):
    """Introduz NaN nas colunas de sensores ESP32 (falha de Wi-Fi/MQTT).
    Tratado depois em preprocessing.py."""
    for col in COLUNAS_SENSOR:
        mask = rng.random(len(df)) < config.PROB_FALHA_SENSOR
        df.loc[mask, col] = np.nan

# Geracao de uma serie (talhao, safra)
def gerar_serie_talhao_safra(talhao, safra, rng):
    cultura = talhao["cultura"]
    params = config.CULTURAS[cultura]
    ciclo = params["ciclo_dias"]

    plantio = data_plantio(safra, cultura, rng)
    datas = pd.date_range(start=plantio, periods=ciclo, freq="D")
    t = np.arange(ciclo)
    dia_juliano = datas.dayofyear.to_numpy()

    # NDVI esperado (fenologia) + evento de seca
    ndvi_esperado = curva_ndvi_esperada(t, params)
    intensidade_seca = gerar_intensidade_seca(ciclo, rng)
    ndvi_esperado_com_estresse = ndvi_esperado * (1 - intensidade_seca)
    ndvi_observado = np.clip(ndvi_esperado_com_estresse + rng.normal(0, 0.02, ciclo), 0, 1)

    # Clima
    temp_media_esperada = config.TEMP_MEDIA_ANUAL + config.TEMP_AMPLITUDE * np.cos(
        2 * np.pi * (dia_juliano - config.TEMP_PICO_DIA_JULIANO) / 365
    )
    temperatura_ar = temp_media_esperada + rng.normal(0, 1.5, ciclo)

    fator_chuva = 0.5 + 0.5 * np.cos(
        2 * np.pi * (dia_juliano - config.CHUVA_PICO_DIA_JULIANO) / 365
    )
    prob_chuva = config.CHUVA_PROB_BASE * fator_chuva
    prob_chuva = np.where(intensidade_seca > 0, prob_chuva * config.SECA_FATOR_CHUVA, prob_chuva)
    chove = rng.random(ciclo) < prob_chuva
    precipitacao = np.where(
        chove, rng.gamma(config.CHUVA_GAMMA_SHAPE, config.CHUVA_GAMMA_SCALE, ciclo), 0.0
    )

    umidade_ar = (
        55
        + 25 * fator_chuva
        - 0.6 * (temperatura_ar - temp_media_esperada)
        + rng.normal(0, 5, ciclo)
    )
    umidade_ar = np.clip(umidade_ar, 20, 98)

    # Umidade do solo (satelite e sensor)
    umidade_solo_satelite, umidade_solo_sensor = simular_umidade_solo(
        precipitacao, temperatura_ar, ndvi_observado, rng
    )

    # EVI e temperatura de superficie (LST)
    evi = np.clip(0.85 * ndvi_observado + rng.normal(0, 0.02, ciclo), 0, 1)
    temperatura_superficie = temperatura_ar + (4 - 9 * ndvi_observado) + rng.normal(0, 1, ciclo)

    # Luminosidade
    fator_luz = 0.65 + 0.35 * np.cos(2 * np.pi * (dia_juliano - 355) / 365)
    luminosidade = 60000 * fator_luz * (1 - 0.5 * np.clip(precipitacao / 30, 0, 1)) + rng.normal(
        0, 3000, ciclo
    )
    luminosidade = np.clip(luminosidade, 500, 100000)

    # Estagio fenologico
    frac_ciclo = t / ciclo
    estagio = np.array([nome_estagio(f) for f in frac_ciclo])

    # Variaveis-alvo
    necessidade_irrigacao = calcular_necessidade_irrigacao(
        umidade_solo_sensor, precipitacao, estagio, params, rng
    )
    status_saude = calcular_status_saude(ndvi_observado, ndvi_esperado)
    produtividade = calcular_produtividade(ndvi_observado, params, rng)

    df = pd.DataFrame(
        {
            "talhao_id": talhao["talhao_id"],
            "fazenda_id": talhao["fazenda_id"],
            "cultura": cultura,
            "area_ha": talhao["area_ha"],
            "latitude": talhao["latitude"],
            "longitude": talhao["longitude"],
            "safra": safra,
            "data": datas,
            "dia_juliano": dia_juliano,
            "dias_desde_plantio": t,
            "estagio_fenologico": estagio,
            "ndvi": np.round(ndvi_observado, 4),
            "evi": np.round(evi, 4),
            "temperatura_superficie_c": np.round(temperatura_superficie, 2),
            "umidade_solo_satelite_pct": np.round(umidade_solo_satelite, 2),
            "precipitacao_mm": np.round(precipitacao, 2),
            "temperatura_ar_c": np.round(temperatura_ar, 2),
            "umidade_ar_pct": np.round(umidade_ar, 2),
            "umidade_solo_sensor_pct": np.round(umidade_solo_sensor, 2),
            "luminosidade_lux": np.round(luminosidade, 1),
            "necessidade_irrigacao": necessidade_irrigacao,
            "status_saude": status_saude,
            "produtividade_estimada_ton_ha": produtividade,
        }
    )

    aplicar_falhas_sensores(df, rng)
    return df

# Orquestracao
def gerar_dataset():
    rng = np.random.default_rng(config.SEED)
    talhoes = gerar_talhoes(rng)

    blocos = []
    for _, talhao in talhoes.iterrows():
        for safra in config.SAFRAS:
            blocos.append(gerar_serie_talhao_safra(talhao, safra, rng))

    df = pd.concat(blocos, ignore_index=True)
    df.insert(0, "id", np.arange(1, len(df) + 1))
    return df

def main():
    df = gerar_dataset()
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_DATASET_PATH, index=False)
    print(f"Dataset sintetico gerado: {len(df)} registros, {df.shape[1]} colunas")
    print(f"Talhoes: {df['talhao_id'].nunique()} | Safras: {df['safra'].nunique()}")
    print(f"Salvo em: {config.RAW_DATASET_PATH}")
    return df

if __name__ == "__main__":
    main()
