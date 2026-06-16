"""
Parametros do gerador de dados do AgroNexusSpace: regiao do Cerrado (GO/MG),
4 culturas (Soja, Milho, Algodao, Feijao) e 2 safras. Tudo que o pipeline usa
(gerador, pre-processamento, feature engineering, split) vem deste arquivo.
"""

from pathlib import Path

# Reprodutibilidade
SEED = 42

# Caminhos (relativos a raiz do projeto)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_DATASET_PATH = DATA_RAW_DIR / "dados_agricolas_sinteticos.csv"
TRAIN_DATASET_PATH = DATA_PROCESSED_DIR / "train.csv"
TEST_DATASET_PATH = DATA_PROCESSED_DIR / "test.csv"
SCALER_PATH = DATA_PROCESSED_DIR / "scaler.joblib"

# Talhoes / fazendas
FAZENDAS = ["F01", "F02", "F03", "F04"]
SAFRAS = ["2023/2024", "2024/2025"]

# Bounding box aproximado da regiao simulada (Cerrado - GO/MG, polo de graos)
LAT_MIN, LAT_MAX = -18.5, -16.0
LON_MIN, LON_MAX = -50.5, -47.5

# Datas de inicio de cada safra e deslocamento (em dias) do plantio de cada
# cultura em relacao ao inicio da safra. Um pequeno jitter aleatorio por
# talhao e somado a essa data base.
SAFRA_INICIO = {
    "2023/2024": "2023-09-15",
    "2024/2025": "2024-09-15",
}
CULTURA_OFFSET_PLANTIO_DIAS = {
    "Soja": 30,
    "Milho": 40,
    "Algodao": 50,
    "Feijao": 20,
}
JITTER_PLANTIO_DIAS = 5  # +/- dias de variacao aleatoria por talhao

# Parametros por cultura
# ciclo_dias: duracao do ciclo produtivo, do plantio a colheita
# ndvi_min/ndvi_max: faixa de NDVI esperada (solo exposto -> pico vegetativo)
# k1/k2: taxas de crescimento/senescencia da curva NDVI dupla-logistica
# t1_frac/t2_frac: posicao (fracao do ciclo) dos pontos de inflexao
# umidade_critica_pct: limiar de umidade do solo (sensor) que, combinado com
#                      ausencia de chuva, dispara necessidade de irrigacao
# produtividade_base_ton_ha: produtividade media de referencia (ton/ha)
CULTURAS = {
    "Soja": dict(
        ciclo_dias=110, ndvi_min=0.18, ndvi_max=0.88,
        k1=0.18, k2=0.18, t1_frac=0.28, t2_frac=0.78,
        umidade_critica_pct=35, produtividade_base_ton_ha=3.3,
    ),
    "Milho": dict(
        ciclo_dias=140, ndvi_min=0.18, ndvi_max=0.90,
        k1=0.15, k2=0.15, t1_frac=0.30, t2_frac=0.80,
        umidade_critica_pct=38, produtividade_base_ton_ha=5.8,
    ),
    "Algodao": dict(
        ciclo_dias=150, ndvi_min=0.18, ndvi_max=0.82,
        k1=0.12, k2=0.12, t1_frac=0.32, t2_frac=0.82,
        umidade_critica_pct=32, produtividade_base_ton_ha=4.0,
    ),
    "Feijao": dict(
        ciclo_dias=85, ndvi_min=0.18, ndvi_max=0.78,
        k1=0.22, k2=0.22, t1_frac=0.30, t2_frac=0.75,
        umidade_critica_pct=40, produtividade_base_ton_ha=1.1,
    ),
}

# Estagios fenologicos (fracao do ciclo -> rotulo)
# A ordem desta lista define tambem a codificacao ordinal usada no feature
# engineering (estagio_fenologico_cod).
ESTAGIOS_FENOLOGICOS = [
    ("germinacao", 0.00, 0.10),
    ("vegetativo", 0.10, 0.35),
    ("floracao", 0.35, 0.55),
    ("frutificacao", 0.55, 0.85),
    ("maturacao", 0.85, 0.97),
    ("colheita", 0.97, 1.01),
]

# Climatologia regional (Cerrado: chuvas concentradas entre out e mar)
TEMP_MEDIA_ANUAL = 24.0       # graus C
TEMP_AMPLITUDE = 6.0          # graus C
TEMP_PICO_DIA_JULIANO = 260   # ~meados de setembro (mais quente, pre-chuvas)

CHUVA_PROB_BASE = 0.55
CHUVA_PICO_DIA_JULIANO = 15   # ~meados de janeiro (auge da estacao chuvosa)
CHUVA_GAMMA_SHAPE = 2.0
CHUVA_GAMMA_SCALE = 8.0       # mm

# Evento de seca: probabilidade de um (talhao, safra) sofrer um periodo de
# estresse hidrico, e parametros desse periodo.
PROB_EVENTO_SECA = 0.30
SECA_DURACAO_MIN, SECA_DURACAO_MAX = 12, 25
SECA_FATOR_CHUVA = 0.15          # reduz a probabilidade de chuva durante a seca
SECA_REDUCAO_NDVI_MAX = 0.35     # reducao maxima do NDVI esperado (35%, seca severa)
SECA_DIAS_RECUPERACAO = 15       # dias apos a seca em que o efeito ainda decai

# Umidade do solo (modelo "balde com vazamento")
UMIDADE_INICIAL_PCT = 45.0
UMIDADE_DECAIMENTO_SATELITE = 0.92
UMIDADE_DECAIMENTO_SENSOR = 0.82
UMIDADE_INFILTRACAO_SATELITE = 0.50
UMIDADE_INFILTRACAO_SENSOR = 0.65
EVAPOTRANSPIRACAO_BASE = 1.2
EVAPOTRANSPIRACAO_TEMP = 0.12

# Ruido / falhas simuladas (realismo de dados de campo)
PROB_FALHA_SENSOR = 0.015          # fracao de leituras ESP32 ausentes (NaN)
PROB_RUIDO_LABEL_IRRIGACAO = 0.04  # ruido no rotulo de necessidade de irrigacao

# Feature engineering
JANELA_ROLLING_DIAS = 7

# Split treino/teste
TEST_SIZE = 0.2
