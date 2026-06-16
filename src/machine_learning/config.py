"""
Parametros centrais da camada de Machine Learning do AgroNexusSpace.

Le os datasets ja prontos em data/processed/ (gerados pelo
src/data_pipeline/) e define onde salvar modelos treinados, metricas e
graficos.
"""

from pathlib import Path

# Reprodutibilidade
RANDOM_STATE = 42

# Caminhos
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
TRAIN_PATH = DATA_PROCESSED_DIR / "train.csv"
TEST_PATH = DATA_PROCESSED_DIR / "test.csv"

ML_DIR = Path(__file__).resolve().parent
MODELS_DIR = ML_DIR / "models"
GRAFICOS_DIR = BASE_DIR / "docs" / "graficos"
METRICAS_PATH = MODELS_DIR / "metricas.json"

# Colunas
# Colunas que identificam a linha (nao sao features de entrada dos modelos).
COLUNAS_ID = ["id", "talhao_id", "fazenda_id", "safra", "data", "estagio_fenologico"]

# As tres variaveis-alvo geradas pela camada de dados.
COL_TARGET_IRRIGACAO = "necessidade_irrigacao"
COL_TARGET_SAUDE = "status_saude"
COL_TARGET_PRODUTIVIDADE = "produtividade_estimada_ton_ha"

COLUNAS_TARGET = [COL_TARGET_IRRIGACAO, COL_TARGET_SAUDE, COL_TARGET_PRODUTIVIDADE]

# Classes do alvo de saude/risco, em ordem de severidade. Usadas para fixar a
# ordem das matrizes de confusao e para codificar os rotulos da RNA (NumPy).
STATUS_SAUDE_CLASSES = ["saudavel", "atencao", "critico"]
