"""
Orquestra a camada de dados do AgroNexusSpace de ponta a ponta:

    1. Gera o dataset sintetico        -> data/raw/dados_agricolas_sinteticos.csv
    2. Aplica o pre-processamento (limpeza)
    3. Aplica o feature engineering
    4. Separa treino/teste e ajusta o scaler
    5. Salva os datasets processados   -> data/processed/{train,test}.csv
                                           data/processed/scaler.joblib

Uso (a partir da raiz do projeto ou de qualquer diretorio):
    python src/data_pipeline/run_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_synthetic_data as gerador
import preprocessing as prep
import feature_engineering as feateng
import split_dataset as split


def main():
    print("=== 1/4: Gerando dataset sintetico ===")
    df_bruto = gerador.main()

    print("\n=== 2/4: Pre-processamento ===")
    df_limpo = prep.preprocessar(df_bruto)

    print("\n=== 3/4: Feature engineering ===")
    df_features = feateng.aplicar_feature_engineering(df_limpo)
    print(f"Linhas: {df_features.shape[0]} | Colunas: {df_features.shape[1]}")

    print("\n=== 4/4: Split treino/teste e salvamento ===")
    split.processar_split(df_features)

    print("\nPipeline concluido com sucesso.")


if __name__ == "__main__":
    main()
