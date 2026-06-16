# `data/processed/`

Contém os artefatos produzidos pelas etapas de pré-processamento, feature
engineering e split do **AgroNexusSpace** — prontos para o treinamento e
avaliação dos modelos de Machine Learning.

## Arquivos

| Arquivo | Descrição |
|---|---|
| `train.csv` | Conjunto de treino: 80 % dos talhões (~2.960 linhas, 12 talhões), com todas as features derivadas. |
| `test.csv` | Conjunto de teste: 20 % dos talhões (~920 linhas, 4 talhões nunca vistos no treino). |
| `scaler.joblib` | `StandardScaler` ajustado exclusivamente nos dados de treino (sem vazamento do teste). |

## Como gerar

```bash
python src/data_pipeline/run_pipeline.py
```

## Diferenças em relação ao dataset bruto (`data/raw/`)

1. **Limpeza**: NaN das colunas ESP32 imputados por interpolação linear
   intra-talhão/safra; valores fora dos limites físicos corrigidos.
2. **Feature engineering**: adicionadas 15+ colunas derivadas (médias móveis,
   lag, one-hot, índices climáticos, codificação fenológica — ver
   `data/dicionario_dados.md`, seção 5).
3. **Split por talhão**: `GroupShuffleSplit(test_size=0.2, random_state=42)`
   garante que nenhum talhão apareça em ambos os conjuntos (evita data
   leakage temporal).

## Importante

- Nunca use `scaler.joblib` para transformar dados de teste antes de ajustá-lo
  no treino: ele já foi ajustado apenas em `train.csv`.
- A lista de features de entrada dos modelos está em
  `src/machine_learning/models/feature_list.json` — use essa lista para
  garantir a ordem correta das colunas na inferência.
