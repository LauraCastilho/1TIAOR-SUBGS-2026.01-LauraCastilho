# `docs/`

Pasta de documentação e artefatos visuais do **AgroNexusSpace**.

## Subpastas

### `graficos/`

Gráficos gerados automaticamente pelo script de treinamento
(`python src/machine_learning/treinar_modelos.py`). Todos os arquivos são
PNG e são exibidos no dashboard (`src/dashboard/app.py`, aba **Métricas dos
Modelos**).

| Arquivo | Descrição |
|---|---|
| `correlacao_variaveis.png` | Heatmap de correlação de Pearson entre as principais variáveis numéricas (EDA). |
| `distribuicao_risco.png` | Distribuição das classes de `status_saude` e `necessidade_irrigacao` em treino e teste. |
| `matriz_confusao_status_saude_random_forest.png` | Matriz de confusão — Saúde da lavoura, Random Forest. |
| `matriz_confusao_status_saude_gradient_boosting.png` | Matriz de confusão — Saúde da lavoura, Gradient Boosting. |
| `matriz_confusao_status_saude_rna_numpy.png` | Matriz de confusão — Saúde da lavoura, RNA (NumPy). |
| `feature_importance_status_saude_random_forest.png` | Top-15 features mais importantes — Random Forest (saúde). |
| `curvas_treinamento_rna.png` | Curvas de perda e acurácia por época da RNA (NumPy) — treino vs. teste. |
| `comparacao_modelos_status_saude.png` | F1 macro comparado entre RF, GB e RNA para saúde da lavoura. |
| `matriz_confusao_irrigacao_random_forest.png` | Matriz de confusão — Irrigação, Random Forest. |
| `matriz_confusao_irrigacao_gradient_boosting.png` | Matriz de confusão — Irrigação, Gradient Boosting. |
| `feature_importance_irrigacao_random_forest.png` | Top-15 features mais importantes — Random Forest (irrigação). |
| `comparacao_modelos_irrigacao.png` | F1 macro comparado entre RF e GB para necessidade de irrigação. |
| `feature_importance_produtividade_random_forest.png` | Top-15 features mais importantes — Random Forest (produtividade). |
| `comparacao_modelos_produtividade.png` | R² comparado entre RF e XGBoost para produtividade. |

> Os gráficos são regenerados a cada execução de
> `python src/machine_learning/treinar_modelos.py` (saída determinística com
> seed 42).
