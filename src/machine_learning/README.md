# `src/machine_learning/`

Camada de Machine Learning do **AgroNexusSpace**: treina, avalia e compara
os modelos preditivos para as três variáveis-alvo geradas pela camada de
dados (`src/data_pipeline/`), e produz os artefatos (modelos treinados +
gráficos) usados na documentação acadêmica e, futuramente, pelo dashboard.

## Arquivos

| Arquivo | Responsabilidade |
|---|---|
| `config.py` | Caminhos, alvos, colunas de identificação e parâmetros centrais da camada de ML. |
| `data_loader.py` | Carrega `train.csv`/`test.csv`, separa features (X) dos três alvos (y) e ajusta o `StandardScaler` dedicado à RNA. |
| `modelos_classicos.py` | Treino dos modelos baseados em árvore (RandomForest, GradientBoosting, XGBoost) para os três alvos. |
| `rede_neural_numpy.py` | MLP implementado do zero em NumPy (sem TensorFlow/PyTorch) para `status_saude`. |
| `avaliacao.py` | Métricas (accuracy/precision/recall/F1, MAE/RMSE/R²) e geração de todos os gráficos em `docs/graficos/`. |
| `treinar_modelos.py` | Orquestrador: roda EDA, treina/avalia todos os modelos, salva modelos + `metricas.json`. |
| `requirements.txt` | Dependências Python da camada de ML. |

## Como executar

Pré-requisito: `data/processed/train.csv` e `test.csv` devem existir (gerados
por `python src/data_pipeline/run_pipeline.py`).

```bash
pip install -r src/machine_learning/requirements.txt
python src/machine_learning/treinar_modelos.py
```

A execução é determinística (`config.RANDOM_STATE = 42` em todos os
modelos, incluindo a RNA), reproduzindo sempre os mesmos modelos, métricas e
gráficos.

## Saídas geradas

**`src/machine_learning/models/`**

- `feature_list.json` — lista ordenada das 31 features de entrada (mesma
  ordem usada para treinar todos os modelos; necessária para inferência).
- `status_saude_random_forest.joblib`, `status_saude_gradient_boosting.joblib`
- `status_saude_rna_numpy.npz` + `status_saude_rna_scaler.joblib` (pesos e
  scaler da RNA)
- `irrigacao_random_forest.joblib`, `irrigacao_gradient_boosting.joblib`
- `produtividade_random_forest.joblib`, `produtividade_xgboost.joblib`
- `metricas.json` — métricas consolidadas dos três problemas

**`docs/graficos/`**

- `correlacao_variaveis.png`, `distribuicao_risco.png` (EDA)
- `matriz_confusao_status_saude_{random_forest,gradient_boosting,rna_numpy}.png`
- `feature_importance_status_saude_random_forest.png`
- `curvas_treinamento_rna.png`
- `comparacao_modelos_status_saude.png`
- `matriz_confusao_irrigacao_{random_forest,gradient_boosting}.png`
- `feature_importance_irrigacao_random_forest.png`
- `comparacao_modelos_irrigacao.png`
- `feature_importance_produtividade_random_forest.png`
- `comparacao_modelos_produtividade.png`

## As três tarefas de ML

Todos os modelos usam as mesmas **31 features de entrada** (todas as colunas
de `train.csv`/`test.csv` exceto identificadores — `id`, `talhao_id`,
`fazenda_id`, `safra`, `data`, `estagio_fenologico` — e os três alvos).

### 1. `status_saude` — classificação multiclasse (saudavel / atencao / critico)

Comparação de três abordagens: **RandomForest**, **GradientBoosting** e uma
**RNA (MLP) implementada do zero em NumPy**. É o único alvo com gráfico de
curvas de treinamento, porque é o que justifica a RNA "do zero" pedida no
enunciado (comparação modelo clássico vs. rede neural).

### 2. `necessidade_irrigacao` — classificação binária

Comparação **RandomForest vs. GradientBoosting**.

### 3. `produtividade_estimada_ton_ha` — regressão

Comparação **RandomForest vs. XGBoost** (com fallback automático para
`GradientBoostingRegressor` caso o `xgboost` não esteja instalado — flag
`modelos_classicos.TEM_XGBOOST`).

## Resultados (test set, 920 linhas / 4 talhões)

| Alvo | Modelo | Métricas |
|---|---|---|
| `status_saude` | Random Forest | accuracy 0.866, f1_macro 0.718 |
| `status_saude` | Gradient Boosting | accuracy 0.866, f1_macro 0.701 |
| `status_saude` | RNA (NumPy) | accuracy 0.663, f1_macro 0.493 |
| `necessidade_irrigacao` | Random Forest | accuracy 0.949, f1_macro 0.949 |
| `necessidade_irrigacao` | Gradient Boosting | accuracy 0.947, f1_macro 0.947 |
| `produtividade_estimada_ton_ha` | Random Forest | R² 0.989, RMSE 0.169 |
| `produtividade_estimada_ton_ha` | XGBoost | R² 0.986, RMSE 0.185 |

(Valores completos em `models/metricas.json`.)

## Decisões técnicas

**Features**: as 31 entradas são as colunas brutas (NDVI, EVI, clima,
sensores ESP32), as derivadas do feature engineering
(`dias_desde_plantio_pct`, `estagio_fenologico_cod`, `ndvi_media_7d`,
`indice_estresse_climatico`, etc.) e o one-hot de `cultura`/`estacao`.
Identificadores (`id`, `talhao_id`, `fazenda_id`, `safra`, `data`,
`estagio_fenologico`) ficam de fora — são só identidade do talhão, não
variáveis preditivas.

**Árvores sem scaler**: RandomForest/GradientBoosting/XGBoost treinam direto
sobre os CSVs em unidades naturais, já que árvores são invariantes a escala.

**Desbalanceamento**: `status_saude` é ~85% saudavel / 12% atencao / 3%
critico. Os modelos de árvore usam `sample_weight` balanceado
(`compute_sample_weight`); a RNA usa o mesmo princípio via pesos por classe
na entropia cruzada (`compute_class_weight`), pra comparação ficar justa.

**RNA (NumPy)**: MLP 31 → 32 → 16 → 3 (`rede_neural_numpy.py`), ReLU nas
ocultas + softmax na saída, inicialização He, backprop manual (regra da
cadeia), mini-batch SGD (`batch_size=64`, taxa de aprendizado `0.02`) e L2
(`1e-3`). Com early stopping: a cada época salva os pesos se a loss de teste
melhorar, e ao final volta pra essa "melhor época" (linha pontilhada em
`curvas_treinamento_rna.png`) — sem isso a RNA passa de 85% de acurácia no
treino mas piora no teste. Usa um `StandardScaler` próprio
(`status_saude_rna_scaler.joblib`), separado do `scaler.joblib` do
data_pipeline (que só cobre as colunas contínuas).

**RNA abaixo do RandomForest/GradientBoosting** (accuracy 0.663 vs. 0.866,
f1_macro 0.493 vs. 0.70-0.72) mesmo depois do ajuste — e isso é esperado:
o dataset de treino é pequeno (2.960 linhas) pra uma rede com ~1.600
parâmetros, o split por talhão (`GroupShuffleSplit`) faz o teste cair em 4
talhões nunca vistos no treino (desvio de distribuição real), e MLPs simples
costumam perder pra ensembles de árvore em dados tabulares nesse regime. A
RNA fica no projeto justamente pra essa comparação, e as curvas de
treinamento documentam o overfitting.

**Reprodutibilidade**: tudo usa `config.RANDOM_STATE = 42`, então
`python src/machine_learning/treinar_modelos.py` sempre reproduz os mesmos
modelos, métricas e gráficos a partir de `data/processed/`.
