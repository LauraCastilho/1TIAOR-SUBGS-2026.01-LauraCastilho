# `src/dashboard/`

Dashboard interativo do **AgroNexusSpace**: carrega os modelos treinados pela
camada de ML e o conjunto de teste e apresenta um painel de monitoramento
agrícola com predições em tempo real, gráficos de séries temporais e métricas
dos modelos.

## Arquivos

| Arquivo | Responsabilidade |
|---|---|
| `app.py` | Aplicação Streamlit completa: sidebar, quatro abas (Monitoramento, Predições ML, Métricas, Sobre). |
| `requirements.txt` | Dependências Python do dashboard (Streamlit, Plotly, scikit-learn, etc.). |

## Como executar

Pré-requisito: os modelos devem estar treinados
(`python src/machine_learning/treinar_modelos.py`) e os dados processados
(`python src/data_pipeline/run_pipeline.py`).

```bash
pip install -r src/dashboard/requirements.txt
streamlit run src/dashboard/app.py
```

O navegador abrirá automaticamente em `http://localhost:8501`.

## Funcionalidades

### Sidebar
- Seleção de **talhão** e **safra** (filtro dos dados de teste)
- Logo FIAP + identificação do projeto

### Aba 🌱 Monitoramento
- **4 KPI cards**: NDVI atual (com delta 7 dias), status de saúde, irrigação
  necessária, produtividade estimada
- **Gráfico de NDVI** ao longo do ciclo, com pontos coloridos pelo status de
  saúde predito (verde = saudável, amarelo = atenção, vermelho = crítico)
- **Umidade do solo** — sensor ESP32 vs. estimativa orbital (SMAP), com marcadores
  nos dias em que o modelo detecta necessidade de irrigação
- **Precipitação diária** (barras) + acumulado de 7 dias (linha), eixo duplo
- **Temperatura** (ar vs. superfície LST) e **luminosidade** (sensor LDR)

### Aba 🤖 Predições ML
- Comparação direta entre o rótulo **real** e a **predição do modelo** para
  cada observação do talhão/safra selecionado
- Acurácia local (para o talhão) nos três alvos
- Gauge de produtividade com referência ao valor real

### Aba 📊 Métricas dos Modelos
- Tabelas de métricas do conjunto de teste para os três problemas (classificação
  multiclasse, binária e regressão)
- Exibição dos 12 gráficos gerados pelo script de treino (matrizes de confusão,
  feature importance, curvas de treinamento da RNA, EDA)

### Aba ℹ️ Sobre o Projeto
- Descrição do AgroNexusSpace, contexto do desafio FIAP Sub GS 2026.1,
  arquitetura da solução e resultados-chave

## Estrutura de caminhos

O dashboard resolve todos os caminhos relativos à raiz do projeto
(`BASE_DIR = Path(__file__).resolve().parents[2]`) e importa `config.py` da
camada de ML para reutilizar as constantes de caminhos de modelos e dados.
Não há hardcode de caminhos absolutos.
