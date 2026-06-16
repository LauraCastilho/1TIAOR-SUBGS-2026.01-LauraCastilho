# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
<a href="https://www.fiap.com.br/">
  <img src="assets/logo-fiap.png" 
       alt="FIAP - Faculdade de Informática e Administração Paulista" 
       width="40%">
</a>
</p>

<br>

# AgroNexusSpace

## Nome do grupo
Projeto individual 

## 👨‍🎓 Integrante: 
- <a href="https://www.linkedin.com/in/laura-castilho/">Laura de Andrade Castilho - RM568507</a> 

## 👩‍🏫 Professores:
### Tutor(a) 
- <a href="https://www.linkedin.com/in/anacristinadossantos/">Ana Cristina dos Santos</a>
### Coordenador(a)
- <a href="https://www.linkedin.com/in/andregodoichiovato/">André Godoi Chiovato</a>


## 📜 Descrição

O AgroNexusSpace é o projeto que estou desenvolvendo para a Global Solution
2026.1: uma plataforma de monitoramento agrícola que combina dados "de
satélite", sensores IoT (ESP32) e Machine Learning para ajudar um produtor a
acompanhar a saúde da lavoura, prever a necessidade de irrigação e estimar a
produtividade da safra.

Como não tenho acesso a imagens reais de satélite nem a uma rede de sensores
física, a primeira parte do projeto é uma camada de dados (`src/data_pipeline/`) que gera um dataset sintético com
estrutura realista: 4 fazendas, 4 culturas (Soja, Milho, Algodão, Feijão) ao
longo de 2 safras, com curvas de NDVI/EVI baseadas em modelos de fenologia,
climatologia sazonal do Cerrado (região GO/MG), umidade do solo simulada por
um modelo de "balde com vazamento" (satélite vs. sensor) e até falhas de
sensor e ruído de rótulo, para se aproximar de dados de campo reais.

A partir desse dataset, a camada de Machine Learning (`src/machine_learning/`)
treina e compara modelos para as três tarefas centrais do projeto:

- **status_saude** da lavoura (saudável / atenção / crítico): RandomForest,
  GradientBoosting e uma rede neural (MLP) implementada em NumPy;
- **necessidade_irrigacao** (sim/não): RandomForest vs. GradientBoosting;
- **produtividade_estimada_ton_ha**: RandomForest vs. XGBoost.

Toda a comparação é documentada com gráficos (matriz de confusão,
importância de features, curvas de treinamento da RNA etc.) em
`docs/graficos/`, e os modelos treinados ficam salvos em
`src/machine_learning/models/`.


## 📁 Estrutura de pastas

```
.
├── assets/                   # logo FIAP usada neste README
├── data/
│   ├── raw/                  # dataset sintético bruto gerado pelo pipeline
│   ├── processed/            # train.csv, test.csv e scaler.joblib
│   └── dicionario_dados.md   # descrição completa de todas as colunas
├── docs/
│   └── graficos/             # gráficos gerados pela camada de ML (PNG)
├── src/
│   ├── data_pipeline/        # geração de dados sintéticos, pré-processamento, feature engineering e split
│   ├── dashboard/            # dashboard interativo Streamlit (monitoramento + predições)
│   └── machine_learning/     # treino e avaliação dos modelos (RF, GB, XGBoost, RNA em NumPy)
└── README.md
```


## 📎 Links e Observações

- <b>Vídeo demonstrativo</b>: [https://youtu.be/LINK_DO_VIDEO](https://youtu.be/LINK_DO_VIDEO) *(inserir antes da entrega final)*
- <b>Explicação de decisões técnicas</b>: ver os READMEs de
  `src/data_pipeline/`, `src/machine_learning/` e `src/dashboard/`, que
  detalham as escolhas de cada camada.
- <b>Observações Gerais</b>: projeto individual (RM568507), não vinculado a
  nenhuma competição externa.


## 🔧 Como executar o código

Pré-requisito: Python 3.12+ (testado com 3.12.10).

```bash
# 1. (opcional) crie e ative um ambiente virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. instale as dependências de cada camada
pip install -r src/data_pipeline/requirements.txt
pip install -r src/machine_learning/requirements.txt
pip install -r src/dashboard/requirements.txt

# 3. gere o dataset sintético e os conjuntos de treino/teste
python src/data_pipeline/run_pipeline.py

# 4. treine e avalie os modelos de ML
python src/machine_learning/treinar_modelos.py

# 5. abra o dashboard interativo
streamlit run src/dashboard/app.py
```

O passo 3 cria `data/raw/dados_agricolas_sinteticos.csv` e
`data/processed/{train,test}.csv` + `scaler.joblib`. O passo 4 usa esses
arquivos para treinar os modelos (salvos em `src/machine_learning/models/`) e
gerar os gráficos de avaliação em `docs/graficos/`. O passo 5 abre o
dashboard em `http://localhost:8501`, onde é possível selecionar o talhão e a
safra, ver o monitoramento em tempo real e comparar as predições dos modelos
com os dados reais. Toda a geração é determinística (seed 42), então rodar os
passos 3 e 4 novamente reproduz exatamente os mesmos resultados.


## 🗃 Histórico de lançamentos

* 0.4.0 - 16/06/2026
    * Dashboard Streamlit interativo: monitoramento por talhão/safra, gráficos
      de séries temporais (NDVI, umidade, precipitação, temperatura,
      luminosidade), predições dos três modelos em tempo real e exibição de
      métricas de avaliação.
* 0.3.0 - 16/06/2026
    * Camada de Machine Learning: Random Forest, Gradient Boosting, XGBoost e
      RNA implementada do zero em NumPy; gráficos de matriz de confusão,
      feature importance, curvas de treinamento e comparação de modelos.
* 0.2.0 - 16/06/2026
    * Camada de dados: gerador de dataset sintético, pré-processamento,
      feature engineering e split treino/teste por talhão.
* 0.1.0 - 15/06/2026
    * Estrutura inicial do repositório a partir do template.

---


## 📋 Licença

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/SabrinaOtoni/TEMPLATE-FIAP-GRAD-ON-IA">MODELO GIT FIAP</a> por <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://fiap.com.br">FIAP</a> está licenciado sobre <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution 4.0 International</a>.</p>
