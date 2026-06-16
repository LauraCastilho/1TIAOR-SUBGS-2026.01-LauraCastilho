# Dicionário de Dados — AgroNexusSpace

Este documento descreve todas as colunas presentes em
`data/raw/dados_agricolas_sinteticos.csv` (dataset bruto) e em
`data/processed/train.csv` / `test.csv` (dataset após pré-processamento +
feature engineering).

Granularidade: **1 linha = 1 talhão, em 1 safra, em 1 dia** do ciclo
produtivo.

---

## 1. Identificação e contexto

| Coluna | Tipo | Unidade | Descrição | Regra de geração |
|---|---|---|---|---|
| `id` | int | — | Identificador sequencial único da linha. | Atribuído após concatenar todas as séries (1..N). |
| `talhao_id` | string | — | Identificador do talhão, formato `FXX-CCC` (fazenda + 3 letras da cultura). | Combinação de `fazenda_id` + prefixo da cultura. |
| `fazenda_id` | string | — | Identificador da fazenda (`F01`–`F04`). | 4 fazendas fixas; cada uma possui 1 talhão de cada cultura. |
| `cultura` | categórica | — | Cultura plantada no talhão: `Soja`, `Milho`, `Algodao`, `Feijao`. | Atribuída na criação do talhão (1 por cultura por fazenda). |
| `area_ha` | float | hectares | Área do talhão. | `uniform(40, 280)` por talhão (fixo entre safras). |
| `latitude` / `longitude` | float | graus decimais | Localização do talhão. | Centro da fazenda sorteado dentro do bounding box `LAT_MIN/MAX, LON_MIN/MAX` (região do Cerrado, GO/MG); cada talhão recebe um deslocamento aleatório de até ±0,05° em torno do centro da fazenda. |
| `safra` | string | — | Identificador da safra: `2023/2024` ou `2024/2025`. | 2 safras simuladas por talhão. |
| `data` | date | `AAAA-MM-DD` | Data da observação. | Sequência diária a partir da data de plantio (ver `dias_desde_plantio`). |
| `dia_juliano` | int | 1–365/366 | Dia do ano correspondente a `data`. | `data.dayofyear` — usado para modelar sazonalidade climática (chuva, temperatura, luminosidade). |
| `dias_desde_plantio` | int | dias | Dias transcorridos desde o plantio (0 = dia do plantio). | Índice sequencial `0..ciclo_dias-1`. |
| `estagio_fenologico` | categórica | — | Estágio do ciclo: `germinacao`, `vegetativo`, `floracao`, `frutificacao`, `maturacao`, `colheita`. | Determinado pela fração `dias_desde_plantio / ciclo_dias`, conforme faixas definidas em `config.ESTAGIOS_FENOLOGICOS`. |

**Datas de plantio**: cada cultura tem um deslocamento fixo (em dias) em
relação ao início da safra (Soja +30, Milho +40, Algodao +50, Feijao +20),
mais um *jitter* aleatório de ±5 dias por talhão — representando variação
real de calendário de plantio entre produtores.

**Ciclo produtivo (dias) por cultura**: Soja = 110, Milho = 140,
Algodao = 150, Feijao = 85.

---

## 2. Dados "espaciais" (inspirados em satélite)

| Coluna | Tipo | Unidade | Descrição | Regra de geração |
|---|---|---|---|---|
| `ndvi` | float | índice 0–1 | Normalized Difference Vegetation Index — vigor/cobertura da vegetação. | Curva dupla-logística de fenologia (cresce até o pico vegetativo e decai na senescência), multiplicada pelo fator de estresse hídrico (se houver evento de seca) e somada a ruído gaussiano `N(0, 0.02)`, limitada a `[0, 1]`. |
| `evi` | float | índice 0–1 | Enhanced Vegetation Index — similar ao NDVI, menos sensível à saturação em alta biomassa. | `0.85 * ndvi + N(0, 0.02)`, limitado a `[0, 1]` — altamente correlacionado ao NDVI, mas não redundante. |
| `temperatura_superficie_c` | float | °C | Land Surface Temperature (LST) — temperatura da superfície do talhão. | `temperatura_ar_c + (4 - 9 * ndvi) + N(0, 1)` — solo exposto (NDVI baixo) é mais quente que o ar; vegetação densa tende a ser mais fria. |
| `umidade_solo_satelite_pct` | float | % | Umidade do solo na zona de raízes (estimativa "orbital", ex. SMAP). | Modelo "balde com vazamento": `umidade[t] = 0.92*umidade[t-1] + 0.50*precipitacao[t] - ET[t]`, limitado a `[8, 95]`. Memória longa (decaimento lento). |
| `precipitacao_mm` | float | mm/dia | Precipitação diária. | Probabilidade de chuva sazonal `0.55 * fator_sazonal(dia_juliano)` (pico em meados de janeiro); se chove, volume `~ Gamma(2.0, 8.0)`. Durante eventos de seca, a probabilidade é multiplicada por `0.15`. |

---

## 3. Dados de sensores (ESP32)

| Coluna | Tipo | Unidade | Descrição | Regra de geração |
|---|---|---|---|---|
| `temperatura_ar_c` | float | °C | Temperatura do ar (sensor DHT22). | `24 + 6*cos(2π*(dia_juliano - 260)/365) + N(0, 1.5)` — média anual 24°C, amplitude 6°C, pico ~meados de setembro (pré-chuvas, padrão Cerrado). |
| `umidade_ar_pct` | float | % | Umidade relativa do ar (sensor DHT22). | `55 + 25*fator_sazonal_chuva - 0.6*(temp - temp_esperada) + N(0, 5)`, limitado a `[20, 98]`. |
| `umidade_solo_sensor_pct` | float | % | Umidade do solo na superfície (sensor capacitivo). | Mesmo modelo "balde com vazamento" da umidade de satélite, porém com decaimento mais rápido (`0.82`), maior infiltração (`0.65`) e ruído `N(0, 1.5)` — responde mais rápido e de forma mais ruidosa à chuva. Limitado a `[5, 100]`. |
| `luminosidade_lux` | float | lux | Luminosidade ambiente (sensor LDR). | `60000 * fator_sazonal_luz(dia_juliano) * (1 - 0.5*min(precipitacao/30, 1)) + N(0, 3000)`, limitado a `[500, 100000]` — dias mais longos no verão (solstício ~dia 355) e céu nublado (chuva) reduzem a luminosidade. |

**Falhas de transmissão**: cada uma das 4 colunas acima tem ~1,5% de
chance, por linha, de ser gravada como `NaN` no dataset bruto, simulando
falhas de Wi-Fi/MQTT do ESP32. Tratadas em `preprocessing.py`.

---

## 4. Variáveis-alvo (Machine Learning)

| Coluna | Tipo | Domínio | Descrição | Regra de geração |
|---|---|---|---|---|
| `necessidade_irrigacao` | int (binário) | `{0, 1}` | Indica se o talhão precisa de irrigação no dia. | `1` se `umidade_solo_sensor_pct < limiar_critico[cultura]` **E** `precipitacao_mm < 3` **E** estágio não é `maturacao`/`colheita`; caso contrário `0`. Limiares: Soja=35%, Milho=38%, Algodao=32%, Feijao=40%. Aplica-se ruído de rótulo de 4% (flip aleatório), simulando exceções/erros de registro. |
| `status_saude` | categórica | `saudavel`, `atencao`, `critico` | Classificação da saúde da vegetação no dia. | Baseado na razão `ndvi_observado / ndvi_esperado_sem_estresse`: `>= 0.92` → `saudavel`; `>= 0.75` → `atencao`; caso contrário → `critico`. Captura desvios do NDVI em relação ao esperado para o estágio fenológico (ex. durante eventos de seca). |
| `produtividade_estimada_ton_ha` | float | ton/ha | Produtividade estimada da safra (mesmo valor para todas as linhas do talhão/safra). | `produtividade_base[cultura] * (0.5 + 0.5 * (ndvi_medio_da_safra / ndvi_max[cultura])) * uniform(0.9, 1.1)`. Bases de referência: Soja=3.3, Milho=5.8, Algodao=4.0, Feijao=1.1 ton/ha. |

---

## 5. Features derivadas (feature engineering)

Geradas por `feature_engineering.py` a partir do dataset limpo. Presentes
apenas em `data/processed/train.csv` e `test.csv`.

### 5.1 Temporais e fenológicas

| Coluna | Tipo | Descrição |
|---|---|---|
| `mes` | int (1–12) | Mês de `data`. |
| `estacao_primavera`, `estacao_verao` | int (0/1) | One-hot da estação do ano derivada de `mes`. Apenas `outono`, `primavera` e `verao` ocorrem no dataset (o ciclo agrícola vai de set/out a mar/abr; `inverno` — jun/jul/ago — nunca ocorre). Baseline = `outono`, removido por `drop_first=True`. |
| `ciclo_total_dias` | int | Duração total do ciclo da cultura do talhão (lookup por `cultura`). |
| `dias_desde_plantio_pct` | float (0–1) | `dias_desde_plantio / ciclo_total_dias` — progresso do ciclo, comparável entre culturas de durações diferentes. |
| `estagio_fenologico_cod` | int (0–5) | Codificação ordinal de `estagio_fenologico`, seguindo a ordem natural do ciclo (`germinacao`=0 ... `colheita`=5). |

### 5.2 Tendência / memória temporal (por talhão-safra)

| Coluna | Tipo | Descrição |
|---|---|---|
| `ndvi_media_7d` | float | Média móvel de `ndvi` nos últimos 7 dias (janela mínima de 1). |
| `ndvi_tendencia_7d` | float | `ndvi - ndvi(7 dias atrás)` — indica se a vegetação está melhorando (>0) ou piorando (<0). |
| `precipitacao_acum_7d` | float | Soma de `precipitacao_mm` nos últimos 7 dias. |
| `umidade_solo_sensor_lag1` | float | `umidade_solo_sensor_pct` do dia anterior (primeiro dia da série repete o valor do próprio dia). |
| `umidade_solo_sensor_delta1` | float | Variação diária: `umidade_solo_sensor_pct - umidade_solo_sensor_lag1`. |

### 5.3 Interações climáticas

| Coluna | Tipo | Descrição |
|---|---|---|
| `amplitude_termica` | float | `temperatura_superficie_c - temperatura_ar_c` — proxy de estresse hídrico/cobertura vegetal (solo exposto tende a ter amplitude maior). |
| `indice_estresse_climatico` | float | `temperatura_ar_c * (100 - umidade_ar_pct) / 100` — combinação simples de calor + baixa umidade do ar. |
| `razao_umidade_sensor_satelite` | float | `umidade_solo_sensor_pct / umidade_solo_satelite_pct` — discrepância entre umidade de superfície (sensor) e de raízes (satélite); `1.0` quando o denominador é 0. |

### 5.4 Codificação categórica (one-hot, `drop_first=True`)

| Coluna | Tipo | Descrição |
|---|---|---|
| `cultura_Feijao`, `cultura_Milho`, `cultura_Soja` | int (0/1) | One-hot de `cultura` (baseline = `Algodao`, removido). |

> Os nomes de culturas (`Algodao`, `Feijao`) são gravados sem acento
> propositalmente, para evitar problemas de codificação em nomes de
> colunas/CSV em diferentes sistemas.

---

## 6. Colunas usadas para padronização (`scaler.joblib`)

O `StandardScaler` (ajustado apenas em `train.csv`) é aplicado às colunas
numéricas contínuas listadas em `COLUNAS_NUMERICAS_PARA_ESCALA`
(`split_dataset.py`): todas as colunas da seção 2, 3 e 5.1–5.3 que são
float/int contínuos (`area_ha`, `latitude`, `longitude`, `dia_juliano`,
`dias_desde_plantio`, `ndvi`, `evi`, `temperatura_superficie_c`,
`umidade_solo_satelite_pct`, `precipitacao_mm`, `temperatura_ar_c`,
`umidade_ar_pct`, `umidade_solo_sensor_pct`, `luminosidade_lux`,
`dias_desde_plantio_pct`, `ndvi_media_7d`, `ndvi_tendencia_7d`,
`precipitacao_acum_7d`, `umidade_solo_sensor_lag1`,
`umidade_solo_sensor_delta1`, `amplitude_termica`,
`indice_estresse_climatico`, `razao_umidade_sensor_satelite`).

Identificadores, datas, variáveis-alvo e colunas binárias/one-hot **não**
são escaladas.
