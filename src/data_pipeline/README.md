# `src/data_pipeline/`

Camada de geração e preparação de dados do **AgroNexusSpace**: produz um
dataset sintético com estrutura realista de monitoramento agrícola por satélite
e sensores ESP32, aplica limpeza, feature engineering e separa treino/teste.

## Arquivos

| Arquivo | Responsabilidade |
|---|---|
| `config.py` | Todos os parâmetros do pipeline: regiões geográficas, culturas, safras, climatologia, falhas de sensor e caminhos de saída. |
| `generate_synthetic_data.py` | Gera o dataset bruto linha a linha: fenologia (NDVI/EVI dupla-logística), climatologia sazonal do Cerrado, modelo de umidade "balde com vazamento", eventos de seca aleatórios, falhas de sensor (~1,5 % de NaN) e rótulos com ruído. |
| `preprocessing.py` | Limpeza: imputa NaN dos sensores ESP32 por interpolação linear intra-talhão/safra, trata valores fora dos limites físicos e padroniza tipos. |
| `feature_engineering.py` | Cria as features derivadas: `mes`, one-hot de estação, `dias_desde_plantio_pct`, `estagio_fenologico_cod`, médias móveis de NDVI e precipitação (7 d), lag/delta de umidade, interações climáticas (`amplitude_termica`, `indice_estresse_climatico`, `razao_umidade_sensor_satelite`) e one-hot de `cultura`. |
| `split_dataset.py` | Split por grupo (`GroupShuffleSplit` em `talhao_id`): garante que nenhum talhão apareça em treino e teste ao mesmo tempo — evita vazamento temporal. Ajusta e salva o `StandardScaler` apenas com dados de treino. |
| `run_pipeline.py` | Orquestrador: executa os quatro passos acima em sequência e salva os artefatos em `data/`. |
| `requirements.txt` | Dependências Python da camada de dados. |

## Como executar

```bash
pip install -r src/data_pipeline/requirements.txt
python src/data_pipeline/run_pipeline.py
```

A execução é determinística (`config.SEED = 42`), sempre gerando os mesmos
datasets a partir dos mesmos parâmetros.

## Saídas geradas

| Caminho | Descrição |
|---|---|
| `data/raw/dados_agricolas_sinteticos.csv` | Dataset bruto gerado (antes de qualquer limpeza). |
| `data/processed/train.csv` | Conjunto de treino (80 % dos talhões, com features derivadas). |
| `data/processed/test.csv` | Conjunto de teste (20 % dos talhões, nunca vistos no treino). |
| `data/processed/scaler.joblib` | `StandardScaler` ajustado apenas no treino, para uso pela RNA. |

## Estrutura do dataset

**Granularidade:** 1 linha = 1 talhão × 1 safra × 1 dia do ciclo produtivo.

| Grupo de colunas | Exemplos | Fonte simulada |
|---|---|---|
| Identificação | `talhao_id`, `fazenda_id`, `cultura`, `safra`, `data` | Gerado |
| Dados espaciais / satélite | `ndvi`, `evi`, `temperatura_superficie_c`, `umidade_solo_satelite_pct`, `precipitacao_mm` | Modelos físicos (fenologia + climatologia) |
| Sensores ESP32 | `temperatura_ar_c`, `umidade_ar_pct`, `umidade_solo_sensor_pct`, `luminosidade_lux` | Simulação com falhas (~1,5 % NaN) |
| Variáveis-alvo (ML) | `status_saude`, `necessidade_irrigacao`, `produtividade_estimada_ton_ha` | Derivado de regras + ruído de rótulo |
| Features derivadas | `ndvi_media_7d`, `indice_estresse_climatico`, `estagio_fenologico_cod`, ... | Feature engineering |

Ver `data/dicionario_dados.md` para a descrição completa de cada coluna.

## Configuração por cultura

| Cultura | Ciclo (dias) | NDVI máx. | Limiar irrigação | Produt. base (ton/ha) |
|---|---|---|---|---|
| Soja | 110 | 0,88 | 35 % umidade | 3,3 |
| Milho | 140 | 0,90 | 38 % umidade | 5,8 |
| Algodão | 150 | 0,82 | 32 % umidade | 4,0 |
| Feijão | 85 | 0,78 | 40 % umidade | 1,1 |

## Decisões técnicas

**Split por talhão (GroupShuffleSplit):** o dataset é um painel temporal —
vários dias do mesmo talhão estão correlacionados. Um split aleatório por
linha colocaria dias quase idênticos do mesmo talhão em treino e teste
(data leakage). Separando por `talhao_id`, o modelo é avaliado em talhões
completamente desconhecidos.

**Evento de seca:** 30 % dos talhões/safras sofrem um período de seca de 12–25
dias, com redução de até 35 % no NDVI esperado e probabilidade de chuva
multiplicada por 0,15. Os 15 dias seguintes ao fim da seca ainda têm efeito
residual decrescente — modelando a recuperação lenta da vegetação.

**Falhas de sensor:** cada coluna ESP32 tem ~1,5 % de NaN por linha, simulando
quedas de Wi-Fi/MQTT. O `preprocessing.py` as imputa por interpolação linear
dentro do grupo (talhão, safra), preservando a tendência temporal.

**Scaler:** o `StandardScaler` é ajustado apenas em `train.csv` e salvo em
`data/processed/scaler.joblib`. É usado exclusivamente pela RNA
(`status_saude_rna_scaler.joblib`), que o refaz a partir do subconjunto de
treino para não vazar informação do teste.
