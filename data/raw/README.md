# `data/raw/`

Contém o dataset bruto gerado pelo pipeline de dados do **AgroNexusSpace**,
antes de qualquer limpeza ou transformação.

## Arquivo

| Arquivo | Descrição |
|---|---|
| `dados_agricolas_sinteticos.csv` | Dataset sintético com 3.880 linhas (4 fazendas × 4 culturas × 2 safras × ciclo em dias) — gerado por `src/data_pipeline/generate_synthetic_data.py`. |

## Como gerar

```bash
python src/data_pipeline/run_pipeline.py
```

O script gera este arquivo automaticamente na primeira etapa do pipeline.

## Conteúdo do arquivo

| Grupo de colunas | Exemplos |
|---|---|
| Identificação | `id`, `talhao_id`, `fazenda_id`, `cultura`, `safra`, `data` |
| Fenologia / ciclo | `dias_desde_plantio`, `estagio_fenologico`, `dia_juliano` |
| Dados espaciais (satélite) | `ndvi`, `evi`, `temperatura_superficie_c`, `umidade_solo_satelite_pct`, `precipitacao_mm` |
| Sensores ESP32 | `temperatura_ar_c`, `umidade_ar_pct`, `umidade_solo_sensor_pct`, `luminosidade_lux` |
| Variáveis-alvo | `status_saude`, `necessidade_irrigacao`, `produtividade_estimada_ton_ha` |

Aproximadamente 1,5 % das leituras de cada coluna ESP32 são `NaN`, simulando
falhas de Wi-Fi/MQTT do dispositivo. Essas ausências são tratadas na etapa de
pré-processamento (`src/data_pipeline/preprocessing.py`).

Ver `data/dicionario_dados.md` para a descrição detalhada de cada coluna,
incluindo as regras de geração e as unidades.
