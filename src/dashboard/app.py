"""
AgroNexusSpace — Dashboard de Monitoramento Inteligente de Lavouras

Carrega os modelos treinados (src/machine_learning/models/) e o conjunto de
teste (data/processed/test.csv) e apresenta um painel interativo com:
  - Monitoramento por talhão (NDVI, umidade, precipitação, temperatura)
  - Predições dos modelos de ML (saúde, irrigação, produtividade)
  - Métricas e gráficos de avaliação dos modelos

Execução:
    streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ML_DIR = BASE_DIR / "src" / "machine_learning"
sys.path.insert(0, str(ML_DIR))

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as ml_config


# ── Configuração da página ────────────────────────────────────────────────────

st.set_page_config(
    page_title="AgroNexusSpace · Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Carregamento de dados e modelos (cacheados) ───────────────────────────────

@st.cache_data
def carregar_dados():
    df = pd.read_csv(ml_config.TEST_PATH, parse_dates=["data"])
    df = df.sort_values(["talhao_id", "safra", "data"]).reset_index(drop=True)
    return df


@st.cache_resource
def carregar_modelos():
    with open(ml_config.MODELS_DIR / "feature_list.json", encoding="utf-8") as f:
        features = json.load(f)

    modelos = {
        "saude": joblib.load(ml_config.MODELS_DIR / "status_saude_random_forest.joblib"),
        "irrigacao": joblib.load(ml_config.MODELS_DIR / "irrigacao_random_forest.joblib"),
        "produtividade": joblib.load(ml_config.MODELS_DIR / "produtividade_random_forest.joblib"),
    }

    with open(ml_config.METRICAS_PATH, encoding="utf-8") as f:
        metricas = json.load(f)

    return modelos, features, metricas


def prever(df, modelos, features):
    """Aplica os três modelos e devolve um DataFrame com colunas pred_*."""
    X = df[features].copy()
    df = df.copy()
    df["pred_saude"] = modelos["saude"].predict(X)
    df["pred_irrigacao"] = modelos["irrigacao"].predict(X)
    df["pred_produtividade"] = modelos["produtividade"].predict(X)
    return df


# ── Paleta / constantes visuais ───────────────────────────────────────────────

COR_SAUDE = {"saudavel": "#2ecc71", "atencao": "#f39c12", "critico": "#e74c3c"}
EMOJI_SAUDE = {"saudavel": "🟢", "atencao": "🟡", "critico": "🔴"}
LABEL_SAUDE = {"saudavel": "Saudável", "atencao": "Atenção", "critico": "Crítico"}
ORDEM_SAUDE = {"saudavel": 0, "atencao": 1, "critico": 2}


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(df):
    with st.sidebar:
        logo = BASE_DIR / "assets" / "logo-fiap.png"
        if logo.exists():
            st.image(str(logo), width=160)

        st.markdown("## 🛰️ AgroNexusSpace")
        st.markdown(
            "Monitoramento agrícola inteligente com dados de satélite "
            "e sensores IoT (ESP32)."
        )
        st.divider()

        talhoes = sorted(df["talhao_id"].unique())
        talhao = st.selectbox("Talhão", talhoes, help="Selecione o talhão a monitorar")

        safras_disponiveis = sorted(df[df["talhao_id"] == talhao]["safra"].unique())
        safra = st.selectbox("Safra", safras_disponiveis)

        st.divider()
        st.caption("RM568507 — Laura de Andrade Castilho")
        st.caption("FIAP 2026.1 · Sub GS")

    return talhao, safra


# ── Tab 1: Monitoramento ──────────────────────────────────────────────────────

def tab_monitoramento(df_sel, modelos, features):
    df_pred = prever(df_sel, modelos, features)
    atual = df_pred.iloc[-1]
    cultura = df_sel["cultura"].iloc[0]
    talhao = df_sel["talhao_id"].iloc[0]

    st.markdown(f"### 📍 Talhão **{talhao}** — Cultura: **{cultura}**")
    st.caption(
        f"Última observação disponível: **{atual['data'].strftime('%d/%m/%Y')}** "
        f"— Estágio fenológico: **{atual['estagio_fenologico']}** "
        f"— {len(df_sel)} dias de ciclo"
    )

    # ── KPI cards ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        delta_ndvi = float(atual["ndvi_tendencia_7d"])
        st.metric(
            label="NDVI atual",
            value=f"{atual['ndvi']:.3f}",
            delta=f"{delta_ndvi:+.3f} (7 d)",
            delta_color="normal",
        )

    with c2:
        status = atual["pred_saude"]
        st.metric(
            label="Saúde da Lavoura",
            value=f"{EMOJI_SAUDE[status]} {LABEL_SAUDE[status]}",
        )

    with c3:
        irrig = int(atual["pred_irrigacao"])
        st.metric(
            label="Irrigação Necessária",
            value="✅ Sim" if irrig == 1 else "❌ Não",
        )

    with c4:
        prod = float(atual["pred_produtividade"])
        st.metric(
            label="Produtividade Estimada",
            value=f"{prod:.2f} ton/ha",
        )

    st.divider()

    # ── NDVI com marcadores de saúde ─────────────────────────────────────────
    fig_ndvi = go.Figure()

    fig_ndvi.add_trace(go.Scatter(
        x=df_pred["data"], y=df_pred["ndvi"],
        mode="lines", name="NDVI",
        line=dict(color="#2ecc71", width=2),
    ))
    fig_ndvi.add_trace(go.Scatter(
        x=df_pred["data"], y=df_pred["ndvi_media_7d"],
        mode="lines", name="Média 7d",
        line=dict(color="#27ae60", width=1.5, dash="dash"),
    ))

    # Pontos coloridos por status de saúde
    for status, cor in COR_SAUDE.items():
        mask = df_pred["pred_saude"] == status
        if mask.any():
            fig_ndvi.add_trace(go.Scatter(
                x=df_pred.loc[mask, "data"],
                y=df_pred.loc[mask, "ndvi"],
                mode="markers",
                name=LABEL_SAUDE[status],
                marker=dict(color=cor, size=6, opacity=0.8),
                showlegend=True,
            ))

    fig_ndvi.update_layout(
        title="Evolução do NDVI ao longo do ciclo",
        xaxis_title="Data",
        yaxis_title="NDVI",
        height=300,
        margin=dict(t=45, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_ndvi, use_container_width=True)

    # ── Umidade e precipitação ────────────────────────────────────────────────
    col_esq, col_dir = st.columns(2)

    with col_esq:
        fig_umid = go.Figure()
        fig_umid.add_trace(go.Scatter(
            x=df_pred["data"], y=df_pred["umidade_solo_sensor_pct"],
            mode="lines", name="Sensor ESP32",
            line=dict(color="#3498db", width=2),
        ))
        fig_umid.add_trace(go.Scatter(
            x=df_pred["data"], y=df_pred["umidade_solo_satelite_pct"],
            mode="lines", name="Satélite (SMAP)",
            line=dict(color="#9b59b6", width=1.5, dash="dot"),
        ))

        irrig_dias = df_pred[df_pred["pred_irrigacao"] == 1]
        if not irrig_dias.empty:
            fig_umid.add_trace(go.Scatter(
                x=irrig_dias["data"],
                y=irrig_dias["umidade_solo_sensor_pct"],
                mode="markers", name="Irrigação necessária",
                marker=dict(color="#e74c3c", size=7, symbol="x"),
            ))

        fig_umid.update_layout(
            title="Umidade do Solo (%)",
            xaxis_title="Data",
            yaxis_title="Umidade (%)",
            height=280, margin=dict(t=45, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_umid, use_container_width=True)

    with col_dir:
        fig_chuva = make_subplots(specs=[[{"secondary_y": True}]])
        fig_chuva.add_trace(go.Bar(
            x=df_pred["data"], y=df_pred["precipitacao_mm"],
            name="Precipitação diária (mm)",
            marker_color="#74b9ff", opacity=0.7,
        ), secondary_y=False)
        fig_chuva.add_trace(go.Scatter(
            x=df_pred["data"], y=df_pred["precipitacao_acum_7d"],
            mode="lines", name="Acumulado 7d (mm)",
            line=dict(color="#0984e3", width=2),
        ), secondary_y=True)
        fig_chuva.update_layout(
            title="Precipitação (mm)",
            height=280, margin=dict(t=45, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_chuva.update_yaxes(title_text="mm/dia", secondary_y=False)
        fig_chuva.update_yaxes(title_text="mm acum. 7d", secondary_y=True)
        st.plotly_chart(fig_chuva, use_container_width=True)

    # ── Temperatura e luminosidade ────────────────────────────────────────────
    col_e2, col_d2 = st.columns(2)

    with col_e2:
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=df_pred["data"], y=df_pred["temperatura_ar_c"],
            mode="lines", name="Temperatura do ar (°C)",
            line=dict(color="#e17055"),
        ))
        fig_temp.add_trace(go.Scatter(
            x=df_pred["data"], y=df_pred["temperatura_superficie_c"],
            mode="lines", name="Temperatura sup. (LST, °C)",
            line=dict(color="#d63031", dash="dash"),
        ))
        fig_temp.update_layout(
            title="Temperatura (°C)",
            height=260, margin=dict(t=45, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with col_d2:
        fig_lux = go.Figure()
        fig_lux.add_trace(go.Scatter(
            x=df_pred["data"], y=df_pred["luminosidade_lux"],
            mode="lines", name="Luminosidade (lux)",
            fill="tozeroy", line=dict(color="#fdcb6e"),
        ))
        fig_lux.update_layout(
            title="Luminosidade — Sensor LDR (lux)",
            height=260, margin=dict(t=45, b=30),
        )
        st.plotly_chart(fig_lux, use_container_width=True)


# ── Tab 2: Predições ML ───────────────────────────────────────────────────────

def tab_predicoes(df_sel, modelos, features):
    df_pred = prever(df_sel, modelos, features)

    st.markdown("### Comparação: Predição do Modelo vs. Rótulo Real")
    st.caption(
        "Todos os talhões exibidos aqui são do **conjunto de teste** — "
        "nunca vistos durante o treinamento dos modelos."
    )

    # ── Status de Saúde ───────────────────────────────────────────────────────
    st.markdown("#### 1. Status de Saúde da Lavoura (multiclasse)")
    acc_saude = (df_pred["pred_saude"] == df_pred["status_saude"]).mean()
    st.metric("Acurácia no talhão/safra selecionado", f"{acc_saude:.1%}")

    df_pred["saude_real_num"] = df_pred["status_saude"].map(ORDEM_SAUDE)
    df_pred["saude_pred_num"] = df_pred["pred_saude"].map(ORDEM_SAUDE)

    fig_saude = go.Figure()
    fig_saude.add_trace(go.Scatter(
        x=df_pred["data"], y=df_pred["saude_real_num"],
        mode="lines+markers", name="Real",
        line=dict(color="#0984e3"), marker=dict(size=4),
        text=df_pred["status_saude"],
        hovertemplate="Data: %{x|%d/%m/%Y}<br>Real: %{text}<extra></extra>",
    ))
    fig_saude.add_trace(go.Scatter(
        x=df_pred["data"], y=df_pred["saude_pred_num"],
        mode="lines+markers", name="Predito (Random Forest)",
        line=dict(color="#e67e22", dash="dash"), marker=dict(size=4),
        text=df_pred["pred_saude"],
        hovertemplate="Data: %{x|%d/%m/%Y}<br>Predito: %{text}<extra></extra>",
    ))
    fig_saude.update_layout(
        yaxis=dict(tickvals=[0, 1, 2], ticktext=["Saudável", "Atenção", "Crítico"]),
        xaxis_title="Data",
        height=280, margin=dict(t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_saude, use_container_width=True)

    # ── Irrigação e Produtividade ─────────────────────────────────────────────
    col_irrig, col_prod = st.columns(2)

    with col_irrig:
        st.markdown("#### 2. Necessidade de Irrigação (binária)")
        acc_irrig = (df_pred["pred_irrigacao"] == df_pred["necessidade_irrigacao"]).mean()
        st.metric("Acurácia no talhão/safra selecionado", f"{acc_irrig:.1%}")

        fig_irrig = go.Figure()
        fig_irrig.add_trace(go.Scatter(
            x=df_pred["data"], y=df_pred["necessidade_irrigacao"],
            mode="lines", name="Real",
            line=dict(color="#0984e3", width=1.5),
        ))
        fig_irrig.add_trace(go.Scatter(
            x=df_pred["data"], y=df_pred["pred_irrigacao"],
            mode="lines", name="Predito",
            line=dict(color="#e74c3c", dash="dash", width=1.5),
        ))
        fig_irrig.update_layout(
            yaxis=dict(tickvals=[0, 1], ticktext=["Não", "Sim"]),
            xaxis_title="Data",
            height=260, margin=dict(t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_irrig, use_container_width=True)

    with col_prod:
        st.markdown("#### 3. Produtividade Estimada (regressão)")
        prod_real = float(df_pred["produtividade_estimada_ton_ha"].iloc[0])
        prod_pred_vals = df_pred["pred_produtividade"].unique()
        prod_pred = float(prod_pred_vals.mean())
        erro = abs(prod_real - prod_pred)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Real", f"{prod_real:.3f} ton/ha")
        with col_b:
            st.metric("Predito", f"{prod_pred:.3f} ton/ha")
        with col_c:
            st.metric("Erro absoluto", f"{erro:.3f} ton/ha")

        # Gauge
        cultura = df_sel["cultura"].iloc[0]
        max_prod = {"Soja": 5, "Milho": 9, "Algodao": 7, "Feijao": 3}.get(cultura, 8)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prod_pred,
            delta={"reference": prod_real, "valueformat": ".3f"},
            title={"text": f"Produtividade — {cultura} (ton/ha)"},
            gauge={
                "axis": {"range": [0, max_prod]},
                "bar": {"color": "#2ecc71"},
                "steps": [
                    {"range": [0, max_prod * 0.4], "color": "#fadbd8"},
                    {"range": [max_prod * 0.4, max_prod * 0.75], "color": "#fef9e7"},
                    {"range": [max_prod * 0.75, max_prod], "color": "#eafaf1"},
                ],
                "threshold": {
                    "line": {"color": "#e74c3c", "width": 3},
                    "thickness": 0.75,
                    "value": prod_real,
                },
            },
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=60, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)


# ── Tab 3: Métricas dos Modelos ───────────────────────────────────────────────

def tab_metricas(metricas):
    st.markdown("### Performance dos Modelos de Machine Learning")
    st.caption(
        "Avaliação no conjunto de teste — 4 talhões (920 linhas) nunca vistos "
        "durante o treinamento. Split por talhão (`GroupShuffleSplit`) para "
        "evitar data leakage temporal."
    )

    # ── Tabelas de métricas ───────────────────────────────────────────────────
    st.markdown("#### 1. Status de Saúde (classificação multiclasse)")
    rows = []
    for modelo, m in metricas["status_saude"].items():
        rows.append({
            "Modelo": modelo.replace("_", " ").title(),
            "Acurácia": f"{m['accuracy']:.1%}",
            "F1 Macro": f"{m['f1_macro']:.4f}",
            "F1 Ponderado": f"{m['f1_weighted']:.4f}",
            "Precisão Macro": f"{m['precision_macro']:.4f}",
            "Recall Macro": f"{m['recall_macro']:.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### 2. Necessidade de Irrigação (classificação binária)")
    rows = []
    for modelo, m in metricas["necessidade_irrigacao"].items():
        rows.append({
            "Modelo": modelo.replace("_", " ").title(),
            "Acurácia": f"{m['accuracy']:.1%}",
            "F1 Macro": f"{m['f1_macro']:.4f}",
            "Precisão Macro": f"{m['precision_macro']:.4f}",
            "Recall Macro": f"{m['recall_macro']:.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### 3. Produtividade Estimada (regressão)")
    rows = []
    for modelo, m in metricas["produtividade_estimada_ton_ha"].items():
        rows.append({
            "Modelo": modelo.replace("_", " ").title(),
            "R²": f"{m['r2']:.4f}",
            "MAE (ton/ha)": f"{m['mae']:.4f}",
            "RMSE (ton/ha)": f"{m['rmse']:.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Gráficos salvos ───────────────────────────────────────────────────────
    st.markdown("#### Gráficos de Avaliação (gerados pelo treino)")
    graficos_dir = BASE_DIR / "docs" / "graficos"

    pares = [
        ("Comparação de Modelos — Saúde",        "comparacao_modelos_status_saude.png"),
        ("Comparação de Modelos — Irrigação",     "comparacao_modelos_irrigacao.png"),
        ("Comparação de Modelos — Produtividade", "comparacao_modelos_produtividade.png"),
        ("Curvas de Treinamento da RNA (NumPy)",  "curvas_treinamento_rna.png"),
        ("Feature Importance — Saúde (RF)",       "feature_importance_status_saude_random_forest.png"),
        ("Feature Importance — Irrigação (RF)",   "feature_importance_irrigacao_random_forest.png"),
        ("Feature Importance — Produtividade (RF)", "feature_importance_produtividade_random_forest.png"),
        ("Distribuição das Classes",              "distribuicao_risco.png"),
        ("Correlação entre Variáveis (EDA)",      "correlacao_variaveis.png"),
        ("Matriz de Confusão — Saúde (RF)",       "matriz_confusao_status_saude_random_forest.png"),
        ("Matriz de Confusão — Saúde (GB)",       "matriz_confusao_status_saude_gradient_boosting.png"),
        ("Matriz de Confusão — Saúde (RNA)",      "matriz_confusao_status_saude_rna_numpy.png"),
    ]

    cols = st.columns(2)
    for i, (titulo, arquivo) in enumerate(pares):
        caminho = graficos_dir / arquivo
        if caminho.exists():
            with cols[i % 2]:
                st.image(str(caminho), caption=titulo, use_container_width=True)


# ── Tab 4: Sobre ──────────────────────────────────────────────────────────────

def tab_sobre():
    col_txt, col_arq = st.columns([2, 1])

    with col_txt:
        st.markdown("""
### AgroNexusSpace

**Monitoramento agrícola inteligente com dados espaciais e IoT**

O AgroNexusSpace é uma plataforma de monitoramento de lavouras que integra
dados de satélite (NDVI, EVI, LST, umidade orbital) com leituras de sensores
IoT (ESP32 + DHT22 + sensor capacitivo + LDR) e aplica Machine Learning para
gerar três saídas preditivas em tempo real:

| Tarefa | Modelos |
|---|---|
| Status de saúde da vegetação (saudável / atenção / crítico) | Random Forest, Gradient Boosting, RNA (NumPy) |
| Necessidade de irrigação (sim / não) | Random Forest, Gradient Boosting |
| Produtividade estimada (ton/ha) | Random Forest, XGBoost |

#### Contexto do Desafio

O projeto responde à pergunta central da Sub GS 2026.1 da FIAP:

> *"Como a Inteligência Artificial e as tecnologias digitais podem transformar
> a nova economia espacial e gerar impacto positivo na Terra?"*

Satélites como o **Landsat**, **Sentinel-2** e **SMAP** fornecem índices de
vegetação (NDVI/EVI), temperatura de superfície e umidade do solo com
cobertura global. Combinados com sensores de campo de baixo custo (ESP32),
essas fontes permitem monitorar dezenas de talhões sem inspeção manual.

#### Arquitetura

```
Satélite (NDVI · EVI · LST · SMAP) ──┐
                                       ├─► Pipeline de Dados ─► Modelos ML ─► Dashboard
Sensores ESP32 (DHT22 · cap. · LDR) ──┘        (este projeto)
```

#### Tecnologias

- **Python** (pandas, numpy, scikit-learn, xgboost)
- **RNA implementada do zero** em NumPy puro (sem TensorFlow/PyTorch)
- **Streamlit** · **Plotly** (este dashboard)
- Dados sintéticos com estrutura realista (Cerrado, GO/MG)
        """)

    with col_arq:
        st.markdown("#### Estrutura do repositório")
        st.code("""
1TIAOR-SUBGS-2026.01-LauraCastilho/
├── assets/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── graficos/
├── src/
│   ├── data_pipeline/
│   ├── dashboard/       ← você está aqui
│   └── machine_learning/
└── README.md
        """, language="")

        st.markdown("#### Resultados-chave")
        st.success("**Irrigação (RF):** 94,9 % acurácia")
        st.success("**Produtividade (RF):** R² = 0,989")
        st.warning("**Saúde (RF):** 86,6 % acurácia · F1 macro 0,718")
        st.info("**RNA (NumPy):** 66,3 % acurácia (dataset pequeno + desvio de distribuição)")

        st.divider()
        st.markdown(
            "**Aluna:** Laura de Andrade Castilho — RM568507  \n"
            "**Tutor:** Ana Cristina dos Santos  \n"
            "**Coordenador:** André Godoi Chiovato  \n"
            "**FIAP** 2026.1 — Sub GS"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = carregar_dados()
    modelos, features, metricas = carregar_modelos()

    talhao, safra = render_sidebar(df)

    df_sel = (
        df[(df["talhao_id"] == talhao) & (df["safra"] == safra)]
        .sort_values("data")
        .reset_index(drop=True)
    )

    if df_sel.empty:
        st.error(f"Sem dados para o talhão **{talhao}** na safra **{safra}**.")
        return

    st.title("🛰️ AgroNexusSpace — Dashboard de Monitoramento Inteligente")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌱 Monitoramento",
        "🤖 Predições ML",
        "📊 Métricas dos Modelos",
        "ℹ️ Sobre o Projeto",
    ])

    with tab1:
        tab_monitoramento(df_sel, modelos, features)
    with tab2:
        tab_predicoes(df_sel, modelos, features)
    with tab3:
        tab_metricas(metricas)
    with tab4:
        tab_sobre()


if __name__ == "__main__":
    main()
