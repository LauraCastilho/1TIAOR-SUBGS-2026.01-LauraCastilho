"""
Funcoes de avaliacao e geracao dos graficos da camada de Machine Learning.

Todas as funcoes de plot salvam o resultado em PNG (pasta
docs/graficos/) e fecham a figura, para que o script orquestrador possa
gerar varias dezenas de graficos sem acumular memoria.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

import matplotlib

matplotlib.use("Agg")  # backend sem interface grafica (execucao via terminal)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

def _salvar(fig, caminho):
    config.GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"  grafico salvo: {caminho.relative_to(config.BASE_DIR)}")

# EDA: correlacao e distribuicao das variaveis-alvo
def plot_correlacao(df, colunas, nome_arquivo="correlacao_variaveis.png"):
    """Heatmap de correlacao (Pearson) entre as principais variaveis
    numericas, para inspecionar relacoes entre clima, NDVI e umidade."""
    corr = df[colunas].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(colunas)))
    ax.set_xticklabels(colunas, rotation=90, fontsize=8)
    ax.set_yticks(range(len(colunas)))
    ax.set_yticklabels(colunas, fontsize=8)
    ax.set_title("Correlacao entre variaveis (treino)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlacao")
    _salvar(fig, config.GRAFICOS_DIR / nome_arquivo)

def plot_distribuicao_risco(treino, teste, nome_arquivo="distribuicao_risco.png"):
    """Distribuicao das classes de status_saude e necessidade_irrigacao em
    treino e teste, para visualizar o desbalanceamento das classes."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, (titulo, classes) in zip(
        axes,
        [
            ("status_saude", config.STATUS_SAUDE_CLASSES),
            ("necessidade_irrigacao", [0, 1]),
        ],
    ):
        prop_treino = treino[titulo].value_counts(normalize=True).reindex(classes, fill_value=0)
        prop_teste = teste[titulo].value_counts(normalize=True).reindex(classes, fill_value=0)

        x = np.arange(len(classes))
        largura = 0.35
        ax.bar(x - largura / 2, prop_treino.values, largura, label="treino")
        ax.bar(x + largura / 2, prop_teste.values, largura, label="teste")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in classes])
        ax.set_ylabel("proporcao")
        ax.set_title(f"Distribuicao de {titulo}")
        ax.legend()

    fig.suptitle("Distribuicao das variaveis de risco/saude")
    _salvar(fig, config.GRAFICOS_DIR / nome_arquivo)

# Classificacao
def avaliar_classificacao(y_true, y_pred, labels=None):
    """Retorna um dicionario com accuracy, precision/recall/f1 macro (e
    ponderado), adequado tanto para classificacao binaria quanto
    multiclasse."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4),
    }

def plot_matriz_confusao(y_true, y_pred, labels, titulo, nome_arquivo):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(titulo, fontsize=10)
    _salvar(fig, config.GRAFICOS_DIR / nome_arquivo)

# Regressao
def avaliar_regressao(y_true, y_pred):
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }

# Importancia de features e comparacao de modelos
def plot_feature_importance(importancias, nomes_features, titulo, nome_arquivo, top_n=15):
    ordem = np.argsort(importancias)[::-1][:top_n]
    nomes = [nomes_features[i] for i in ordem][::-1]
    valores = [importancias[i] for i in ordem][::-1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(nomes, valores, color="seagreen")
    ax.set_xlabel("importancia")
    ax.set_title(titulo)
    _salvar(fig, config.GRAFICOS_DIR / nome_arquivo)

def plot_comparacao_modelos(resultados, metrica, titulo, nome_arquivo, maior_melhor=True):
    """resultados: dict {nome_modelo: dict_de_metricas} (saida de
    avaliar_classificacao/avaliar_regressao)."""
    nomes = list(resultados.keys())
    valores = [resultados[n][metrica] for n in nomes]
    cores = ["seagreen" if maior_melhor else "indianred"] * len(nomes)

    fig, ax = plt.subplots(figsize=(6, 4))
    barras = ax.bar(nomes, valores, color="steelblue")
    for barra, valor in zip(barras, valores):
        ax.text(barra.get_x() + barra.get_width() / 2, valor, f"{valor:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(metrica)
    ax.set_title(titulo)
    _salvar(fig, config.GRAFICOS_DIR / nome_arquivo)

# Curvas de treinamento da RNA (NumPy)
def plot_curvas_treinamento(historico, titulo, nome_arquivo):
    """historico: dict com listas 'train_loss', 'val_loss', 'train_acc',
    'val_acc' (uma posicao por epoca) e, opcionalmente, 'melhor_epoca'
    (epoca com menor loss de teste, usada via early stopping)."""
    epocas = range(1, len(historico["train_loss"]) + 1)
    melhor_epoca = historico.get("melhor_epoca")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epocas, historico["train_loss"], label="treino")
    axes[0].plot(epocas, historico["val_loss"], label="teste")
    axes[0].set_xlabel("epoca")
    axes[0].set_ylabel("loss (entropia cruzada)")
    axes[0].set_title("Curva de perda")
    if melhor_epoca:
        axes[0].axvline(melhor_epoca, color="gray", linestyle="--", label="melhor epoca")
    axes[0].legend()

    axes[1].plot(epocas, historico["train_acc"], label="treino")
    axes[1].plot(epocas, historico["val_acc"], label="teste")
    axes[1].set_xlabel("epoca")
    axes[1].set_ylabel("acuracia")
    axes[1].set_title("Curva de acuracia")
    if melhor_epoca:
        axes[1].axvline(melhor_epoca, color="gray", linestyle="--", label="melhor epoca")
    axes[1].legend()

    fig.suptitle(titulo)
    _salvar(fig, config.GRAFICOS_DIR / nome_arquivo)
