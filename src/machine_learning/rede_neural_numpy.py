"""
MLP implementado do zero com NumPy (sem TensorFlow/PyTorch), usado para
prever status_saude e comparar com o RandomForest/GradientBoosting de
modelos_classicos.py.

Arquitetura: N camadas ocultas ReLU + saida softmax, treinada com gradiente
descendente em mini-lotes e backprop manual. O desbalanceamento de classes
entra como pesos por classe na entropia cruzada (mesma ideia do
sample_weight dos modelos classicos).
"""

import numpy as np


class MLPClassifier:
    def __init__(self, tamanho_entrada, tamanhos_ocultos, n_classes, taxa_aprendizado=0.05, l2=0.0, seed=42):
        self.taxa_aprendizado = taxa_aprendizado
        self.l2 = l2
        self._rng = np.random.default_rng(seed)

        camadas = [tamanho_entrada] + list(tamanhos_ocultos) + [n_classes]
        self.pesos = []
        self.bias = []
        for entrada, saida in zip(camadas[:-1], camadas[1:]):
            # Inicializacao He: adequada para ativacoes ReLU.
            limite = np.sqrt(2.0 / entrada)
            self.pesos.append(self._rng.normal(0, limite, size=(entrada, saida)))
            self.bias.append(np.zeros((1, saida)))

    # Funcoes de ativacao
    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _relu_derivada(z):
        return (z > 0).astype(z.dtype)

    @staticmethod
    def _softmax(z):
        z_deslocado = z - z.max(axis=1, keepdims=True)
        exp = np.exp(z_deslocado)
        return exp / exp.sum(axis=1, keepdims=True)

    # Forward / backward
    def _forward(self, X):
        ativacoes = [X]
        zs = []
        a = X
        n_camadas = len(self.pesos)
        for i, (W, b) in enumerate(zip(self.pesos, self.bias)):
            z = a @ W + b
            zs.append(z)
            a = self._relu(z) if i < n_camadas - 1 else self._softmax(z)
            ativacoes.append(a)
        return ativacoes, zs

    @staticmethod
    def _perda(y_prob, y_onehot, pesos_amostra=None):
        eps = 1e-12
        perdas = -np.sum(y_onehot * np.log(y_prob + eps), axis=1)
        if pesos_amostra is not None:
            return np.average(perdas, weights=pesos_amostra)
        return perdas.mean()

    def _backward(self, ativacoes, zs, y_onehot, pesos_amostra=None):
        n_camadas = len(self.pesos)
        grads_W = [None] * n_camadas
        grads_b = [None] * n_camadas

        # Gradiente da saida: para softmax + entropia cruzada, dL/dz = y_prob - y_onehot
        delta = ativacoes[-1] - y_onehot
        if pesos_amostra is not None:
            delta = delta * pesos_amostra[:, None]
            denom = pesos_amostra.sum()
        else:
            denom = ativacoes[0].shape[0]

        for i in reversed(range(n_camadas)):
            entrada_camada = ativacoes[i]
            grads_W[i] = entrada_camada.T @ delta / denom + self.l2 * self.pesos[i]
            grads_b[i] = delta.sum(axis=0, keepdims=True) / denom
            if i > 0:
                delta = (delta @ self.pesos[i].T) * self._relu_derivada(zs[i - 1])

        return grads_W, grads_b

    def _atualizar_pesos(self, grads_W, grads_b):
        for i in range(len(self.pesos)):
            self.pesos[i] -= self.taxa_aprendizado * grads_W[i]
            self.bias[i] -= self.taxa_aprendizado * grads_b[i]

    # API publica
    def prever_proba(self, X):
        ativacoes, _ = self._forward(X)
        return ativacoes[-1]

    def prever(self, X):
        return np.argmax(self.prever_proba(X), axis=1)

    def treinar(self, X_train, y_train, X_val, y_val, epochs=150, batch_size=64,
                 pesos_classe=None, verbose=True):
        """Treina por `epochs` epocas com gradiente descendente em
        mini-lotes. `y_train`/`y_val` sao vetores de inteiros (0..n_classes-1).
        `pesos_classe`: array de tamanho n_classes com o peso de cada classe
        na funcao de perda (para lidar com desbalanceamento)."""
        n_classes = self.pesos[-1].shape[1]
        y_train_oh = np.eye(n_classes)[y_train]
        y_val_oh = np.eye(n_classes)[y_val]

        n = X_train.shape[0]
        historico = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        melhor_val_loss = np.inf
        melhores_pesos, melhores_bias, melhor_epoca = None, None, 0

        for epoca in range(epochs):
            indices = self._rng.permutation(n)
            for inicio in range(0, n, batch_size):
                lote = indices[inicio:inicio + batch_size]
                X_lote = X_train[lote]
                y_lote_oh = y_train_oh[lote]
                pesos_lote = pesos_classe[y_train[lote]] if pesos_classe is not None else None

                ativacoes, zs = self._forward(X_lote)
                grads_W, grads_b = self._backward(ativacoes, zs, y_lote_oh, pesos_lote)
                self._atualizar_pesos(grads_W, grads_b)

            pesos_train = pesos_classe[y_train] if pesos_classe is not None else None
            pesos_val = pesos_classe[y_val] if pesos_classe is not None else None

            train_proba = self.prever_proba(X_train)
            val_proba = self.prever_proba(X_val)

            train_loss = self._perda(train_proba, y_train_oh, pesos_train)
            val_loss = self._perda(val_proba, y_val_oh, pesos_val)
            train_acc = float((train_proba.argmax(axis=1) == y_train).mean())
            val_acc = float((val_proba.argmax(axis=1) == y_val).mean())

            historico["train_loss"].append(float(train_loss))
            historico["val_loss"].append(float(val_loss))
            historico["train_acc"].append(train_acc)
            historico["val_acc"].append(val_acc)

            if val_loss < melhor_val_loss:
                melhor_val_loss = val_loss
                melhor_epoca = epoca + 1
                melhores_pesos = [W.copy() for W in self.pesos]
                melhores_bias = [b.copy() for b in self.bias]

            if verbose and (epoca == 0 or (epoca + 1) % 10 == 0 or epoca == epochs - 1):
                print(
                    f"  epoca {epoca + 1:3d}/{epochs} | "
                    f"loss treino={train_loss:.4f} acc treino={train_acc:.4f} | "
                    f"loss teste={val_loss:.4f} acc teste={val_acc:.4f}"
                )

        # Volta para os pesos da epoca com menor loss de teste (o ultimo
        # modelo tende a estar sobreajustado nesse dataset pequeno).
        self.pesos, self.bias = melhores_pesos, melhores_bias
        historico["melhor_epoca"] = melhor_epoca
        if verbose:
            print(f"  melhor epoca (menor loss teste): {melhor_epoca}")

        return historico

    # Persistencia
    def salvar(self, caminho):
        arrays = {"n_camadas": np.array(len(self.pesos)), "taxa_aprendizado": np.array(self.taxa_aprendizado)}
        for i, (W, b) in enumerate(zip(self.pesos, self.bias)):
            arrays[f"W{i}"] = W
            arrays[f"b{i}"] = b
        np.savez(caminho, **arrays)

    @classmethod
    def carregar(cls, caminho):
        dados = np.load(caminho)
        n_camadas = int(dados["n_camadas"])
        modelo = cls.__new__(cls)
        modelo.taxa_aprendizado = float(dados["taxa_aprendizado"])
        modelo._rng = np.random.default_rng(0)
        modelo.pesos = [dados[f"W{i}"] for i in range(n_camadas)]
        modelo.bias = [dados[f"b{i}"] for i in range(n_camadas)]
        return modelo
