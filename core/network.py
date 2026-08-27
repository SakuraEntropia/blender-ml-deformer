# Copyright (c) 2026 Blender ML Deformer contributors.
# Licensed under the MIT License. See LICENSE in the project root.

"""Feed-forward neural network (MLP) with Adam, input standardization and
selectable activation functions. Pure numpy.

The network maps a feature vector (F,) to morph weights (M,).  The final
per-vertex delta is reconstructed as ``morph_deltas @ weights`` by the
caller (``NeuralMorphRegressor`` below).

Activation ids: "relu", "elu", "tanh".  The engine-side exchange format we
interoperate with uses ELU after every linear layer, so "elu" is the default
for exported networks.
"""

from __future__ import annotations

import numpy as np

_ACTIVATIONS = {
    "relu": lambda x: np.maximum(x, 0.0),
    "elu": lambda x: np.where(x > 0.0, x, np.exp(x) - 1.0),
    "tanh": np.tanh,
}


class FeedForwardNet:
    """Simple FC network: hidden layers + linear output."""

    def __init__(self, sizes, activation="elu", rng=None, params=None):
        self.sizes = [int(s) for s in sizes]
        if activation not in _ACTIVATIONS:
            raise ValueError("unknown activation %r" % activation)
        self.activation = activation
        self.weights = []
        self.biases = []
        if params is not None:
            for k in range(len(self.sizes) - 1):
                self.weights.append(np.asarray(params["layer_%d_weights" % k],
                                               dtype=np.float64))
                self.biases.append(np.asarray(params["layer_%d_biases" % k],
                                              dtype=np.float64))
        else:
            if rng is None:
                rng = np.random.default_rng()
            for k in range(len(self.sizes) - 1):
                bound = np.sqrt(6.0 / (self.sizes[k] + self.sizes[k + 1]))
                self.weights.append(rng.uniform(-bound, bound,
                                                (self.sizes[k + 1], self.sizes[k])))
                self.biases.append(np.zeros(self.sizes[k + 1]))

    def forward(self, X):
        """X: (F, N) -> (O, N)."""
        act = _ACTIVATIONS[self.activation]
        A = np.asarray(X, dtype=np.float64)
        for k in range(len(self.weights) - 1):
            A = act(self.weights[k] @ A + self.biases[k][:, None])
        return self.weights[-1] @ A + self.biases[-1][:, None]

    def param_dict(self):
        out = {"sizes": np.asarray(self.sizes, dtype=np.int64),
               "activation": np.asarray([self.activation])}
        for k, (w, b) in enumerate(zip(self.weights, self.biases)):
            out["layer_%d_weights" % k] = w
            out["layer_%d_biases" % k] = b
        return out


class _Adam:
    def __init__(self, params):
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads, lr):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * (g * g)
            m_hat = self.m[i] / (1.0 - 0.9 ** self.t)
            v_hat = self.v[i] / (1.0 - 0.999 ** self.t)
            p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)


class NeuralMorphRegressor:
    """Features (F,) -> morph weights (M,) -> per-vertex deltas (3V,)."""

    def __init__(self, num_features, morph_deltas, hidden_sizes,
                 activation="relu", rng=None, params=None):
        self.num_features = int(num_features)
        self.morph_deltas = np.asarray(morph_deltas, dtype=np.float64)  # (3V, M)
        self.hidden_sizes = [int(s) for s in hidden_sizes]
        self.net = FeedForwardNet([self.num_features] + self.hidden_sizes
                                  + [self.morph_deltas.shape[1]],
                                  activation=activation, rng=rng, params=params)
        if params is not None and "input_mean" in params:
            self.input_mean = np.asarray(params["input_mean"], dtype=np.float64)
            self.input_std = np.asarray(params["input_std"], dtype=np.float64)
        else:
            self.input_mean = np.zeros(self.num_features)
            self.input_std = np.ones(self.num_features)
        self.clamp_weights = True

    def _standardize(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.input_mean[:, None]) / self.input_std[:, None]

    def predict_weights(self, X):
        return self.net.forward(self._standardize(X))  # (M, N)

    def predict(self, X):
        w = self.predict_weights(X)
        if self.clamp_weights:
            w = np.clip(w, 0.0, 1.0)
        return self.morph_deltas @ w  # (3V, N)

    def fit_iter(self, X, Y, iterations=3000, batch_size=64, learning_rate=1e-3,
                 regularization=1e-4, clamp=True, chunk=None, seed=None):
        """Generator yielding (progress, batch_loss) every ``chunk`` steps."""
        self.clamp_weights = bool(clamp)
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        self.input_mean = X.mean(axis=1)
        self.input_std = X.std(axis=1)
        self.input_std[self.input_std < 1e-6] = 1.0
        X = self._standardize(X)
        n_samples = X.shape[1]
        if batch_size > n_samples:
            batch_size = n_samples
        rng = np.random.default_rng(seed)
        params = self.net.weights + self.net.biases
        adam = _Adam(params)
        act = _ACTIVATIONS[self.net.activation]
        if chunk is None:
            chunk = max(1, iterations // 100)

        def _step(Xb, Yb):
            A = Xb
            acts = [A]
            pre = []
            for k in range(len(self.net.weights) - 1):
                z = self.net.weights[k] @ A + self.net.biases[k][:, None]
                pre.append(z)
                A = act(z)
                acts.append(A)
            z = self.net.weights[-1] @ A + self.net.biases[-1][:, None]
            pre.append(z)
            out = z
            d = (out - Yb) * (2.0 / batch_size)
            grads_w = [None] * len(self.net.weights)
            grads_b = [None] * len(self.net.biases)
            for k in range(len(self.net.weights) - 1, -1, -1):
                grads_w[k] = d @ acts[k].T + 2.0 * regularization * self.net.weights[k]
                grads_b[k] = d.sum(axis=1)
                if k > 0:
                    zprev = pre[k - 1]
                    if self.net.activation == "relu":
                        d = (self.net.weights[k].T @ d) * (zprev > 0.0)
                    elif self.net.activation == "elu":
                        d = (self.net.weights[k].T @ d) * np.where(zprev > 0.0, 1.0,
                                                                   np.exp(zprev))
                    else:  # tanh
                        d = (self.net.weights[k].T @ d) * (1.0 - np.tanh(zprev) ** 2)
            return grads_w, grads_b, out

        loss = float("inf")
        for it in range(iterations):
            idx = rng.integers(0, n_samples, size=batch_size)
            grads_w, grads_b, out = _step(X[:, idx], Y[:, idx])
            adam.step(params, grads_w + grads_b, learning_rate)
            if (it + 1) % chunk == 0 or it == iterations - 1:
                loss = float(np.mean((out - Y[:, idx]) ** 2))
                yield (it + 1) / iterations, loss
        yield 1.0, loss

    def fit(self, X, Y, **kwargs):
        loss = float("inf")
        for _frac, loss in self.fit_iter(X, Y, **kwargs):
            pass
        return loss

    def param_dict(self):
        out = {"morph_deltas": self.morph_deltas,
               "hidden_sizes": np.asarray(self.hidden_sizes, dtype=np.int64),
               "input_mean": self.input_mean,
               "input_std": self.input_std}
        out.update(self.net.param_dict())
        return out
