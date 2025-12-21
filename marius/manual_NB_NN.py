class NBManualNeuralNetwork:
    def __init__(self, layers, alpha=0.01, l2_lambda=0.01, dropout_rate=0.0):
        self.layers = layers
        self.alpha = alpha
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.train_loss_history = []
        self.test_loss_history = []
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 2):
            w = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        # Output layer: 2 neurons (mu, alpha)
        # Small weight init + bias: mu_bias = 0, alpha_bias = log(initial alpha)
        w_out = np.random.randn(layers[-2], 2) * 0.01
        b_out = np.array([[0.0, 1.0]])
        self.weights.append(w_out)
        self.biases.append(b_out)

    def ReLU(self, x):
        return np.maximum(0, x)

    def ReLU_derivative(self, x):
        return (x > 0).astype(float)

    def negative_binomial_nll(self, y_true, mu, alpha, sample_weight=None):
        mu = np.maximum(mu, 1e-8)
        alpha = np.maximum(alpha, 1e-8)

        r = 1.0 / alpha
        p = r / (r + mu)

        # The negative log-likelihood
        nll = -(gammaln(y_true + r) - gammaln(r) - gammaln(y_true + 1)
                + r * np.log(p + 1e-8) + y_true * np.log(1 - p + 1e-8))

        # <-------------------> Not scaling loss
        # if sample_weight is not None:
        #     nll = nll * sample_weight
        #     return np.sum(nll) / np.sum(sample_weight)
        # else:
        return np.mean(nll)

    def negative_binomial_gradients(self, y_true, mu, alpha):
        mu = np.maximum(mu, 1e-8)
        alpha = np.maximum(alpha, 1e-8)
        r = 1.0 / alpha
        grad_mu = -(y_true - mu) / (mu * (1 + alpha * mu))
        grad_alpha = (1.0 / (alpha ** 2)) * (
                digamma(r) - digamma(y_true + r) + np.log(1 + alpha * mu) + (alpha * mu - y_true) / (1 + alpha * mu)
        )
        return grad_mu, grad_alpha

    def fit(self, X, y, exposure=None, epochs=100, batch_size=512, X_test=None, y_test=None,
            exposure_test=None, sample_weights=None, test_sample_weights=None, displayUpdate=10):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if exposure is not None:
            exposure = np.asarray(exposure, dtype=float).squeeze()

        n_samples = len(y)
        if sample_weights is None:
            sample_weights = np.ones(n_samples)
        else:
            sample_weights = np.asarray(sample_weights, dtype=float)
        if test_sample_weights is not None:
            test_sample_weights = np.asarray(test_sample_weights, dtype=float)

        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(n_samples)
            epoch_loss = 0.0

            for i in range(0, n_samples, batch_size):
                idx = perm[i:i + batch_size]
                X_batch = X[idx]
                y_batch = y[idx]
                exp_batch = exposure[idx] if exposure is not None else None
                w_batch = sample_weights[idx]

                loss = self.fit_batch(X_batch, y_batch, exp_batch, w_batch)
                epoch_loss += loss * len(idx)

            epoch_loss /= n_samples
            self.train_loss_history.append(epoch_loss)

            if X_test is not None and y_test is not None:
                mu_test, alpha_test = self.predict_params(X_test, exposure=exposure_test)
                test_loss = self.negative_binomial_nll(y_test, mu_test, alpha_test, test_sample_weights)
                self.test_loss_history.append(test_loss)

            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                if X_test is not None:
                    print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {epoch_loss:.6f}, Test Loss = {test_loss:.6f}")
                else:
                    print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {epoch_loss:.6f}")

    def fit_batch(self, X, y, exposure, w):
        batch_size = len(y)
        activations = [X]
        pre_acts = []
        dropout_masks = []

        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            pre_acts.append(z)
            a = self.ReLU(z)

            if self.dropout_rate > 0:
                mask = np.random.binomial(1, 1 - self.dropout_rate, size=a.shape)
                a *= mask / (1 - self.dropout_rate)
                dropout_masks.append(mask)
            else:
                dropout_masks.append(None)

            a = np.clip(a, -20, 20)
            activations.append(a)

        z_out = activations[-1] @ self.weights[-1] + self.biases[-1]
        z_out = np.clip(z_out, -10, 10)

        z_mu = z_out[:, 0].copy()
        z_alpha = z_out[:, 1].copy()

        if exposure is not None:
            exposure = np.asarray(exposure, dtype=float).squeeze()
            z_mu_with_offset = z_mu + np.log(exposure + 1e-8)
            mu = np.exp(z_mu_with_offset)
        else:
            mu = np.exp(z_mu)

        alpha = np.exp(z_alpha)
        nll_loss = self.negative_binomial_nll(y, mu, alpha, sample_weight=w)
        l2_penalty = 0.0
        for W in self.weights:
            l2_penalty += np.sum(W ** 2)
        l2_penalty *= (self.l2_lambda / 2.0)

        batch_loss = nll_loss + l2_penalty
        grad_mu, grad_alpha = self.negative_binomial_gradients(y, mu, alpha)

        # multiply grad with exp
        grad_mu = grad_mu * w
        grad_alpha = grad_alpha * w

        grad_z_mu = grad_mu * mu
        grad_z_alpha = grad_alpha * alpha

        grad_out = np.column_stack([grad_z_mu, grad_z_alpha])

        deltas = [grad_out]
        for i in range(len(self.weights) - 2, -1, -1):
            delta = deltas[-1] @ self.weights[i + 1].T
            delta *= self.ReLU_derivative(pre_acts[i])
            if self.dropout_rate > 0 and dropout_masks[i] is not None:
                delta *= dropout_masks[i] / (1 - self.dropout_rate)

            deltas.append(delta)
        deltas = deltas[::-1]

        for i in range(len(self.weights)):
            grad = (activations[i].T @ deltas[i]) / batch_size
            grad = np.clip(grad, -10, 10)

            self.weights[i] -= self.alpha * (grad + self.l2_lambda * self.weights[i])
            self.biases[i] -= self.alpha * np.mean(deltas[i], axis=0, keepdims=True)

        return batch_loss

    def predict_params(self, X, exposure=None):
        a = np.asarray(X, dtype=float)

        # forward pass for pred (w/out dropout)
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            a = self.ReLU(z)
            a = np.clip(a, -20, 20)

        # output
        z_out = a @ self.weights[-1] + self.biases[-1]
        z_out = np.clip(z_out, -10, 10)

        # log offset <---
        if exposure is not None:
            exposure = np.asarray(exposure, dtype=float).squeeze()
            z_out[:, 0] += np.log(exposure + 1e-8)

        mu = np.exp(z_out[:, 0])
        alpha = np.exp(z_out[:, 1])
        return mu, alpha

    def predict(self, X, exposure=None, return_distribution=True):
        mu, alpha = self.predict_params(X, exposure)
        if return_distribution:
            variance = mu + alpha * mu ** 2
            return {
                'mu': mu,
                'alpha': alpha,
                'variance': variance,
                'std': np.sqrt(variance)
            }
        else:
            return mu

if __name__ == '__main__':
    nnl_nb = NBManualNeuralNetwork(
        layers=[11, 32, 32, 2],
        alpha=0.001,
        l2_lambda=0.01,
        dropout_rate=0
    )

    nnl_nb.fit(
        X_train_scaled,
        y_train.values,
        exposure=exposure_trainset,
        epochs=20,
        sample_weights=None,
        X_test=X_test_scaled,
        y_test=y_test.values,
        exposure_test=exposure_testset,
        displayUpdate=1
    )

    mu, alpha = nnl_nb.predict_params(
        X_test_scaled,
        exposure=None
    )
    print(f"Mean prediction: {mu.mean()}")
    print(f"Uncertainty (alpha): {alpha.mean()}")
    print(f'Learned disp. parameter: {np.exp(nnl_nb.biases[-1][0, 1])}')