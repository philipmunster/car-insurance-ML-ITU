from scipy.special import gammaln, digamma


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

        # Hidden layers: He init
        for i in range(len(layers) - 2):
            w = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        # Output layer: 2 neurons (mu, alpha)
        # Small weight init + bias: mu_bias = 0, alpha_bias = log(initial alpha)
        w_out = np.random.randn(layers[-2], 2) * 0.01
        b_out = np.array([[0.0, 1.0]])  # exp(1.0) ≈ 2.718 → closer to typical starting alpha
        self.weights.append(w_out)
        self.biases.append(b_out)

    def ReLU(self, x):
        return np.maximum(0, x)

    def ReLU_derivative(self, x):
        return (x > 0).astype(float)

    def negative_binomial_nll(self, y_true, mu, alpha, sample_weight=None):
        """
        Negative Binomial Negative Log-Likelihood.
        Note: Returns POSITIVE values (we're computing the negative of log-likelihood).
        """
        mu = np.maximum(mu, 1e-8)
        alpha = np.maximum(alpha, 1e-8)

        r = 1.0 / alpha
        p = r / (r + mu)

        # The negative log-likelihood
        nll = -(gammaln(y_true + r) - gammaln(r) - gammaln(y_true + 1)
                + r * np.log(p + 1e-8) + y_true * np.log(1 - p + 1e-8))

        if sample_weight is not None:
            nll = nll * sample_weight
            return np.sum(nll) / np.sum(sample_weight)
        else:
            return np.mean(nll)

    def negative_binomial_gradients(self, y_true, mu, alpha):
        """
        Compute gradients of NLL with respect to mu and alpha.
        These are the gradients we want to DESCEND, so they point in the direction of increasing loss.
        """
        mu = np.maximum(mu, 1e-8)
        alpha = np.maximum(alpha, 1e-8)
        r = 1.0 / alpha

        # Gradient of NLL w.r.t. mu (positive when mu < y, negative when mu > y)
        grad_mu = -(y_true - mu) / (mu * (1 + alpha * mu))

        # Gradient of NLL w.r.t. alpha
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

        # Forward pass
        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            pre_acts.append(z)
            a = self.ReLU(z)

            # Dropout - Store masks for backprop
            if self.dropout_rate > 0:
                mask = np.random.binomial(1, 1 - self.dropout_rate, size=a.shape)
                a *= mask / (1 - self.dropout_rate)
                dropout_masks.append(mask)
            else:
                dropout_masks.append(None)

            # Clip hidden activation
            a = np.clip(a, -20, 20)
            activations.append(a)

        # Output layer - Get network predictions BEFORE adding exposure
        z_out = activations[-1] @ self.weights[-1] + self.biases[-1]
        z_out = np.clip(z_out, -10, 10)

        # Store the network output z for backprop (before exposure)
        z_mu = z_out[:, 0].copy()
        z_alpha = z_out[:, 1].copy()

        # Add exposure offset ONLY for computing mu (not for backprop through network)
        if exposure is not None:
            exposure = np.asarray(exposure, dtype=float).squeeze()
            z_mu_with_offset = z_mu + np.log(exposure + 1e-8)
            mu = np.exp(z_mu_with_offset)
        else:
            mu = np.exp(z_mu)

        alpha = np.exp(z_alpha)

        # Loss (includes L2 regularization)
        nll_loss = self.negative_binomial_nll(y, mu, alpha, sample_weight=w)

        # Add L2 regularization to the loss
        l2_penalty = 0.0
        for W in self.weights:
            l2_penalty += np.sum(W ** 2)
        l2_penalty *= (self.l2_lambda / 2.0)

        batch_loss = nll_loss + l2_penalty

        # ============================================================================
        # CRITICAL: Backward pass - exposure offset does NOT backprop through network
        # ============================================================================
        # Gradients of NLL w.r.t. mu and alpha
        grad_mu, grad_alpha = self.negative_binomial_gradients(y, mu, alpha)

        # Apply sample weights
        grad_mu = grad_mu * w
        grad_alpha = grad_alpha * w

        # Chain rule for output layer:
        # We have: mu = exp(z_mu + log(exposure)) = exp(z_mu) * exposure
        # So: d(loss)/d(z_mu) = d(loss)/d(mu) * d(mu)/d(z_mu)
        #                      = grad_mu * exp(z_mu) * exposure
        #                      = grad_mu * mu
        #
        # The exposure multiplies into mu, but its derivative doesn't flow back
        # through the network weights because exposure is not learned!
        #
        # Similarly: alpha = exp(z_alpha)
        # So: d(loss)/d(z_alpha) = grad_alpha * alpha

        grad_z_mu = grad_mu * mu
        grad_z_alpha = grad_alpha * alpha

        grad_out = np.column_stack([grad_z_mu, grad_z_alpha])

        # Backpropagate through hidden layers
        deltas = [grad_out]
        for i in range(len(self.weights) - 2, -1, -1):
            delta = deltas[-1] @ self.weights[i + 1].T

            # Apply ReLU derivative
            delta *= self.ReLU_derivative(pre_acts[i])

            # Apply dropout mask in backprop
            if self.dropout_rate > 0 and dropout_masks[i] is not None:
                delta *= dropout_masks[i] / (1 - self.dropout_rate)

            deltas.append(delta)
        deltas = deltas[::-1]

        # Weight updates
        for i in range(len(self.weights)):
            # Average gradients over batch
            grad = (activations[i].T @ deltas[i]) / batch_size
            grad = np.clip(grad, -10, 10)

            # Apply L2 regularization and learning rate
            self.weights[i] -= self.alpha * (grad + self.l2_lambda * self.weights[i])

            # Bias update (no L2 regularization on biases)
            self.biases[i] -= self.alpha * np.mean(deltas[i], axis=0, keepdims=True)

        return batch_loss

    def predict_params(self, X, exposure=None):
        """
        Predict the parameters (mu, alpha) of the Negative Binomial distribution.

        IMPORTANT: When exposure is provided, mu represents the expected count
        for that exposure duration. To get the annualized rate, compute mu/exposure.
        """
        a = np.asarray(X, dtype=float)

        # Forward pass through hidden layers (no dropout during prediction)
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            a = self.ReLU(z)
            a = np.clip(a, -20, 20)

        # Output layer
        z_out = a @ self.weights[-1] + self.biases[-1]
        z_out = np.clip(z_out, -10, 10)

        # Apply exposure offset
        if exposure is not None:
            exposure = np.asarray(exposure, dtype=float).squeeze()
            z_out[:, 0] += np.log(exposure + 1e-8)

        mu = np.exp(z_out[:, 0])
        alpha = np.exp(z_out[:, 1])
        return mu, alpha

    def predict(self, X, exposure=None, return_distribution=True):
        """
        Predict either the distribution parameters or just the mean.

        For insurance claims with exposure as offset:
        - mu represents the expected claim count for the given exposure
        - To get annualized risk: mu / exposure gives claims per unit time

        ------
        # To get expected claims for actual exposure periods:
        predictions = model.predict(X_test, exposure=exposure_test)
        expected_claims = predictions['mu']

        # To get annualized claim rates (claims per year):
        predictions = model.predict(X_test, exposure=exposure_test)
        annual_rate = predictions['mu'] / exposure_test

        # Or equivalently, predict with exposure=1:
        predictions_annual = model.predict(X_test, exposure=np.ones(len(X_test)))
        annual_rate = predictions_annual['mu']
        """
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

# @@@@@@@@@@@
# @@@@@@@@@@@
# @@@@@@@@@@@
# Usage
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