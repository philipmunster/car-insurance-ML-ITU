class ManualNeuralNetwork:
    def __init__(self, layers, alpha=0.01, l2_lambda=0.01, dropout_rate=0):
        self.train_loss_history = []
        self.test_loss_history = []
        self.weights = []
        self.layers = layers
        self.alpha = alpha
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.dropout_mask = []

        for i in np.arange(0, len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.weights.append(w * np.sqrt(2.0 / layers[i]))

        # Output layer
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.weights.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        return "PoissonLossNN: {}".format("-".join(str(l) for l in self.layers))

    def ReLU_af(self, input):
        return np.maximum(0, input)

    def ReLU_derivative(self, input):
        return (input > 0).astype(float)

    def exponential_af(self, input):
        return np.exp(np.clip(input, -10, 10))

    def poisson_deviance(self, y_true, y_pred):
        """
        Poisson deviance loss with y_true = 0 fix.
        """
        y_pred = np.maximum(y_pred, 1e-8)
        deviance = np.zeros_like(y_true, dtype=float)

        nonzero_mask = y_true > 0
        if np.any(nonzero_mask):
            y_t = y_true[nonzero_mask]
            y_p = y_pred[nonzero_mask]
            deviance[nonzero_mask] = 2 * (y_t * np.log(y_t / y_p) - (y_t - y_p))

        zero_mask = y_true == 0
        if np.any(zero_mask):
            deviance[zero_mask] = 2 * y_pred[zero_mask]

        return np.mean(deviance)

    def poisson_deviance_gradient(self, y_true, y_pred):
        y_pred = np.maximum(y_pred, 1e-8)
        return 2 * (1 - y_true / y_pred)

    def fit(self, X, y, exposure, epochs=100, displayUpdate=100,
            X_test=None, y_test=None, exposure_test=None):

        X_bias = np.c_[X, np.ones((X.shape[0]))]
        exposure = np.array(exposure)

        if len(exposure) != len(y):
            raise ValueError(f"Exposure length ({len(exposure)}) doesn't match y length ({len(y)})!")
        if np.any(exposure <= 0):
            raise ValueError("Exposure must be positive!")

        if X_test is not None and y_test is not None:
            if exposure_test is None:
                raise ValueError("Test data exposures must be provided when X_test and y_test are provided.")
            exposure_test = np.array(exposure_test)

        for epoch in np.arange(0, epochs):
            epoch_loss = 0
            n_samples = len(y)

            for (x, target, exp) in zip(X_bias, y, exposure):
                loss = self.fit_epoch(x, target, exp)
                epoch_loss += loss

            avg_loss = epoch_loss / n_samples
            self.train_loss_history.append(avg_loss)

            # val loss
            if X_test is not None and y_test is not None:
                test_preds = self.predict(X_test, exposure_test)
                test_loss = self.poisson_deviance(y_test, test_preds)
                self.test_loss_history.append(test_loss)

            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                if X_test is not None:
                    print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {avg_loss:.7f}, Test Loss = {test_loss:.7f}")
                else:
                    print(f"Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.7f}")

    def fit_epoch(self, x, y, exposure):
        # forward pass
        activations = [np.atleast_2d(x)]
        pre_activations = []
        self.dropout_mask = []

        # hidden layers
        for layer in np.arange(0, len(self.weights) - 1):
            net = activations[layer].dot(self.weights[layer])
            pre_activations.append(net)
            out = self.ReLU_af(net)

            # dropout reg (set nprandom seed for reproductablity)
            if self.dropout_rate > 0:
                drop_mask = np.random.binomial(1, 1 - self.dropout_rate, size=out.shape)
                out = (out * drop_mask) / (1 - self.dropout_rate)
                self.dropout_mask.append(drop_mask)
            else:
                self.dropout_mask.append(None)

            activations.append(out)

        # output layer log(rate) predictor
        net = activations[-1].dot(self.weights[-1])
        pre_activations.append(net)

        log_offset = np.log(exposure)
        net_with_offset = net + log_offset
        y_pred = self.exponential_af(net_with_offset)

        activations.append(y_pred)

        y_pred_scalar = y_pred.flatten()[0]
        sample_loss = self.poisson_deviance(np.array([y]), np.array([y_pred_scalar]))

        # backward pass
        grad_loss = self.poisson_deviance_gradient(np.array([y]), np.array([y_pred_scalar]))
        error = np.atleast_2d(grad_loss * y_pred)

        D = [error]

        # backpropagation
        for layer in np.arange(len(activations) - 2, 0, -1):
            delta = D[-1].dot(self.weights[layer].T)
            delta *= self.ReLU_derivative(activations[layer])

            #]dropout mask
            mask_idx = layer - 1
            if mask_idx >= 0 and mask_idx < len(self.dropout_mask) and self.dropout_mask[mask_idx] is not None:
                delta = (delta * self.dropout_mask[mask_idx]) / (1 - self.dropout_rate)

            D.append(delta)

        D = D[::-1]

        # w8 update
        for layer in np.arange(0, len(self.weights)):
            gradient = activations[layer].T.dot(D[layer])

            # added clip gradient because it kept booming :(
            if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
                print(f"Warning: Invalid gradient at layer {layer}")
                gradient = np.nan_to_num(gradient, nan=0.0, posinf=1.0, neginf=-1.0)

            gradient = np.clip(gradient, -10, 10)

            # l2
            l2_penalty = self.l2_lambda * self.weights[layer]

            # w8s
            self.weights[layer] += -self.alpha * (gradient + l2_penalty)

        return sample_loss

    def predict(self, X, exposure, addBias=True):
        p = np.atleast_2d(X)
        exposure = np.array(exposure)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        if len(exposure) != p.shape[0]:
            raise ValueError(f"Exposure length ({len(exposure)}) doesn't match length ({p.shape[0]})!")
        if np.any(exposure <= 0):
            raise ValueError("Exposure must be positive!")

        for layer in np.arange(0, len(self.weights) - 1):
            net = np.dot(p, self.weights[layer])
            p = self.ReLU_af(net)

        net = np.dot(p, self.weights[-1])

        # EXPOSURE OFFSET @@@@@@@@@
        log_offset = np.log(exposure).reshape(-1, 1)
        net_with_offset = net + log_offset
        predictions = self.exponential_af(net_with_offset)

        return predictions.flatten()

    def predict_rate(self, X, addBias=True):
        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.weights)):
            net = np.dot(p, self.weights[layer])
            if layer == len(self.weights) - 1:
                p = self.exponential_af(net)
            else:
                p = self.ReLU_af(net)

        return p.flatten()

if __name__ == '__main__':
    nnl_poisson = ManualNeuralNetwork([11, 32, 32, 1], alpha=0.001, l2_lambda=0.01, dropout_rate=0)
    nnl_poisson.fit(X_train_scaled, y_train, exposure_trainset, epochs=5, X_test=X_test_scaled, y_test=y_test,
                    displayUpdate=1, exposure_test=exposure_testset)

    # Check mean pred
    preds_poisson = nnl_poisson.predict(X_test, exposure=exposure_testset)
    print(f"Mean prediction: {np.mean(preds_poisson):.2f}")