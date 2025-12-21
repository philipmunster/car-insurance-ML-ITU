class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes=[32, 32], alpha=0.001, l2_lambda=0.0001, dropout_rate=0.2):
        super().__init__()
        self.alpha = alpha
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())

            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))

            prev_size = hidden_size

        self.net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, 2)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        nn.init.normal_(self.output_layer.weight, mean=0, std=0.01)

        if self.output_layer.bias is not None:
            with torch.no_grad():
                self.output_layer.bias[0] = 0.0
                self.output_layer.bias[1] = 1.0

    def forward(self, x, exposure=None):
        h = self.net(x)
        output = self.output_layer(h)

        output = torch.clamp(output, -10, 10)
        z_mu = output[:, 0]
        z_alpha = output[:, 1]

        if exposure is not None:
            if len(exposure.shape) > 1:
                exposure = exposure.squeeze()
            z_mu = z_mu + torch.log(exposure + 1e-8)

        # Apply exponential activation
        mu = torch.exp(z_mu)
        alpha = torch.exp(z_alpha)

        return mu, alpha

    def predict(self, x, exposure=None):
        self.eval()
        with torch.no_grad():
            mu, alpha = self.forward(x, exposure)
            variance = mu + alpha * mu ** 2
            return {
                'mu': mu,
                'alpha': alpha,
                'variance': variance,
                'std': torch.sqrt(variance)
            }


def negative_binomial_nll(y_true, mu, alpha, sample_weights=None):
    mu = torch.clamp(mu, min=1e-8)
    alpha = torch.clamp(alpha, min=1e-8)

    r = 1.0 / alpha
    p = r / (r + mu)

    nll = -(
            torch.lgamma(y_true + r)
            - torch.lgamma(r)
            - torch.lgamma(y_true + 1)
            + r * torch.log(p + 1e-8)
            + y_true * torch.log(1 - p + 1e-8)
    )

    # <--> Not scaled by sample weights anymore
    # if sample_weights is not None:
    #     nll = nll * sample_weights
    #     return torch.sum(nll) / torch.sum(sample_weights)
    # else:
    return torch.mean(nll)


def create_optimizer(model, alpha=0.01, l2_lambda=0.01):
    weight_params = []
    bias_params = []

    for name, param in model.named_parameters():
        if 'bias' in name:
            bias_params.append(param)
        else:
            weight_params.append(param)

    optimizer = torch.optim.SGD([
        {'params': weight_params, 'weight_decay': l2_lambda},
        {'params': bias_params, 'weight_decay': 0.0}  # No L2 on biases!
    ], lr=alpha, momentum=0.0)

    return optimizer


def train_model(
        model,
        train_loader,
        test_loader=None,
        epochs=20,
        alpha=0.01,
        l2_lambda=0.01,
        device='cpu',
        display_update=1
):
    model = model.to(device)

    optimizer = create_optimizer(model, alpha=alpha, l2_lambda=l2_lambda)

    train_loss_history = []
    test_loss_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            if len(batch) == 4:
                X_batch, y_batch, weights_batch, exposure_batch = batch
                exposure_batch = exposure_batch.to(device)
            elif len(batch) == 3:
                X_batch, y_batch, weights_batch = batch
                exposure_batch = None
            else:
                X_batch, y_batch = batch
                weights_batch = None
                exposure_batch = None

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            if weights_batch is not None:
                weights_batch = weights_batch.to(device)

            mu, alpha = model(X_batch, exposure=exposure_batch)
            loss = negative_binomial_nll(y_batch.squeeze(), mu, alpha, sample_weights=weights_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        train_loss_history.append(avg_train_loss)

        # Validation loss
        if test_loader is not None:
            model.eval()
            test_loss = 0.0
            n_test_batches = 0

            with torch.no_grad():
                for batch in test_loader:
                    if len(batch) == 4:
                        X_batch, y_batch, weights_batch, exposure_batch = batch
                        exposure_batch = exposure_batch.to(device)
                    elif len(batch) == 3:
                        X_batch, y_batch, weights_batch = batch
                        exposure_batch = None
                    else:
                        X_batch, y_batch = batch
                        weights_batch = None
                        exposure_batch = None

                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    if weights_batch is not None:
                        weights_batch = weights_batch.to(device)

                    mu, alpha = model(X_batch, exposure=exposure_batch)
                    loss = negative_binomial_nll(y_batch.squeeze(), mu, alpha, sample_weights=weights_batch)
                    test_loss += loss.item()
                    n_test_batches += 1

            avg_test_loss = test_loss / n_test_batches
            test_loss_history.append(avg_test_loss)

        # display
        if (epoch + 1) % display_update == 0 or epoch == 0 or epoch == epochs - 1:
            if test_loader is not None:
                print(f"Epoch {epoch + 1}/{epochs} — Train Loss: {avg_train_loss:.7f}, Test Loss: {avg_test_loss:.7f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs} — Train Loss: {avg_train_loss:.7f}")

    return train_loss_history, test_loss_history

if __name__ == '__main__':
    #make sure to have all imports needed for torch (see forum)
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values)
    exposure_train_tensor = torch.FloatTensor(exposure_trainset)
    weights_train_tensor = torch.FloatTensor(exposure_trainset)

    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.values)
    exposure_test_tensor = torch.FloatTensor(exposure_testset)
    weights_test_tensor = torch.FloatTensor(exposure_testset)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, weights_train_tensor, exposure_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, weights_test_tensor, exposure_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_sizes=[32, 32],
        alpha=0.001,
        l2_lambda=0.01,
        dropout_rate=0
    )

    # Train
    train_loss_history, test_loss_history = train_model(
        model,
        train_loader,
        test_loader,
        epochs=10,
        alpha=0.001,
        l2_lambda=0.01,
        device='cpu'
    )

    model.eval()
    with torch.no_grad():
        # For annualized risk predictions (don't pass exposure)
        # mu_test, alpha_test = model(X_test_tensor, exposure=None)

        # For predictions with actual exposure
        mu_test_scaled, alpha_test_scaled = model(X_test_tensor, exposure=exposure_test_tensor)

    # check preds
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled.values if hasattr(X_test_scaled, 'values') else X_test_scaled)

        # get predictions (without exposure = annualized rate)
        preds = model.predict(X_test_tensor, exposure=None)

        # access results (they're tensors, convert to numpy if needed)
        print(f"Mean prediction: {preds['mu'][0].item():.2f}")
        print(f"Uncertainty (alpha): {preds['alpha'][0].item():.3f}")
        print(f"Standard deviation: {preds['std'][0].item():.2f}")

    # for predictions with actual exposure:
    with torch.no_grad():
        exposure_test_tensor = torch.FloatTensor(exposure_testset)
        preds_with_exposure = model.predict(X_test_tensor, exposure=exposure_test_tensor)

        print(f"Expected claims: {preds_with_exposure['mu'][0].item():.2f}")