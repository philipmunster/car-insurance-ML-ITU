class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes=[32, 32], alpha=0.001, l2_lambda=0.0001, dropout_rate=0.2):
        """
        PyTorch model that properly mirrors the manual implementation

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes (default: [32, 32])
            alpha: Learning rate (for optimizer)
            l2_lambda: L2 regularization strength (for optimizer weight decay)
            dropout_rate: Dropout rate after ALL hidden layers
        """
        super().__init__()

        self.alpha = alpha
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate

        # Build hidden layers
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer (matches manual implementation with bias)
            layers.append(nn.Linear(prev_size, hidden_size))

            # ReLU activation
            layers.append(nn.ReLU())

            # FIX 1: Dropout after ALL hidden layers (including last)
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))

            prev_size = hidden_size

        self.net = nn.Sequential(*layers)

        # Output heads: 2 outputs (mu and alpha)
        self.output_layer = nn.Linear(prev_size, 2)

        # Initialize weights to match manual implementation
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights to match manual implementation"""
        # Hidden layers: He initialization (for ReLU)
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # FIX 2: Output layer initialization to match manual
        # Manual uses: w_out = np.random.randn(layers[-2], 2) * 0.01
        nn.init.normal_(self.output_layer.weight, mean=0, std=0.01)  # Changed from 0.1 to 0.01!

        # FIX 3: Output bias to match manual [0.0, 1.0]
        if self.output_layer.bias is not None:
            with torch.no_grad():
                self.output_layer.bias[0] = 0.0  # mu bias
                self.output_layer.bias[1] = 1.0  # alpha bias (exp(1) ≈ 2.7 initial alpha)

    def forward(self, x, exposure=None):
        """
        Forward pass with exposure as log-offset

        Args:
            x: Input features (batch_size, input_size)
            exposure: Optional exposure values (batch_size,) or (batch_size, 1)
                      If provided, mu is scaled as mu = exp(z_mu + log(exposure))

        Returns:
            mu: Predicted mean (batch_size,)
            alpha: Predicted dispersion (batch_size,)
        """
        h = self.net(x)
        output = self.output_layer(h)

        # Clip to prevent overflow
        output = torch.clamp(output, -10, 10)

        # FIX 4: Simplified forward pass matching manual implementation
        # Split into mu and alpha components
        z_mu = output[:, 0]
        z_alpha = output[:, 1]

        # Add exposure offset BEFORE exponential (matches manual)
        if exposure is not None:
            if len(exposure.shape) > 1:
                exposure = exposure.squeeze()
            z_mu = z_mu + torch.log(exposure + 1e-8)

        # Apply exponential activation
        mu = torch.exp(z_mu)
        alpha = torch.exp(z_alpha)

        return mu, alpha

    def predict(self, x, exposure=None):
        """
        Predict in eval mode

        Args:
            x: Input features
            exposure: Optional exposure values

        Returns:
            Dictionary with mu, alpha, variance, std
        """
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
    """
    Negative Binomial Negative Log-Likelihood

    Args:
        y_true: Observed counts
        mu: Predicted mean
        alpha: Predicted dispersion parameter
        sample_weights: Optional sample weights (NOT exposure - that's in mu!)

    Returns:
        Scalar loss value
    """
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

    if sample_weights is not None:
        nll = nll * sample_weights
        return torch.sum(nll) / torch.sum(sample_weights)
    else:
        return torch.mean(nll)


def create_optimizer(model, alpha=0.01, l2_lambda=0.01):
    """
    Create optimizer matching manual implementation

    FIX 5: Exclude biases from L2 regularization (weight_decay)
    Manual implementation only applies L2 to weights, not biases!

    Args:
        model: The neural network model
        alpha: Learning rate (matches manual implementation)
        l2_lambda: L2 regularization strength

    Returns:
        optimizer: SGD optimizer with weight decay only on weights
    """
    # Separate parameters into weights and biases
    weight_params = []
    bias_params = []

    for name, param in model.named_parameters():
        if 'bias' in name:
            bias_params.append(param)
        else:
            weight_params.append(param)

    # Create parameter groups: weights with L2, biases without
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
    """
    Training function that matches manual implementation behavior

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data (optional)
        epochs: Number of training epochs
        alpha: Learning rate
        l2_lambda: L2 regularization strength
        device: 'cpu' or 'cuda'
        display_update: How often to print progress

    Returns:
        train_loss_history: List of training losses
        test_loss_history: List of test losses
    """
    model = model.to(device)

    # FIX 5: Use corrected optimizer that excludes biases from L2
    optimizer = create_optimizer(model, alpha=alpha, l2_lambda=l2_lambda)

    train_loss_history = []
    test_loss_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            # Unpack batch - flexible to handle different formats
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

            # Forward pass with exposure as log-offset
            mu, alpha = model(X_batch, exposure=exposure_batch)

            # Compute NB loss
            # NOTE: sample_weights are for observation importance, NOT exposure!
            # Exposure is already incorporated in mu via the offset
            loss = negative_binomial_nll(y_batch.squeeze(), mu, alpha, sample_weights=weights_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # FIX 6: Gradient clipping (matches manual implementation)
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

        # Display
        if (epoch + 1) % display_update == 0 or epoch == 0 or epoch == epochs - 1:
            if test_loader is not None:
                print(f"Epoch {epoch + 1}/{epochs} — Train Loss: {avg_train_loss:.7f}, Test Loss: {avg_test_loss:.7f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs} — Train Loss: {avg_train_loss:.7f}")

    return train_loss_history, test_loss_history


# @@@@@@@@
# @@@@@@@@
# @@@@@@@@
# Usage
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

# Create model (matches manual implementation architecture)
model = NeuralNetwork(
    input_size=X_train.shape[1],
    hidden_sizes=[32, 32],  # Same as manual implementation
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

# Predict
model.eval()
with torch.no_grad():
    # For annualized risk predictions (don't pass exposure)
    # mu_test, alpha_test = model(X_test_tensor, exposure=None)

    # For predictions with actual exposure
    mu_test_scaled, alpha_test_scaled = model(X_test_tensor, exposure=exposure_test_tensor)

### Inspect predictions
model.eval()
with torch.no_grad():
    # Convert X_test to tensor if not already
    X_test_tensor = torch.FloatTensor(X_test_scaled.values if hasattr(X_test_scaled, 'values') else X_test_scaled)

    # Get predictions (without exposure = annualized rate)
    preds = model.predict(X_test_tensor, exposure=None)

    # Access results (they're tensors, convert to numpy if needed)
    print(f"Mean prediction: {preds['mu'][0].item():.2f}")
    print(f"Uncertainty (alpha): {preds['alpha'][0].item():.3f}")
    print(f"Standard deviation: {preds['std'][0].item():.2f}")

# Or for predictions with actual exposure:
with torch.no_grad():
    exposure_test_tensor = torch.FloatTensor(exposure_testset)
    preds_with_exposure = model.predict(X_test_tensor, exposure=exposure_test_tensor)

    print(f"Expected claims: {preds_with_exposure['mu'][0].item():.2f}")
