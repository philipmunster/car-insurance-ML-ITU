def copy_pytorch_to_manual(pytorch_model, manual_model):
    """Copy weights from PyTorch model to manual NumPy model"""
    pytorch_modules = [m for m in pytorch_model.net.modules() if isinstance(m, torch.nn.Linear)]

    for i, pytorch_layer in enumerate(pytorch_modules):
        manual_model.weights[i] = pytorch_layer.weight.data.numpy().T
        manual_model.biases[i] = pytorch_layer.bias.data.numpy().reshape(1, -1)

    manual_model.weights[-1] = pytorch_model.output_layer.weight.data.numpy().T
    manual_model.biases[-1] = pytorch_model.output_layer.bias.data.numpy().reshape(1, -1)

    print("Weights copied from PyTorch to manual model")

if __name__ == '__main__':
    copy_pytorch_to_manual(pytorch_model, manual_model)
