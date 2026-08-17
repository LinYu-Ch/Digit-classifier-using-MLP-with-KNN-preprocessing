import torch
import torch.nn as nn
from model import ConvNetwork
from PIL import Image
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inference_model = ConvNetwork(num_classes=10)
inference_model.load_state_dict(torch.load("cnn_mnist_weights.pth", weights_only=True))
inference_model.to(device)
inference_model.eval()

def image_preprocessing(path):
    transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((28, 28)),
        T.ToTensor(),
    ])

    img = Image.open(path)
    tensor_3d = transform(img)
    tensor_3d = 1.0 - tensor_3d # inverted

    # faking dataloader format
    tensor_4d = tensor_3d.unsqueeze(0)
    return tensor_4d.to(device)

input_tensor = image_preprocessing("21.jpeg")

with torch.no_grad():
    logits = inference_model(input_tensor)
    p_spread = torch.softmax(logits, dim=1)
    prediction = torch.argmax(p_spread, dim=1).item()
    confidence = p_spread[0][prediction].item()

print(f"predicted Digit: {prediction} ({confidence * 100:.2f}% confidence)")