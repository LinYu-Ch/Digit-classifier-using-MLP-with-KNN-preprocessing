import torch
import torch.nn as nn

class ConvNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # FEATURE EXTRACTION:
        # conv2D (channel_in: 1 (grayscale), channel_out: 16 (feature maps), kernel_size: 3, stride: 1, padding: 1)
        # relu (cuts down negative values)
        # maxpool (kernel_size: 2, stride: 2 -> shrinks tensor spatial dimensions by 1/2)
        # conv2D (16 in, 32 out, 3 kernel, 1 stride, 1 padding)
        # relu
        # maxpool
        self.feature_extractor = nn.Sequential()
        self.feature_extractor.add_module("conv2D 1", nn.Conv2d(1, 16, 3, 1, 1))
        self.feature_extractor.add_module("relu 1", nn.ReLU())
        self.feature_extractor.add_module("MaxPool 1", nn.MaxPool2d(2, 2))
        self.feature_extractor.add_module("conv2D 2", nn.Conv2d(16, 32, 3, 1, 1))
        self.feature_extractor.add_module("relu 2", nn.ReLU())
        self.feature_extractor.add_module("MaxPool 2", nn.MaxPool2d(2, 2))

        # INTERMEDIARY OPERATIONS
        self.flatten = nn.Flatten()
        dummy_input = torch.zeros(1, 1, 28, 28)
        dummy_output = self.flatten(self.feature_extractor(dummy_input))
        in_features = dummy_output.shape[1]

        # CLASSIFIER:
        # 1D flattened feature vector -> Linear hidden layers -> Class probabilities (Logits)
        self.classifier = nn.Sequential()
        self.classifier.add_module("Dense Hidden 1", nn.Linear(in_features=in_features, out_features=128))
        self.classifier.add_module("dense relu 1", nn.ReLU())
        self.classifier.add_module("Dense Hidden 2", nn.Linear(128, 64))
        self.classifier.add_module("dense relu 2", nn.ReLU())
        self.classifier.add_module("output layer", nn.Linear(64, num_classes))

    def forward(self, batch):
        x = self.feature_extractor(batch)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits
