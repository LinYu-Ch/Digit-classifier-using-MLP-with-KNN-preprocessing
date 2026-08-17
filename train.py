import os
import struct
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import ConvNetwork

class MNISTDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        labels = df.pop('class')
        self.labels = torch.tensor(labels.values, dtype=torch.long)

        raw_data = df.values
        tensor_1d = torch.tensor(raw_data, dtype=torch.float32)
        tensor_normalized = tensor_1d / 255.0
        self.images = tensor_normalized.view(-1, 1, 28, 28)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class MNISTDataManager:
    def __init__(self, data_path="archive"):
        self.data_path = data_path

    @staticmethod
    def _parse_idx_images(file_path):
        with open(file_path, 'rb') as f:
            _, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            pixels = np.frombuffer(f.read(), dtype=np.uint8)
            return pixels.reshape(num_images, rows * cols)

    @staticmethod
    def _parse_idx_labels(file_path):
        with open(file_path, 'rb') as f:
            _, num_items = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    def load_dataframe(self, images_filename, labels_filename):
        images_file = os.path.join(self.data_path, images_filename)
        labels_file = os.path.join(self.data_path, labels_filename)

        pixels_array = self._parse_idx_images(images_file)
        labels_array = self._parse_idx_labels(labels_file)

        df = pd.DataFrame(pixels_array)
        df['class'] = labels_array
        return df

    def get_dataloaders(self, batch_size=32):
        train_df = self.load_dataframe("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
        train_dataset = MNISTDataset(train_df)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        val_df = self.load_dataframe("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
        val_dataset = MNISTDataset(val_df)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

        print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
        return train_loader, val_loader
    
class CNNTrainer:
    def __init__(self, model, train_loader, val_loader=None, lr=0.001, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_epoch(self):
        self.model.train()  # Enable training mode (enables gradient tracking)
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = (correct_preds / total_samples) * 100
        return epoch_loss, epoch_acc

    def evaluate(self):
        """Runs validation without backprop or gradient updates."""
        if self.val_loader is None:
            return None, None

        self.model.eval()  # Enable evaluation mode
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        with torch.no_grad():  # Disable autograd memory allocation
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_preds += torch.sum(preds == labels).item()
                total_samples += labels.size(0)

        val_loss = running_loss / total_samples
        val_acc = (correct_preds / total_samples) * 100
        return val_loss, val_acc

    def fit(self, epochs=5):
        print(f"Starting execution on device: {self.device}\n" + "-" * 55)
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate()

            if val_loss is not None:
                print(f"Epoch [{epoch}/{epochs}] | "
                      f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
            else:
                print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}%")

    # Save weights immediately after all epochs complete
        self.save_model("cnn_mnist_weights.pth")

    def save_model(self, filepath="cnn_mnist_weights.pth"):
        """Saves only the learned weights (state_dict) to disk."""
        torch.save(self.model.state_dict(), filepath)
        print(f"\nModel weights saved successfully to '{filepath}'!")
        
if __name__ == "__main__":
    data_manager = MNISTDataManager(data_path="archive")
    train_loader, val_loader = data_manager.get_dataloaders(batch_size=32)

    cnn_model = ConvNetwork(num_classes=10)
    trainer = CNNTrainer(
        model=cnn_model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        lr=0.001
    )

    trainer.fit(epochs=5)