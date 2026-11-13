import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import CloudDataset
from model import UNet


def train(data_dir="data/38-Cloud_training", epochs=3, batch_size=8, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en dispositivo: {device}")

    dataset = CloudDataset(base_dir=data_dir)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, _ = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    model = UNet(in_channels=4, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Época [{epoch+1}/{epochs}]")

        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Resumen Época {epoch+1}: Loss = {running_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "unet_cloud_segmentation.pth")
    print("Modelo guardado en 'unet_cloud_segmentation.pth'")


if __name__ == "__main__":
    train()