import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_msssim import ssim
import pandas as pd

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels,in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#U-Net, le modèle prend en entrée le nombre de channels d'entrée et de sortie (les mêmes valeurs dans notre cas)
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1

        self.down3 = Down(256, 512//factor)
        
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# Fonction pour entraîner un modèle
# Prend en entrée:
# model_type, Le nom du modèle à entraîner
# train_dataset, le dataset d'entraînement
# val_dataset, le dataset de validation
# loss_type="mse", le nom de la fonction de loss, soit mse, soit ssim
# optimizer_type="adam", le nom de la fonction d'optimisation, soit adam, soit sgd
# epochs=50, le nombre d'epoch
# batch_size=32, la taille des batchs
# learning_rate=0.001, le learning rate
# device="cuda", le device sur lequel entaîner le modèle
# La fonction suavegarde automatiquement les modèles avec la val_loss la plus faible
# Elle renvoit le modèle après toute les epochs ainsi que les données d'entraînement

def train_model(
    train_dataset,
    val_dataset,
    train_data_file_name,
    loss_type="mse",
    optimizer_type="adam",
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    device="cuda" if torch.cuda.is_available() else "cpu",
):

    
    in_channels = train_dataset[0][0].shape[-3]
    out_channels = train_dataset[0][1].shape[-3]
    
    model = UNet(n_channels=in_channels, n_classes=out_channels)

    model = model.to(device)

    if loss_type == "mse":
        loss_fn = nn.MSELoss()
    elif loss_type == "ssim":
        loss_fn = lambda x, y: 1 - ssim(x, y, data_range=1.0, size_average=True)
    else:
        raise ValueError("Loss type must be 'mse' or 'ssim'.")

    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Optimizer type must be 'adam' or 'sgd'.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_loss_data = []
    val_loss_data = []

    min_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Processing"):
            grey_imgs, color_imgs = batch
            grey_imgs, color_imgs = grey_imgs.to(device), color_imgs.to(device)

            outputs = model(grey_imgs)
            loss = loss_fn(outputs, color_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size
            
            torch.save(model.state_dict(), "model/dernier_modele.pth")

        train_loss = train_loss / len(train_loader.dataset)
        train_loss_data.append(train_loss)

        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Processing"):
                grey_imgs, color_imgs = batch
                grey_imgs, color_imgs = grey_imgs.to(device), color_imgs.to(device)


                outputs = model(grey_imgs)
                loss = loss_fn(outputs, color_imgs)
                val_loss += loss.item() * batch_size

        val_loss = val_loss / len(val_loader.dataset)
        val_loss_data.append(val_loss)


        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}"
        )
        
        df = pd.DataFrame({'Train_loss': train_loss_data, 'Val_loss': val_loss_data})
        df.to_csv("training_data/"+train_data_file_name, index=False)

        if (val_loss < min_val_loss):
            min_val_loss = val_loss
            torch.save(model.state_dict(), "model/U_Net_meilleur_model.pth")
            print("Meilleur modèle sauvegardé")
        


    return model, (train_loss_data,val_loss_data)
