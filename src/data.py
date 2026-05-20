import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
from skimage import io, color
import numpy as np


# Affiche les n première images d'un dataset avec comme éléments (noisy img, img simple)
def show_data(data,n=4):
    fig, axis = plt.subplots(n)


    for i in range(n):
        grey_img = data[i][0][0].detach().cpu().numpy()
        axis[i].imshow(grey_img, cmap='gray')
        axis[i].axis('off')

def L_ab_to_Lab(Lab):
    stack = torch.cat((Lab[0],Lab[1])).permute(1,2,0).detach().numpy()
    return(stack)
    
#Affiche un tuple Lab de la forme (L, ab)
def show_L_ab(Lab):
    image_rgb = color.lab2rgb(L_ab_to_Lab(Lab))

    # Display
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()



# Dataset pour COCO
# Prends en input le chemin vers le dataset d'imgs et la transformation à effectué sur les imgs
# Retourne un éléments de la forme (L, ab)
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        # Load image in RGB format
        image_rgb = io.imread(img_name)

        if image_rgb.ndim == 2:
            image_rgb = color.gray2rgb(image_rgb)

        # Convert RGB to Lab
        image_lab = color.rgb2lab(image_rgb)

        # Split into L, a, b channels
        L = image_lab[:, :, 0]  # Lightness
        a = image_lab[:, :, 1]  # Green-Red axis
        b = image_lab[:, :, 2]  # Blue-Yellow axis

        if self.transform:
            L = self.transform(L).float()
            a = self.transform(a).float()
            b = self.transform(b).float()

        ab = torch.cat((a, b))

        return L, ab