import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from tqdm import tqdm
from skimage.color import lab2rgb
from .data import L_ab_to_Lab
from skimage import io, color
import numpy as np

# Prend deux listes de même taille en entré
# Affiche un graphique représentant l'avance des fonctions de loss sur le dataset de training et de validation
def plot_loss_curves(train_loss,val_loss):
    plt.figure(figsize=(8, 5))

    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()

    plt.grid(True)
    plt.show()

# Prend un modèle et une image en entrée
# Affiche l'image originale suivi de l'image en gris puis l'image en gris après être passé dans le modèle
def visualize(model,imgs):
    model.eval()

    L = imgs[0]
    ab = imgs[1]


    base_Lab = L_ab_to_Lab((L,ab))
    base_rgb = lab2rgb(base_Lab)

    pred_ab = model(L.unsqueeze(0)).squeeze(0)

    pred_Lab = L_ab_to_Lab((L,pred_ab))
    pred_rgb = lab2rgb(pred_Lab)

    fig, axis = plt.subplots(1, 3)


    axis[0].imshow(base_rgb)

    axis[1].imshow(imgs[0].permute(1,2,0))

    axis[2].imshow(pred_rgb)

    axis[0].axis('off')
    axis[1].axis('off')
    axis[2].axis('off')

    plt.show()

def compute_mse(denoised, clean):
    return F.mse_loss(denoised, clean)

def compute_psnr(denoised, clean, data_range=1.0):
    mse = compute_mse(denoised, clean)
    return 20 * torch.log10(data_range / torch.sqrt(mse))

def compute_ssim(denoised, clean, data_range=1.0):
    return ssim(denoised, clean, data_range=data_range, size_average=True)

# Prend en input un modèle ainsi qu'un dataset de test
# Renvoie plusieurs métrique de performance du modèle sur le dataset
def evaluate_model_metrics(model, test_dataset, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    with torch.no_grad():
        for noisy_img, clean_img in tqdm(test_loader, desc="Processing"):
            noisy_img, clean_img = noisy_img.to(device), clean_img.to(device)

            denoised_img = model(noisy_img)

            mse = compute_mse(denoised_img, clean_img)
            psnr = compute_psnr(denoised_img, clean_img)
            ssim_val = compute_ssim(denoised_img, clean_img)

            total_mse += mse.item() * noisy_img.size(0)
            total_psnr += psnr.item() * noisy_img.size(0)
            total_ssim += ssim_val.item() * noisy_img.size(0)

    avg_mse = total_mse / len(test_loader.dataset)
    avg_psnr = total_psnr / len(test_loader.dataset)
    avg_ssim = total_ssim / len(test_loader.dataset)

    return {
        "MSE": avg_mse,
        "PSNR": avg_psnr,
        "SSIM": avg_ssim
    }

# Prend un modèle et le chemin vers une image
# Affiche l'image originale suivi de l'image en gris puis l'image en gris après être passé dans le modèle
def color_img(model,path,save=False):

    # Load image in RGB format
    image_rgb = io.imread(path)

    if image_rgb.ndim == 2:
        image_rgb = color.gray2rgb(image_rgb)

    # Convert RGB to Lab
    image_lab = color.rgb2lab(image_rgb)

    # Split into L, a, b channels
    L = image_lab[:, :, 0]  # Lightness
    a = image_lab[:, :, 1]  # Green-Red axis
    b = image_lab[:, :, 2]  # Blue-Yellow axis
    
    L = torch.from_numpy(L).unsqueeze(0).float()
    a = torch.from_numpy(a).unsqueeze(0).float()
    b = torch.from_numpy(b).unsqueeze(0).float()

    ab = torch.cat((a, b))
    
    model.eval()

    base_Lab = L_ab_to_Lab((L,ab))
    base_rgb = lab2rgb(base_Lab)

    pred_ab = model(L.unsqueeze(0)).squeeze(0)

    pred_Lab = L_ab_to_Lab((L,pred_ab))
    pred_rgb = lab2rgb(pred_Lab)

    plt.imshow(base_rgb)
    plt.axis("off")
    bbox_inches='tight'
    plt.show()

    # Normalize and convert to RGB grayscale
    L_normalized = L[0] / 100.0  
    gray_rgb = np.stack([L_normalized] * 3, axis=-1)
    plt.imshow(gray_rgb)    
    plt.axis("off")
    bbox_inches='tight'
    plt.show()

    plt.imshow(pred_rgb)
    plt.axis("off")
    bbox_inches='tight'

    if save != False:
        plt.savefig(save, bbox_inches='tight', pad_inches=0)
    plt.show()
