# Sujet de projet 12: Colorization of Grayscale Photos

### Overview

Automatically predict plausible color for a grayscale input image. Start with a
U-Net and optionally explore other models like GANs to get better results.

### Objectives

• Implement a U-Net (or encoder–decoder) that takes the lightness channel and
predicts (a, b) in CIELab color. Compare loss functions: L2 in Lab vs.
perceptual (VGG) etc.

• Evaluate the outputs using PSNR/SSIM in RGB; explore other metrics for
perceptual quality and colorfulness; analyze failure cases.

### Dataset

COCO
Use the COCO dataset to get a natural image set. convert to grayscale for inputs
and keep color as targets.