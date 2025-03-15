import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from tqdm import tqdm

def estimate_psf_from_brightest_star(image, patch_size=61, thresh_ratio=0.5):
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(image)
    print(f"[INFO] Brightest pixel value: {maxVal} at location {maxLoc}")
    
    half = patch_size // 2
    x, y = maxLoc
    x1 = max(x - half, 0)
    y1 = max(y - half, 0)
    x2 = min(x + half + 1, image.shape[1])
    y2 = min(y + half + 1, image.shape[0])
    patch = image[y1:y2, x1:x2].astype(np.float32)
    
    patch_max = patch.max()
    thresh_value = thresh_ratio * patch_max
    _, patch_thresh = cv2.threshold(patch, thresh_value, 255, cv2.THRESH_BINARY)
    patch_filtered = patch * (patch_thresh / 255)
    
    psf = patch_filtered.copy()
    psf_sum = psf.sum()
    if psf_sum > 0:
        psf /= psf_sum
    else:
        print("[WARNING] The extracted PSF patch is all zeros!")
    
    return psf, patch, patch_thresh

def richardson_lucy_deblur(blurred, psf, num_iterations=30):
    latent = blurred.copy()
    psf_mirror = np.flip(psf)
    
    for i in tqdm(range(num_iterations), desc="Richardson-Lucy Iterations"):
        estimated_blur = fftconvolve(latent, psf, mode='same')
        relative_blur = blurred / (estimated_blur + 1e-12)
        error_estimate = fftconvolve(relative_blur, psf_mirror, mode='same')
        error_estimate = np.clip(error_estimate, 0, 10)
        
        latent *= error_estimate
        latent = np.nan_to_num(latent, nan=0, posinf=1, neginf=0)
        latent = np.clip(latent, 0, 1)
    
    return latent

def main():

    image_path = 'star_trail_landscape.jpg'
    color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if color_image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    print("[INFO] Loaded color image:")
    print(f"       Shape: {color_image.shape}, Dtype: {color_image.dtype}")
    
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    color_float = color_image.astype(np.float32) / 255.0
    
    patch_size = 61       
    thresh_ratio = 0.5
    psf, patch, patch_thresh = estimate_psf_from_brightest_star(gray_image, patch_size, thresh_ratio)
    
    cv2.imwrite("psf_patch.png", patch)
    cv2.imwrite("psf_patch_thresh.png", patch_thresh)
    print(f"[INFO] PSF patch sum (should be > 0): {psf.sum()}")

    num_iterations = 30
    channels = cv2.split(color_float)
    deblurred_channels = []
    for idx, ch in enumerate(channels):
        print(f"[INFO] Deblurring channel {idx}...")
        deblurred_ch = richardson_lucy_deblur(ch, psf, num_iterations)
        deblurred_channels.append(np.clip(deblurred_ch, 0, 1))

    deblurred_color = cv2.merge(deblurred_channels)

    deblurred_uint8 = (deblurred_color * 255).astype(np.uint8)

    denoised = cv2.fastNlMeansDenoisingColored(deblurred_uint8, None, h=25, hColor=25, 
                                                templateWindowSize=7, searchWindowSize=21)
    

    deblurred_gray = cv2.cvtColor(deblurred_uint8, cv2.COLOR_BGR2GRAY)
    _, star_mask = cv2.threshold(deblurred_gray, 150, 255, cv2.THRESH_BINARY)
    star_mask = star_mask.astype(np.float32) / 255.0
    star_mask_3c = cv2.merge([star_mask, star_mask, star_mask])
    
    final_img = (star_mask_3c * deblurred_uint8 + (1 - star_mask_3c) * denoised).astype(np.uint8)
    
    color_rgb = cv2.cvtColor((color_float*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    deblurred_rgb = cv2.cvtColor(deblurred_uint8, cv2.COLOR_BGR2RGB)
    denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 5, 1)
    plt.title('Original Color Image')
    plt.imshow(color_rgb)
    plt.axis('off')
    
    plt.subplot(1, 5, 2)
    plt.title('PSF Patch (Thresh)')
    plt.imshow(patch_thresh, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 5, 3)
    plt.title('Deblurred Color Image')
    plt.imshow(deblurred_rgb)
    plt.axis('off')
    
    plt.subplot(1, 5, 4)
    plt.title('Denoised Image')
    plt.imshow(denoised_rgb)
    plt.axis('off')
    
    plt.subplot(1, 5, 5)
    plt.title('Final Blended Result')
    plt.imshow(final_rgb)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    filename = os.path.splitext(os.path.basename(image_path))[0]
    outputname = f"{filename}_deblurdenoised"
    cv2.imwrite(f"{outputname}.png", final_img)
    print(f"[INFO] Final deblurred and denoised image saved as '{outputname}.png'")

if __name__ == "__main__":
    main()