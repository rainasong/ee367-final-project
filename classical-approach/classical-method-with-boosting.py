import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from tqdm import tqdm

def remove_background_morph(image_float, radius=50, removal_strength=1.0):
    img8u = (image_float * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    background = cv2.morphologyEx(img8u, cv2.MORPH_OPEN, kernel)
    background_float = background.astype(np.float32) / 255.0
    result = image_float - removal_strength * background_float
    result = np.clip(result, 0, 1)
    return result, background_float

def remove_background_morph_color(image_float, radius=50, removal_strength=1.0):
    channels = cv2.split(image_float)
    processed_channels = []
    backgrounds = []
    for ch in channels:
        proc, bg = remove_background_morph(ch, radius, removal_strength)
        processed_channels.append(proc)
        backgrounds.append(bg)
    result = cv2.merge(processed_channels)
    bg_total = cv2.merge(backgrounds)
    return result, bg_total

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
    # 1. Load the blurred color image.
    image_path = 'star_trail_landscape.jpg'
    color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if color_image is None:
        raise ValueError(f"Could not load image from {image_path}")
    print(f"[INFO] Loaded image: {color_image.shape}, dtype: {color_image.dtype}")
    color_float = color_image.astype(np.float32) / 255.0

    # 2. Remove background gradient.
    radius = 50
    removal_strength = 0.5
    color_no_gradient, gradient = remove_background_morph_color(color_float, radius, removal_strength)
    cv2.imwrite("background_gradient.png", (gradient * 255).astype(np.uint8))
    print(f"[INFO] Partial background gradient removed with removal_strength={removal_strength}")

    # 3. PSF estimation from gradient-removed image.
    gray_for_psf = cv2.cvtColor((color_no_gradient * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    psf, patch, patch_thresh = estimate_psf_from_brightest_star(gray_for_psf, patch_size=61, thresh_ratio=0.5)
    cv2.imwrite("psf_patch.png", patch)
    cv2.imwrite("psf_patch_thresh.png", patch_thresh)

    # 4. Richardson-Lucy deblurring for each channel.
    channels = cv2.split(color_no_gradient)
    deblurred_channels = []
    for idx, ch in enumerate(channels):
        print(f"[INFO] Deblurring channel {idx}...")
        deblurred_ch = richardson_lucy_deblur(ch, psf, num_iterations=30)
        deblurred_channels.append(deblurred_ch)
    deblurred_color = cv2.merge(deblurred_channels)

    # 5. Denoise the deblurred image using non-local means.
    deblurred_uint8 = (deblurred_color * 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoisingColored(deblurred_uint8, None, h=10, hColor=10,
                                                templateWindowSize=7, searchWindowSize=21)

    # 6.Boost star brightness via a star mask.
    deblurred_gray = cv2.cvtColor(deblurred_uint8, cv2.COLOR_BGR2GRAY)
    _, star_mask = cv2.threshold(deblurred_gray, 150, 255, cv2.THRESH_BINARY)
    star_mask_f = star_mask.astype(np.float32) / 255.0
    star_mask_3c = cv2.merge([star_mask_f, star_mask_f, star_mask_f])
    star_boost_factor = 1.8
    boosted = np.clip(deblurred_uint8.astype(np.float32) * star_boost_factor, 0, 255).astype(np.uint8)
    
    # Blend: For star regions (mask=1) keep the original deblurred values; for background (mask=0) use the denoised image.
    final_img = (star_mask_3c * boosted + (1 - star_mask_3c) * denoised).astype(np.uint8)

    orig_rgb = cv2.cvtColor((color_float * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    grad_rgb = cv2.cvtColor((gradient * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    nograd_rgb = cv2.cvtColor((color_no_gradient * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    deb_rgb = cv2.cvtColor(deblurred_uint8, cv2.COLOR_BGR2RGB)
    den_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    boost_rgb = cv2.cvtColor(boosted, cv2.COLOR_BGR2RGB)
    fin_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(28, 5))
    plt.subplot(1,7,1)
    plt.title('Original')
    plt.imshow(orig_rgb)
    plt.axis('off')
    
    plt.subplot(1,7,2)
    plt.title('Gradient')
    plt.imshow(grad_rgb)
    plt.axis('off')
    
    plt.subplot(1,7,3)
    plt.title('No Gradient')
    plt.imshow(nograd_rgb)
    plt.axis('off')
    
    plt.subplot(1,7,4)
    plt.title('Deblurred')
    plt.imshow(deb_rgb)
    plt.axis('off')
    
    plt.subplot(1,7,5)
    plt.title('Denoised')
    plt.imshow(den_rgb)
    plt.axis('off')
    
    plt.subplot(1,7,6)
    plt.title('Boosted Stars')
    plt.imshow(boost_rgb)
    plt.axis('off')
    
    plt.subplot(1,7,7)
    plt.title('Final Output')
    plt.imshow(fin_rgb)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    filename = os.path.splitext(os.path.basename(image_path))[0]
    outputname = f"{filename}_boosted"
    cv2.imwrite(f"{outputname}.png", final_img)
    print(f"[INFO] Final image saved as '{outputname}.png'")

if __name__ == "__main__":
    main()
