import os
import cv2
import numpy as np
import random

crop_size = 256              
num_crops = 50               
train_ratio = 0.9           

resize_factor = 0.5 # Scale factor for resizing the original images (0.5 = half size)

gt_dir = "real-data-pairs/ground-truth"
blur_dir = "real-data-pairs/blurred"

dataset_dir = "real-dataset-aug"
train_target_dir = os.path.join(dataset_dir, "train", "target")
train_input_dir  = os.path.join(dataset_dir, "train", "input")
val_target_dir   = os.path.join(dataset_dir, "val", "target")
val_input_dir    = os.path.join(dataset_dir, "val", "input")

for d in [train_target_dir, train_input_dir, val_target_dir, val_input_dir]:
    os.makedirs(d, exist_ok=True)

def increase_contrast_clahe(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab_enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return enhanced

def denoise_preserve_stars(patch_bgr, star_threshold=200, 
                           bilateral_d=9, sigmaColor=75, sigmaSpace=75,
                           heavy_h=15, heavy_hColor=15):
    """
    Use a two-stage denoising strategy:
      1. Heavy non-local means denoising (fastNlMeansDenoisingColored) to reduce color noise.
      2. Light bilateral filter to preserve edges (and star details).
      
    A star mask is computed (brightness threshold) and used to blend:
      - Star areas: light denoised result (to keep stars sharp).
      - Background: heavy denoised result.
    """
    heavy_denoised = cv2.fastNlMeansDenoisingColored(
        patch_bgr, None, h=heavy_h, hColor=heavy_hColor,
        templateWindowSize=7, searchWindowSize=21
    )
    light_denoised = cv2.bilateralFilter(patch_bgr, bilateral_d, sigmaColor, sigmaSpace)
    
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    _, star_mask = cv2.threshold(gray, star_threshold, 255, cv2.THRESH_BINARY)
    star_mask = cv2.GaussianBlur(star_mask, (3, 3), 0)  # smooth mask
    star_mask = star_mask.astype(np.float32) / 255.0

    blended = (star_mask[..., None] * light_denoised.astype(np.float32) +
               (1 - star_mask[..., None]) * heavy_denoised.astype(np.float32)).astype(np.uint8)
    return blended

def flatten_background_morphological(image_bgr, kernel_size=31):
    image_bgr_flat = np.zeros_like(image_bgr)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    for c in range(3):
        channel = image_bgr[..., c]
        bg_est = cv2.morphologyEx(channel, cv2.MORPH_OPEN, kernel)
        top_hat = cv2.subtract(channel, bg_est)
        top_hat = cv2.normalize(top_hat, None, 0, 255, cv2.NORM_MINMAX)
        image_bgr_flat[..., c] = top_hat
    
    return image_bgr_flat

def preprocess_patch_color(patch_bgr):
    denoised = denoise_preserve_stars(patch_bgr)
    
    flattened = flatten_background_morphological(denoised, kernel_size=31)

    gamma = 0.9
    contrast_enhanced_float = flattened.astype(np.float32) / 255.0
    gamma_corrected = np.power(contrast_enhanced_float, gamma)
    gamma_corrected = np.uint8(gamma_corrected * 255)
    
    result_rgb = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB)
    return result_rgb

def random_augment(gt_patch, blur_patch):
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        M = cv2.getRotationMatrix2D((crop_size/2, crop_size/2), angle, 1.0)
        gt_patch = cv2.warpAffine(gt_patch, M, (crop_size, crop_size))
        blur_patch = cv2.warpAffine(blur_patch, M, (crop_size, crop_size))

    if random.random() < 0.5:
        gt_patch = cv2.flip(gt_patch, 1)
        blur_patch = cv2.flip(blur_patch, 1)
    
    return gt_patch, blur_patch

def get_star_crop_coordinates(img, crop_size, threshold=200, min_star_pixels=10, max_attempts=100):
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright_mask = gray > threshold
    bright_indices = np.argwhere(bright_mask)
    
    if bright_indices.size == 0:
        return random.randint(0, width - crop_size), random.randint(0, height - crop_size)
    
    for _ in range(max_attempts):
        idx = bright_indices[random.randint(0, len(bright_indices)-1)]
        y_center, x_center = idx
        x0 = max(0, min(x_center - crop_size // 2, width - crop_size))
        y0 = max(0, min(y_center - crop_size // 2, height - crop_size))
        crop_gray = gray[y0:y0+crop_size, x0:x0+crop_size]
        if np.sum(crop_gray > threshold) >= min_star_pixels:
            return x0, y0

    return random.randint(0, width - crop_size), random.randint(0, height - crop_size)

def resize_image(img, scale_factor):
    height, width = img.shape[:2]
    new_w = int(width * scale_factor)
    new_h = int(height * scale_factor)
    
    if new_w < 256 or new_h < 256:
        new_w = max(new_w, 256)
        new_h = max(new_h, 256)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return resized

def generate_real_dataset_from_dirs(gt_dir, blur_dir, crop_size, num_crops, train_ratio,
                                    train_target_dir, train_input_dir, 
                                    val_target_dir, val_input_dir, 
                                    resize_factor=1.0):
    gt_files = sorted(os.listdir(gt_dir))
    blur_files = sorted(os.listdir(blur_dir))
    
    paired_files = list(set(gt_files).intersection(set(blur_files)))
    paired_files.sort()

    if len(paired_files) == 0:
        raise ValueError("No matching files found between ground truth and blurred directories.")
    
    train_count = 0
    val_count = 0
    random.seed(42)
    
    for filename in paired_files:
        gt_path = os.path.join(gt_dir, filename)
        blur_path = os.path.join(blur_dir, filename)
        
        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        blur_img = cv2.imread(blur_path, cv2.IMREAD_COLOR)
        
        if gt_img is None or blur_img is None:
            print(f"Error reading pair {filename}. Skipping.")
            continue
        
        if gt_img.shape != blur_img.shape:
            print(f"Image pair {filename} have different dimensions. Skipping.")
            continue

        gt_resized = gt_img
        blur_resized = blur_img
        
        height, width = gt_resized.shape[:2]
        if height < crop_size or width < crop_size:
            print(f"Resized pair {filename} is smaller than {crop_size}x{crop_size}. Skipping.")
            continue
        
        for i in range(num_crops):
            x, y = get_star_crop_coordinates(gt_resized, crop_size)
            gt_patch = gt_resized[y:y+crop_size, x:x+crop_size]
            blur_patch = blur_resized[y:y+crop_size, x:x+crop_size]
            
            gt_patch, blur_patch = random_augment(gt_patch, blur_patch)

            processed_gt_rgb = preprocess_patch_color(gt_patch)
            processed_blur_rgb = preprocess_patch_color(blur_patch)
            
            if random.random() < train_ratio:
                out_filename = f"patch_{train_count:04d}.png"
                cv2.imwrite(os.path.join(train_target_dir, out_filename),
                            cv2.cvtColor(processed_gt_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(train_input_dir, out_filename),
                            cv2.cvtColor(processed_blur_rgb, cv2.COLOR_RGB2BGR))
                print(f"Saved TRAIN patch {train_count:04d} from {filename} at (x={x}, y={y})")
                train_count += 1
            else:
                out_filename = f"patch_{val_count:04d}.png"
                cv2.imwrite(os.path.join(val_target_dir, out_filename),
                            cv2.cvtColor(processed_gt_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(val_input_dir, out_filename),
                            cv2.cvtColor(processed_blur_rgb, cv2.COLOR_RGB2BGR))
                print(f"Saved VAL patch {val_count:04d} from {filename} at (x={x}, y={y})")
                val_count += 1

if __name__ == "__main__":
    generate_real_dataset_from_dirs(
        gt_dir, blur_dir, crop_size, num_crops, train_ratio,
        train_target_dir, train_input_dir, val_target_dir, val_input_dir,
        resize_factor=resize_factor
    )
    print("Real dataset generation complete.")
