import os
import cv2
import numpy as np
import random

crop_size = 256
num_crops_per_image = 50
blur_min_length = 10
blur_max_length = 30
train_ratio = 0.9

clean_images_dir = "../processed-data"
dataset_dir = "deblur-dataset"
train_target_dir = os.path.join(dataset_dir, "train", "target")
train_input_dir  = os.path.join(dataset_dir, "train", "input")
val_target_dir   = os.path.join(dataset_dir, "val", "target")
val_input_dir    = os.path.join(dataset_dir, "val", "input")

for d in [train_target_dir, train_input_dir, val_target_dir, val_input_dir]:
    os.makedirs(d, exist_ok=True)

def motion_blur_kernel(kernel_size, angle, curvature):
    if abs(curvature) < 0.05:
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = np.ones(kernel_size, dtype=np.float32)
        kernel /= kernel_size
    else:
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = (kernel_size // 2, kernel_size // 2)
        arc = 60 * np.sign(curvature)
        startAngle = -arc / 2
        endAngle = arc / 2
        cv2.ellipse(kernel, center, (kernel_size // 2, kernel_size // 2),
                    angle, startAngle, endAngle, 1, -1)
        kernel_sum = np.sum(kernel)
        if kernel_sum != 0:
            kernel /= kernel_sum
    return kernel

def apply_color_motion_blur_rgb(image_rgb, kernel):
    R = cv2.filter2D(image_rgb[:, :, 0], -1, kernel, borderType=cv2.BORDER_REPLICATE)
    G = cv2.filter2D(image_rgb[:, :, 1], -1, kernel, borderType=cv2.BORDER_REPLICATE)
    B = cv2.filter2D(image_rgb[:, :, 2], -1, kernel, borderType=cv2.BORDER_REPLICATE)
    blurred = np.stack([R, G, B], axis=2)
    return blurred

def preprocess_patch_color(patch_bgr):
    denoised = cv2.fastNlMeansDenoisingColored(patch_bgr, None, h=10, hColor=10,
                                               templateWindowSize=7, searchWindowSize=21)
    background = cv2.GaussianBlur(denoised, (15, 15), 0)
    diff = cv2.subtract(denoised, background)
    normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    gamma = 0.8
    normalized_float = normalized.astype(np.float32) / 255.0
    gamma_corrected = np.power(normalized_float, gamma)
    gamma_corrected = np.uint8(gamma_corrected * 255)
    result_rgb = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB)
    return result_rgb

def generate_patches_and_split():
    train_count = 0
    val_count = 0
    for img_filename in os.listdir(clean_images_dir):
        img_path = os.path.join(clean_images_dir, img_filename)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Could not read image {img_path}. Skipping.")
            continue
        
        height, width = img_bgr.shape[:2]
        if height < crop_size or width < crop_size:
            print(f"Image {img_filename} is smaller than {crop_size}x{crop_size}. Skipping.")
            continue

        for i in range(num_crops_per_image):
            x = random.randint(0, width - crop_size)
            y = random.randint(0, height - crop_size)
            patch = img_bgr[y:y+crop_size, x:x+crop_size]
            processed_patch_rgb = preprocess_patch_color(patch)
            kernel_length = random.randint(blur_min_length, blur_max_length)
            if kernel_length % 2 == 0:
                kernel_length += 1
            angle = random.uniform(0, 360)
            curvature = random.uniform(-0.5, 0.8)
            kernel = motion_blur_kernel(kernel_length, angle, curvature)
            blurred_patch_rgb = apply_color_motion_blur_rgb(processed_patch_rgb, kernel)
            
            if random.random() < train_ratio:
                filename = f"patch_{train_count:04d}.png"
                cv2.imwrite(os.path.join(train_target_dir, filename), 
                            cv2.cvtColor(processed_patch_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(train_input_dir, filename), 
                            cv2.cvtColor(blurred_patch_rgb, cv2.COLOR_RGB2BGR))
                print(f"Saved TRAIN patch {train_count:04d} from {img_filename} (kernel: {kernel_length}px, angle: {angle:.2f}, curvature: {curvature:.2f})")
                train_count += 1
            else:
                filename = f"patch_{val_count:04d}.png"
                cv2.imwrite(os.path.join(val_target_dir, filename), 
                            cv2.cvtColor(processed_patch_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(val_input_dir, filename), 
                            cv2.cvtColor(blurred_patch_rgb, cv2.COLOR_RGB2BGR))
                print(f"Saved VAL patch {val_count:04d} from {img_filename} (kernel: {kernel_length}px, angle: {angle:.2f}, curvature: {curvature:.2f})")
                val_count += 1

if __name__ == "__main__":
    generate_patches_and_split()