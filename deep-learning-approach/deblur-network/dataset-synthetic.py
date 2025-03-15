import os
import cv2
import numpy as np
import random

crop_size = 256              
num_crops_per_image = 50     
blur_min_length = 5 # pixels
blur_max_length = 15 # pixels      
train_ratio = 0.9

clean_images_dir = "../processed-data"

dataset_dir = "training-dataset"
train_target_dir = os.path.join(dataset_dir, "train", "target")
train_input_dir  = os.path.join(dataset_dir, "train", "input")
val_target_dir   = os.path.join(dataset_dir, "val", "target")
val_input_dir    = os.path.join(dataset_dir, "val", "input")

for d in [train_target_dir, train_input_dir, val_target_dir, val_input_dir]:
    os.makedirs(d, exist_ok=True)

def motion_blur_kernel(kernel_size, angle):

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = np.ones(kernel_size, dtype=np.float32)
    kernel /= kernel_size

    center = (kernel_size // 2, kernel_size // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel_sum = np.sum(kernel)
    if kernel_sum != 0:
        kernel /= kernel_sum
    return kernel

def apply_motion_blur(image, kernel):

    blurred = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return blurred

def preprocess_patch(patch):

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(gray, None, h=10,
                                        templateWindowSize=7, searchWindowSize=21)
    
    background = cv2.GaussianBlur(denoised, (15, 15), 0)
    
    diff = cv2.subtract(denoised, background)
    
    normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    gamma = 0.8
    normalized_float = normalized.astype(np.float32) / 255.0
    gamma_corrected = np.power(normalized_float, gamma)
    gamma_corrected = np.uint8(gamma_corrected * 255)
    
    return gamma_corrected

def generate_patches_and_split():
    train_count = 0
    val_count = 0
    for img_filename in os.listdir(clean_images_dir):
        img_path = os.path.join(clean_images_dir, img_filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image {img_path}. Skipping.")
            continue
        
        height, width = img.shape[:2]
        if height < crop_size or width < crop_size:
            print(f"Image {img_filename} is smaller than {crop_size}x{crop_size}. Skipping.")
            continue

        for i in range(num_crops_per_image):
            x = random.randint(0, width - crop_size)
            y = random.randint(0, height - crop_size)
            patch = img[y:y+crop_size, x:x+crop_size]

            processed_patch = preprocess_patch(patch)

            kernel_length = random.randint(blur_min_length, blur_max_length)
            if kernel_length % 2 == 0:
                kernel_length += 1
            angle = random.uniform(0, 360)
            kernel = motion_blur_kernel(kernel_length, angle)
            blurred_patch = apply_motion_blur(processed_patch, kernel)
            
            if random.random() < train_ratio:
                gt_filename = f"patch_{train_count:04d}.png"
                cv2.imwrite(os.path.join(train_target_dir, gt_filename), processed_patch)
                cv2.imwrite(os.path.join(train_input_dir, gt_filename), blurred_patch)
                print(f"Saved TRAIN patch {train_count:04d} from image {img_filename} (kernel: {kernel_length}px, angle: {angle:.2f})")
                train_count += 1
            else:
                gt_filename = f"patch_{val_count:04d}.png"
                cv2.imwrite(os.path.join(val_target_dir, gt_filename), processed_patch)
                cv2.imwrite(os.path.join(val_input_dir, gt_filename), blurred_patch)
                print(f"Saved VAL patch {val_count:04d} from image {img_filename} (kernel: {kernel_length}px, angle: {angle:.2f})")
                val_count += 1

if __name__ == "__main__":
    generate_patches_and_split()
