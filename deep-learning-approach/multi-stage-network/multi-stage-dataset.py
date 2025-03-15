"""

Generates 4 separate datasets:
 - Stage 1: random patches from raw => denoised, with train/val split
 - Stage 2: from Stage1 -> background removed
 - Stage 3: from Stage2 -> add star streaks => final deburring data
 - Stage 4: from Stage2 -> add noise, partial background reintroduction, star streak, all at once

We only do a random split in Stage1. Stages 2, 3, 4 reuse the same filenames from the previous stage.

"""

import os
import cv2
import numpy as np
import random
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

root_dir = ""
raw_data_dir = os.path.join(root_dir, "../processed-data")

stage1_dir = os.path.join(root_dir, "stage1_denoising-mid-2")
stage2_dir = os.path.join(root_dir, "stage2_bg_removed-mid-2")
stage3_dir = os.path.join(root_dir, "stage3_final-mid-2")
stage4_dir = os.path.join(root_dir, "stage4_allinone-mid-2")

for d in [stage1_dir, stage2_dir, stage3_dir, stage4_dir]:
    for split in ["train/input", "train/target", "val/input", "val/target"]:
        os.makedirs(os.path.join(d, split), exist_ok=True)

# Stage1: random patches + train/val split
def mild_denoise(img_bgr):
    """Example mild color denoise."""
    denoised = cv2.fastNlMeansDenoisingColored(
        img_bgr, None, h=10, hColor=10,
        templateWindowSize=7, searchWindowSize=21
    )
    return denoised

def save_image(img, path):
    cv2.imwrite(path, img)

def split_train_val(files, test_size=0.2, random_seed=42):
    if len(files) < 2:
        return files, []
    return train_test_split(files, test_size=test_size, random_state=random_seed)

def generate_stage1_dataset(crop_size=256, patches_per_image=50):
    print("[Stage1] Generating dataset with random 256×256 patches from raw images.")
    raw_files = sorted(glob(os.path.join(raw_data_dir, "*.png")))
    if not raw_files:
        print(f"No raw images found in {raw_data_dir}.")
        return

    all_patches = []
    patch_index = 0
    for f in raw_files:
        img_bgr = cv2.imread(f)
        if img_bgr is None:
            print(f"Could not read {f}, skipping.")
            continue
        H,W = img_bgr.shape[:2]
        base_name = os.path.splitext(os.path.basename(f))[0]

        for _ in tqdm(range(patches_per_image), desc=f"Stage1 Cropping {base_name}", leave=False):
            if H<crop_size or W<crop_size:
                continue
            x = random.randint(0, W - crop_size)
            y = random.randint(0, H - crop_size)
            patch = img_bgr[y:y+crop_size, x:x+crop_size].copy()
            patch_filename = f"{base_name}_patch{patch_index:05d}.png"
            all_patches.append((patch, patch_filename))
            patch_index+=1
    
    if not all_patches:
        print("No patches generated. Check data or crop_size.")
        return

    train_items, val_items = split_train_val(all_patches, test_size=0.2, random_seed=42)

    for (patch, filename) in tqdm(train_items, desc="Stage1 Train Save"):
        denoised = mild_denoise(patch)
        cv2.imwrite(os.path.join(stage1_dir, "train", "input",  filename), patch)
        cv2.imwrite(os.path.join(stage1_dir, "train", "target", filename), denoised)

    for (patch, filename) in tqdm(val_items, desc="Stage1 Val Save"):
        denoised = mild_denoise(patch)
        cv2.imwrite(os.path.join(stage1_dir, "val", "input",  filename), patch)
        cv2.imwrite(os.path.join(stage1_dir, "val", "target", filename), denoised)

    print(f"[Stage1] Done => total patches {len(all_patches)} => train={len(train_items)}, val={len(val_items)}")


# Stage2: from Stage1's target => background removal
def background_removal(img_bgr):
    """Large Gaussian blur, partial subtraction for background removal."""
    blur = cv2.GaussianBlur(img_bgr, (31,31), 0)
    alpha = 0.9 # lower alpha => more background retained
    float_img = img_bgr.astype(np.float32)
    float_bg  = blur.astype(np.float32)
    sub = float_img - alpha*float_bg
    sub = np.clip(sub, 0, 255).astype(np.uint8)
    return sub

def generate_stage2_dataset():
    print("[Stage2] Using EXACT Stage1 train/val filenames. No new random split.")
    stage1_train_input = sorted(os.listdir(os.path.join(stage3_dir, "train", "target")))
    stage1_val_input   = sorted(os.listdir(os.path.join(stage3_dir, "val",   "target")))

    for filename in tqdm(stage1_train_input, desc="Stage2 Train"):
        stage1_target_path = os.path.join(stage3_dir, "train", "target", filename)
        img_bgr = cv2.imread(stage1_target_path)
        if img_bgr is None:
            continue
        bg_removed = background_removal(img_bgr)
        cv2.imwrite(os.path.join(stage2_dir, "train", "input",  filename), img_bgr)
        cv2.imwrite(os.path.join(stage2_dir, "train", "target", filename), bg_removed)

    for filename in tqdm(stage1_val_input, desc="Stage2 Val"):
        stage1_target_path = os.path.join(stage3_dir, "val", "target", filename)
        img_bgr = cv2.imread(stage1_target_path)
        if img_bgr is None:
            continue
        bg_removed = background_removal(img_bgr)
        cv2.imwrite(os.path.join(stage2_dir, "val", "input",  filename), img_bgr)
        cv2.imwrite(os.path.join(stage2_dir, "val", "target", filename), bg_removed)

    print("[Stage2] done => filenames match Stage1 exactly.")


# Stage3: star streak => input, stage2 input => final GT
blur_min_length=8
blur_max_length=25
p_curved=0.6
curvature_min=-0.5
curvature_max=0.9

def linear_motion_blur_kernel(kernel_size, angle):
    kernel = np.zeros((kernel_size, kernel_size), np.float32)
    kernel[kernel_size//2, :] = np.ones(kernel_size, np.float32)
    kernel /= kernel_size
    center = (kernel_size//2, kernel_size//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    ksum = np.sum(kernel)
    if ksum>1e-8:
        kernel/=ksum
    return kernel

def curved_motion_blur_kernel(kernel_size, angle, curvature):
    kernel = np.zeros((kernel_size, kernel_size), np.float32)
    num_points = kernel_size
    t = np.linspace(0,1,num_points)
    x = t*(kernel_size-1)
    y = curvature*(t-0.5)**2*kernel_size + (kernel_size-1)/2
    x = np.clip(np.round(x).astype(int),0,kernel_size-1)
    y = np.clip(np.round(y).astype(int),0,kernel_size-1)
    for i in range(num_points-1):
        pt1=(x[i], y[i])
        pt2=(x[i+1],y[i+1])
        cv2.line(kernel, pt1, pt2, color=1, thickness=1)
    s=np.sum(kernel)
    if s>1e-8:
        kernel/=s
    center=(kernel_size//2,kernel_size//2)
    M=cv2.getRotationMatrix2D(center, angle,1.0)
    kernel=cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    s2=np.sum(kernel)
    if s2>1e-8:
        kernel/=s2
    return kernel

def apply_color_motion_blur_rgb(image_rgb, kernel):
    R = cv2.filter2D(image_rgb[:,:,0], -1, kernel, borderType=cv2.BORDER_REPLICATE)
    G = cv2.filter2D(image_rgb[:,:,1], -1, kernel, borderType=cv2.BORDER_REPLICATE)
    B = cv2.filter2D(image_rgb[:,:,2], -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return np.stack([R,G,B], axis=2)

def add_star_streaks_random(img_bgr):
    """
    Randomly produce a star streak on the background-removed patch 'img_bgr'.
    Then we brighten the entire patch by some factor to ensure the streak is more visible.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    k_size = random.randint(blur_min_length, blur_max_length)
    if k_size%2==0:
        k_size+=1
    angle = random.uniform(0,360)
    if random.random() < p_curved:
        curvature = random.uniform(curvature_min, curvature_max)
        kernel = curved_motion_blur_kernel(k_size, angle, curvature)
    else:
        kernel = linear_motion_blur_kernel(k_size, angle)

    streaked_rgb = apply_color_motion_blur_rgb(img_rgb, kernel)
    bright_factor = random.uniform(1.2, 2.0)
    streaked_f = streaked_rgb.astype(np.float32)*bright_factor
    streaked_f = np.clip(streaked_f,0,255)
    streaked_rgb = streaked_f.astype(np.uint8)

    return streaked_rgb

def generate_stage3_dataset():
    print("[Stage3] Generating star-streaked => same filenames from stage2.")
    train_input_files = sorted(os.listdir(os.path.join(stage2_dir, "train", "target")))
    val_input_files   = sorted(os.listdir(os.path.join(stage2_dir, "val",   "target")))

    for fname in tqdm(train_input_files, desc="Stage3 Train"):
        stage2_input_path  = os.path.join(stage1_dir, "train", "input",  fname)
        stage2_target_path = os.path.join(stage1_dir, "train", "target", fname)
        clean_bgr   = cv2.imread(stage2_input_path)
        bgrem_bgr   = cv2.imread(stage2_target_path)
        if clean_bgr is None or bgrem_bgr is None:
            continue
        streaked_rgb = add_star_streaks_random(bgrem_bgr)
        streaked_bgr = cv2.cvtColor(streaked_rgb, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(stage3_dir, "train", "input",  fname), streaked_bgr)
        cv2.imwrite(os.path.join(stage3_dir, "train", "target", fname), bgrem_bgr)

    for fname in tqdm(val_input_files, desc="Stage3 Val"):
        stage2_input_path  = os.path.join(stage1_dir, "val", "input",  fname)
        stage2_target_path = os.path.join(stage1_dir, "val", "target", fname)
        clean_bgr = cv2.imread(stage2_input_path)
        bgrem_bgr = cv2.imread(stage2_target_path)
        if clean_bgr is None or bgrem_bgr is None:
            continue

        streaked_rgb = add_star_streaks_random(bgrem_bgr)
        streaked_bgr = cv2.cvtColor(streaked_rgb, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(stage3_dir, "val", "input",  fname), streaked_bgr)
        cv2.imwrite(os.path.join(stage3_dir, "val", "target", fname), bgrem_bgr)

    print("[Stage3] done => star-streaked (brighter), same filenames from stage2.")


# Stage4: all-in-one degrade => noise + partial BG + star streak
def add_noise(img_bgr, sigma=10):
    noise = np.random.normal(0, sigma, img_bgr.shape).astype(np.float32)
    float_img = img_bgr.astype(np.float32)
    noisy = float_img + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def partial_background_reintroduce(img_bgr):
    blur = cv2.GaussianBlur(img_bgr, (31,31),0)
    alpha = 0.4
    float_img = img_bgr.astype(np.float32)
    float_bg  = blur.astype(np.float32)
    re = float_img + alpha * float_bg
    re = np.clip(re,0,255).astype(np.uint8)
    return re

def degrade_all_in_one(img_bgr):
    sigma = random.randint(5,25)
    noisy = add_noise(img_bgr, sigma=sigma)
    partial_bg = partial_background_reintroduce(noisy)

    streaked_rgb = add_star_streaks_random(partial_bg)
    final_bgr = cv2.cvtColor(streaked_rgb, cv2.COLOR_RGB2BGR)
    return final_bgr

def generate_stage4_dataset():
    print("[Stage4] Generating all-in-one degrade => noise + partial BG + star streak => from stage2 input (clean).")
    stage2_train_input = sorted(os.listdir(os.path.join(stage2_dir, "train", "input")))
    stage2_val_input   = sorted(os.listdir(os.path.join(stage2_dir, "val",   "input")))

    os.makedirs(os.path.join(stage4_dir,"train","input"), exist_ok=True)
    os.makedirs(os.path.join(stage4_dir,"train","target"),exist_ok=True)
    os.makedirs(os.path.join(stage4_dir,"val","input"),  exist_ok=True)
    os.makedirs(os.path.join(stage4_dir,"val","target"), exist_ok=True)

    # train
    for fname in tqdm(stage2_train_input, desc="Stage4 Train"):
        stage2_input_path = os.path.join(stage2_dir, "train", "input", fname)
        clean_bgr = cv2.imread(stage2_input_path)
        if clean_bgr is None:
            continue
        degrade_bgr = degrade_all_in_one(clean_bgr)
        cv2.imwrite(os.path.join(stage4_dir,"train","input", fname), degrade_bgr)
        cv2.imwrite(os.path.join(stage4_dir,"train","target",fname), clean_bgr)

    # val
    for fname in tqdm(stage2_val_input, desc="Stage4 Val"):
        stage2_input_path = os.path.join(stage2_dir, "val", "input", fname)
        clean_bgr = cv2.imread(stage2_input_path)
        if clean_bgr is None:
            continue
        degrade_bgr = degrade_all_in_one(clean_bgr)
        cv2.imwrite(os.path.join(stage4_dir,"val","input", fname), degrade_bgr)
        cv2.imwrite(os.path.join(stage4_dir,"val","target",fname), clean_bgr)

    print("[Stage4] done => 'all-in-one' degrade => noise + partial BG + star streak.")


# -----------------------------------------------------------
if __name__ == "__main__":
    generate_stage1_dataset(crop_size=256, patches_per_image=100)
    generate_stage3_dataset()
    generate_stage2_dataset()
    generate_stage4_dataset()
    print("All 4 stages data generation complete.")
