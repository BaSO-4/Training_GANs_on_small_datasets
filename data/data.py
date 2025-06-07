from torchvision.datasets import Flowers102
from torchvision import transforms
import os, random, shutil

sample_size=4000
def prepare_flowers102(root_dir):
    print(f"Downloading Oxford 102 Flowers into {root_dir}/flowers-102/jpg/ …")
    Flowers102(root=root_dir, split="train", download=True, transform=transforms.ToTensor())
    print("Done.")
    
    jpg_dir = os.path.join(root_dir, "flowers102", "jpg")
    all_imgs = [f for f in os.listdir(jpg_dir) if f.lower().endswith(".jpg")]
    if len(all_imgs) < sample_size:
        raise ValueError(f"Only found {len(all_imgs)} images; need at least {sample_size}.")
    random.seed(42)
    chosen = random.sample(all_imgs, sample_size)

    out_dir = os.path.join(root_dir, "flowers102_5k")
    os.makedirs(out_dir, exist_ok=True)
    for fname in chosen:
        shutil.copy(os.path.join(jpg_dir, fname), os.path.join(out_dir, fname))
    print(f"Copied {sample_size} images to {out_dir}")

    # 4) Delete the original to save space
    # shutil.rmtree(os.path.join(root_dir, "flowers102"))
    # print(f"Removed original download folder {root_dir}/flowers102/")

if __name__ == "__main__":
    prepare_flowers102("./data")
