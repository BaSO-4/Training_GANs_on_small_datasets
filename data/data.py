from torchvision.datasets import Flowers102
from torchvision import transforms
import os, random, shutil

sample_size=4000
def prepare_flowers102(root_dir):
    print(f"Downloading Oxford 102 Flowers into {root_dir}/flowers-102/jpg/ …")
    Flowers102(root=root_dir, split="train", download=True, transform=transforms.ToTensor())
    print("Done.")
    
    jpg_dir = os.path.join(root_dir, "flowers-102", "jpg")
    all_imgs = [f for f in os.listdir(jpg_dir) if f.lower().endswith(".jpg")]
    chosen = random.sample(all_imgs, sample_size)
    for fname in all_imgs:
        if fname not in chosen:
            os.remove(os.path.join(jpg_dir, fname))
            
if __name__ == "__main__":
    prepare_flowers102("./data")
