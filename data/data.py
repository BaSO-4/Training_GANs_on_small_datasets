from torchvision.datasets import Flowers102
from torchvision import transforms

def prepare_flowers102(root_dir, output_dir_name="flowers102_5k", sample_size=5000):
    print(f"Downloading Oxford 102 Flowers into {root_dir}/flowers102/jpg/ …")
    Flowers102(root=root_dir, split="train", download=True, transform=transforms.ToTensor())
    print("Done.")

if __name__ == "__main__":
    prepare_flowers102(".\\data")
