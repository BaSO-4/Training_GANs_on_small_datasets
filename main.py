import argparse
from train import train
from generate import generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StyleGAN2-ADA Transfer Learning / Generation")
    # General arguments
    parser.add_argument('--mode', choices=['train', 'generate'], required=True, help="Either 'train' to train or fine-tune, or 'generate' to sample images.")
    parser.add_argument('--device', type=str, default='cuda', help="cuda or cpu")
    parser.add_argument('--resolution',   type=int, default=256, help='Image resolution (HxW)')
    parser.add_argument('--W_dim',   type=int, default=256, help='Dimensionality of latent w')
    parser.add_argument('--Z_dim',   type=int, default=256, help='Dimensionality of latent z')
    
    # Training arguments
    parser.add_argument('--data_dir',         type=str, default=".\\data", help='Path to dataset root (for training)')
    parser.add_argument('--batch_size',        type=int, default=16, help='Batch size per iteration')
    parser.add_argument('--lr',           type=float, default=1e-3, help='Learning rate for G and D')
    parser.add_argument('--epochs',     type=float, default=300, help='How long to train for')
    parser.add_argument('--lmbd',   type=int, default=10, help='Weight for the gradient loss of D')
    parser.add_argument('--ada_num_img',     type=float, default=500000, help='ADA adjustment speed')
    parser.add_argument('--save_dir',     type=str, help='Dir to sve the models')

    # Generation arguments
    parser.add_argument('--num',        type=int, default=25, help='Number of images to generate')
    parser.add_argument('--seed',       type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--output_dir', type=str, help='Where to save generated grid')
    parser.add_argument('--models_path', type=str, help='Where to get the models')

    args = parser.parse_args()

    if args.mode == 'train':
        assert args.data is not None, "--data is required for training"
        assert args.save_dir is not None, "--save_dir is required for training"
        train(args.save_dir, args.data_dir, args.batch_size, args.lr, args.epochs, args.resolution, args.W_dim, args.Z_dim, args.lmbd, args.ada_num_img, args.device)
    elif args.mode == 'generate':
        assert args.models_path is not None, "--models_path is required for generation"
        assert args.output_dir is not None, "--output_dir is required for generation"
        generate(args.models_path, args.output_dir, args.num, args.seed, args.resolution, args.W_dim, args.Z_dim)