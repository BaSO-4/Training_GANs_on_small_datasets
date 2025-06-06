import argparse
from train import train
from generate import generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StyleGAN2-ADA Transfer Learning / Generation")
    # General arguments
    parser.add_argument('--mode', choices=['train', 'generate'], required=True, help="Either 'train' to train or fine-tune, or 'generate' to sample images.")
    parser.add_argument('--device', type=str, default='cuda', help="cuda or cpu")
    
    # Training arguments
    parser.add_argument('--data',         type=str, default=".\\data", help='Path to dataset root (for training)')
    parser.add_argument('--outdir',       type=str, help='Output directory to save checkpoints')
    parser.add_argument('--batch',        type=int, default=32, help='Batch size per iteration')
    parser.add_argument('--resolution',   type=int, default=256, help='Image resolution (HxW)')
    parser.add_argument('--latent_dim',   type=int, default=512, help='Dimensionality of latent z & w')
    parser.add_argument('--r1',           type=float, default=10.0, help='R1 regularization weight')
    parser.add_argument('--ema',          type=float, default=0.999, help='EMA decay for generator')
    parser.add_argument('--lr',           type=float, default=2.5e-4, help='Learning rate for G and D')
    parser.add_argument('--total_kimg',   type=int, default=2500, help='Total thousands of images to train on')
    parser.add_argument('--ada_target',   type=float, default=0.6, help='ADA target real accuracy')
    parser.add_argument('--ada_interval', type=int, default=4, help='ADA update interval (D steps)')
    parser.add_argument('--ada_kimg',     type=float, default=500, help='ADA adjustment speed (kimg)')
    parser.add_argument('--log_interval', type=int, default=100, help='Steps between logging')
    
    # Transfer learning arguments
    parser.add_argument('--pretrained_g', type=str, default=None, help="Path to pretrained Generator checkpoint for fine-tuning")
    parser.add_argument('--pretrained_d', type=str, default=None, help="Path to pretrained Discriminator checkpoint (optional)")
    parser.add_argument('--freeze_upto', type=int, default=None, help="Freeze G.blocks[0..freeze_upto-1] during fine-tuning")

    # Generation arguments
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint with G_ema (for generation)')
    parser.add_argument('--num',        type=int, default=25, help='Number of images to generate')
    parser.add_argument('--seed',       type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--output_dir', type=str, default='.\\generations', help='Where to save generated grid')

    args = parser.parse_args()

    if args.mode == 'train':
        assert args.data is not None, "--data is required for training"
        assert args.outdir is not None, "--outdir is required for training"
        train(
            data_dir=args.data,
            outdir=args.outdir,
            batch_size=args.batch,
            resolution=args.resolution,
            latent_dim=args.latent_dim,
            r1_gamma=args.r1,
            ema_beta=args.ema,
            lr=args.lr,
            total_kimg=args.total_kimg,
            ada_target=args.ada_target,
            ada_interval=args.ada_interval,
            ada_kimg=args.ada_kimg,
            log_interval=args.log_interval,
            device=args.device,
            pretrained_g=args.pretrained_g,
            pretrained_d=args.pretrained_d,
            freeze_upto=args.freeze_upto
        )
    elif args.mode == 'generate':
        assert args.checkpoint is not None, "--checkpoint is required for generation"
        generate(
            checkpoint=args.checkpoint,
            num=args.num,
            latent_dim=args.latent_dim,
            seed=args.seed,
            output_dir=args.output_dir,
            device=args.device
        )