# Run small resnet18 (frozen) with MLP (128, 64) behind on isic2024 with 10% label noise
python train.py --dataset "isic2024" --model "resnet18" --label_noise 0.1 --hidden_sizes 128 64
