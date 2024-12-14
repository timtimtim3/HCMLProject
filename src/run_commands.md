# Training

## MNIST

### Run small cnn with MLP (128, 64) on MNIST with 10% label noise
python src/train.py --dataset "mnist" --model "cnn" --label_noise 0.1 --hidden_sizes 128 64

### Run small MLP (128, 64) on MNIST with 10% label noise
python src/train.py --dataset "mnist" --model "mlp" --label_noise 0.1 --hidden_sizes 128 64

## ISIC2024

### Run small resnet18 (frozen) with MLP (128, 64) on isic2024 with 10% label noise
python src/train.py --dataset "isic2024" --model "resnet18" --label_noise 0.1 --hidden_sizes 128 64


# Get Influences

## MNIST
python src/get_influences.py --dataset "mnist" --model "cnn" --label_noise 0.1 --hidden_sizes 128 64 --epochs 1 2 3 4 5 6 7 8 9 10

## ISIC2024


# Plotting

python src/plot_mislabeled_identified.py --dataset "mnist" --model "cnn" --label_noises 0.1 --hidden_sizes 128 64 --epochs 1 2 3 4 5 6 7 8 9 10

