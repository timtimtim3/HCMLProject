import argparse

from torch import get_device

from datasets import AVAILABLE_DATASETS
from models import AVAILABLE_MODELS
from utils.functions import get_checkpoint_dir_from_args, get_output_dir_from_args, set_seed
from utils.logger import setup_logger
from utils.parser import add_shared_parser_arguments


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate the influence scores based on a already trained model")
    add_shared_parser_arguments(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    logger = setup_logger() 
    device = get_device() 

    # Log parameters
    logger.info(f'Using device: {device}')
    logger.info('Using seed 42')
    logger.info('Hyperparameters:')
    logger.info(f'Hidden sizes: {args.hidden_sizes}')
    logger.info(f'Learning rate: {args.lr}')
    logger.info(f'Batch size: {args.batch_size}')
    logger.info(f'Number of epochs: {args.num_epochs}')

    checkpoint_dir = get_checkpoint_dir_from_args(args) 
    output_dir = get_output_dir_from_args(args)

    ModelClass = AVAILABLE_MODELS[args.model]
    DatasetClass = AVAILABLE_DATASETS[args.dataset]


    # python src/calculate_baseline.py --model cnn --dataset mnist

    # get best model from checkpoint dir (epoch number)

    # do inference

    # get logits from model
    # parse through softmax

    # save in output_dir/baseline.[] 
