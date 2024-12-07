import argparse

from torch import get_device

from datasets import AVAILABLE_DATASETS
from models import AVAILABLE_MODELS
from utils.functions import get_checkpoint_dir_from_args, set_seed
from utils.logger import setup_logger
from utils.parser import add_shared_parser_arguments


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate the influence scores based on a already trained model")
    add_shared_parser_arguments(parser)
    args = parser.parse_args()

    set_seed(42)
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

    ModelClass = AVAILABLE_MODELS[args.model]
    DatasetClass = AVAILABLE_DATASETS[args.dataset]


    # TO DO
    # - loop over the modelcheckpoints in checkpoint_dir
    # - populate the ModelClass with the checkpoint data
    # - load the dataset
    # - do inference
    # - calculate scores
    # - save scores to checkpoint dir e.g. scores_epoch_1.[]


    # For reference, the data is saved in the following way:

    # torch.save({
    #     'epoch': epoch + 1,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'learning_rate': args.lr,  # Save the current learning rate
    #     'train_loss': training_loss,
    #     'val_loss': validation_loss,
    #     'val_accuracy': accuracy,
    # }, checkpoint_path)
