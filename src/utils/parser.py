
from datasets import AVAILABLE_DATASETS
from models import AVAILABLE_MODELS


def add_shared_parser_arguments(parser):
    """
    Add a list of arguments to a parser which are common among files. These arguments are for 
    example used to define a checkpoint folder name in train.py and load these checkpoints in get_influences.py.
    """

    parser.add_argument("--dataset", type=str, choices=list(AVAILABLE_DATASETS.keys()), required=True)
    parser.add_argument("--model", type=str, choices=list(AVAILABLE_MODELS.keys()), required=True)
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[50, 50], help='List of hidden layer sizes')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--label_noise', type=float, default=0.0, help="Percentage of added noise to dataset")
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')

