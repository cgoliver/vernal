import sys
import argparse
import os
import pickle

FUNCTIONS = ['predict', 'train', 'rm', 'setup']

if __name__ != '__main__':
    raise ImportError('Cannot import the main')

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

try:
    function = sys.argv[1]
except IndexError:
    # raise ValueError("Specify a valid function please")
    function = 'train'

assert function in FUNCTIONS

if function == 'setup':
    from tools.learning_utils import setup

    setup()

if function == 'rm':
    from tools.learning_utils import remove

    exp = sys.argv[2]
    remove(exp)
    print(f"removed {exp}")

if function == 'train':
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("-ini", "--ini", default=None, help="name of the additional .ini to use")
    parser.add_argument("-da", "--annotated_data", default='rna_graphs_nr')
    parser.add_argument("-bs", "--batch_size", type=int, default=2, help="choose the batch size")
    parser.add_argument("-nw", "--workers", type=int, default=0, help="Number of workers to load data")
    parser.add_argument("-wt", "--wall_time", type=int, default=None, help="Max time to run the model")
    parser.add_argument("-n", "--name", type=str, default='default_name', help="Name for the logs")
    parser.add_argument("-t", "--timed", help="to use timed learning", action='store_true')
    parser.add_argument("-ep", "--num_epochs", type=int, help="number of epochs to train", default=30)
    parser.add_argument("-dev", "--device", default=0, type=int, help="gpu device to use")

    # Kernel function arguments
    parser.add_argument('-sf', '--sim_function', type=str,
                        help='Node similarity function (Supported Options: R_1, R_IDF, R_iso, hungarian).',
                        default="R_iso")
    parser.add_argument("-kd", "--kernel_depth", type=int, help="Number of hops to use in kernel.", default=3)
    parser.add_argument("--decay", type=float, help="decay for the kernel", default=0.8)
    parser.add_argument("--idf", default=False, action='store_true', help="To use or not idf"),
    parser.add_argument('-norm', '--normalization', type=str,
                        help='Normalization function (Supported Options None, sqrt, log)',
                        default='sqrt')

    # Reconstruction arguments
    parser.add_argument('--optim', type=str,
                        help='Supported Options: sgd, adam',
                        default="adam")
    parser.add_argument('-lr', '--lr', type=float,
                        default=0.002)
    parser.add_argument("-sim", "--similarity", default=True,
                        help="If we want to train reconstruction on distance instead of cosine",
                        action='store_false'),
    parser.add_argument("-sl", "--self_loop", default=False,
                        help="Add a self loop to graphs for convolution. Default: False",
                        action='store_true'),
    parser.add_argument('-ed', '--embedding_dims', nargs='+', type=int, help='Dimensions for embeddings.',
                        default=[32, 64])
    parser.add_argument("--weight", help="Whether to weight the K-matrix for NC", action='store_true')
    parser.add_argument("--normalize", help="Whether to use cosine instead of dot product", action='store_true')
    parser.add_argument('-co', '--conv_output',
                        default=True,
                        help='Apply graph conv to last later. Default: True',
                        action='store_false')
    args, _ = parser.parse_known_args()

    print(f"OPTIONS USED \n ",
          '-' * 10 + '\n',
          '\n'.join(map(str, vars(args).items()))
          )

    # Torch imports
    import torch
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter

    import os

    # Homemade modules
    from train_embeddings.loader import Loader, loader_from_hparams
    from train_embeddings.model import Model, model_from_hparams
    from train_embeddings.learn import train_model
    from tools.node_sim import SimFunctionNode, simfunc_from_hparams
    from tools.learning_utils import mkdirs_learning, ConfParser

    # Create .exp and dump it
    if args.ini is not None:
        args.ini = os.path.join(script_dir, 'inis',
                                f'{args.ini}.ini')
    hparams = ConfParser(default_path=os.path.join(script_dir, 'inis/default.ini'),
                         path_to_ini=args.ini,
                         argparse=args)

    # Hardware settings
    # torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # Dataloader creation
    annotated_path = os.path.join(script_dir, '../data/annotated', args.annotated_data)
    loader = loader_from_hparams(annotated_path=annotated_path, hparams=hparams)
    hparams.add_value('argparse', 'num_edge_types', loader.num_edge_types)
    train_loader, test_loader, all_loader = loader.get_data()

    if len(train_loader) == 0 & len(test_loader) == 0:
        raise ValueError('there are not enough points compared to the BS')

    # Model and optimizer setup
    model = model_from_hparams(hparams=hparams)
    model = model.to(device)
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Experiment Setup
    name = args.name
    result_folder, save_path = mkdirs_learning(name)

    writer = SummaryWriter(result_folder)
    print(f'Saving result in {name}')

    # write model metadata
    hparams.dump(dump_path=os.path.join(script_dir, '../results/trained_models', args.name, f'{args.name}.exp'))
    pickle.dump({
        'dims': args.embedding_dims,
        'edge_map': train_loader.dataset.dataset.edge_map,
        'depth': args.kernel_depth,
        'sim_function': args.sim_function
    },
        open(os.path.join(os.path.dirname(save_path), 'meta.p'), 'wb'))
    # Get Summary of the model
    # from torchsummary import summary

    # Run
    train_model(model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                test_loader=test_loader,
                save_path=save_path,
                writer=writer,
                num_epochs=args.num_epochs,
                wall_time=args.wall_time)
