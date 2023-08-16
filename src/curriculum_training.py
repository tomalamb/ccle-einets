"""
Training script for curriculum training of EiNets models for density estimation. Code
builds upon the code from the original EiNets repository, which can be found
at https://github.com/cambridge-mlg/EinsumNetworks.
"""

import torch
from EinsumNetwork import Graph, EinsumNetwork
import argparse
import csv
import numpy as np
from datasets import Dataset, load_data
import os
import wandb
import sys

def parse_args() -> argparse.Namespace:
    """Function that gathers the the arguments used for training
    a baseline EiNet model.

    Returns:
        argparse.Namespace: Argparser object which stores training arguments for the EiNet model to be trained.
    """
    parser = argparse.ArgumentParser(
        description='EiNet argument parser.')
    parser.add_argument('-i', type=str, default='../data/datasets', metavar='D',
                        help='directory where the input data is stored (default: ../data/input/datasets)')
    parser.add_argument('-o', type=str, default='../data/output/curriculum_training', metavar='D',
                        help='directory where the outputs of training are to be stored (default: ../data/output/curriculum_training)')
    parser.add_argument('-K', type=int, default=10, metavar='D',
                        help='number of vectorised distributions in sum nodes and leaf inputs (default: 10)')
    parser.add_argument('--max_num_epochs', type=int, default=64, metavar='D',
                        help='maximum number of training epochs (default: 64)')
    parser.add_argument('--setting_epochs', type=int, default=15, metavar='D',
                        help='number of epochs to train each setting for (default: 15)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='D',
                        help='batch size (default: 32)')
    parser.add_argument('--patience', type=int, default=8, metavar='D',
                        help='patience for early stopping (default: 8)') 
    parser.add_argument('--pd_deltas', type=str, default='7,14', metavar='D',
                        help='Poon-Domingos structre delta step sizes for image splits (default: 1)')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
                        help='dataset to train on, (default: mnist, options are mnist, f_mnist, celeba, svhn).')
    parser.add_argument('--optimiser', type=str, default='adam', metavar='D',
                        help='optimiser to use for cll optimisation (default: adam,, options are adam, sgd).')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='D',
                        help='initial learning rate for optimiser (default: 1e-3)')
    parser.add_argument('--patch_size', type=str, default='4,4', metavar='D',
                        help='Window width and height (in this order) for sampled marginal during ccle training.')
    parser.add_argument('--wandb_online', action='store_true',
                        help='Whether to use wandb for logging. If not provided, the default is False.')
    parser.add_argument('--wandb_project', type=str, default='EiNets - Curriculum Training', metavar='D',
                        help='Wandb project name (default: EiNets).')
    parser.add_argument('--run', type=int, default=0, metavar='D',
                        help='Which ccle run to fine tune from. If not provided, the default is the first run.')
    parser.add_argument('--runs', type=int, default=1, metavar='D',
                        help='Number of runs to perform. If not provided, the default is 1.')
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    """Main function used to train EiNet according to the given arguments. Multiple runs are performed for
    significance testing. These results are then saved to a file.
    Args:
        args (argparse.Namespace): Baseline model training arguments.
    """
    print("Staring Curriculum training script...")

    # Parse training arguments to create baseline model.
    input_dir = args.i
    output_dir = args.o
    max_num_epochs = args.max_num_epochs
    batch_size = args.batch_size
    patience = args.patience
    pd_deltas = [int(delta) for delta in args.pd_deltas.split(',')]
    dataset = args.dataset
    optimiser_chosen = args.optimiser
    lr = args.lr
    wandb_project = args.wandb_project
    wandb_online = args.wandb_online
    setting_epochs = args.setting_epochs
    k = args.K

    # Check window dimensions are for ccle training are valid
    # and generate identifier for run.
    curriculum_settings = [
        {"patch_size": 1, "gamma": 0.5},
        {"patch_size": 4, "gamma": 0.6272},
        {"patch_size": 8, "gamma": 0.8889}
    ]

    patch_sizes = '_'.join(str(setting["patch_size"]) for setting in curriculum_settings)
    gammas = '_'.join(str(setting["gamma"]) for setting in curriculum_settings)

    identifier = f'k={k}_dataset={dataset}_patch_sizes_{patch_sizes}_gammas_{gammas}'

    config = vars(args)
    config['identifier'] = identifier

    print("Loading data...")
    # Set up dataset and associated parameters.
    if dataset == 'mnist':
        dataset = Dataset.MNIST
        img_dims = {'height': 28, 'width': 28}
        num_pixel_vars = img_dims['height'] * img_dims['width']
        valid_size = 10000
        input_dir = input_dir + '/mnist/'
        output_dir = output_dir + '/cll_training/mnist/curriculum_training/decay/'
        output_dir += 'patch_sizes_' + '_'.join(str(setting['patch_size']) for setting in curriculum_settings) + '_'
        output_dir += 'gammas_' + '_'.join(str(setting['gamma']) for setting in curriculum_settings)
        output_dir += '/'
    elif dataset == 'f-mnist':
        dataset = Dataset.F_MNIST
        img_dims = {'height': 28, 'width': 28}
        num_pixel_vars = img_dims['height'] * img_dims['width']
        valid_size = 10000
        input_dir = input_dir + '/f_mnist/'
        output_dir = output_dir + '/cll_training/f_mnist/curriculum_training/'
        output_dir += 'patch_sizes_' + '_'.join(str(setting['patch_size']) for setting in curriculum_settings) + '_'
        output_dir += 'gammas_' + '_'.join(str(setting['gamma']) for setting in curriculum_settings)
        output_dir += '/'
    elif dataset == 'CELEBA':
        dataset = Dataset.CELEBA
        img_dims = {'height': 32, 'width': 32}
        num_pixel_vars = img_dims['height'] * img_dims['width']
        valid_size = 10000
        input_dir = input_dir + '/celeba/'
        output_dir = output_dir + '/cll_training/celeba/curriculum_training/'
        output_dir += 'patch_sizes_' + '_'.join(str(setting['patch_size']) for setting in curriculum_settings) + '_'
        output_dir += 'gammas_' + '_'.join(str(setting['gamma']) for setting in curriculum_settings)
        output_dir += '/'

    elif dataset == 'SVHN':
        dataset = Dataset.SVHN
        img_dims = {'height': 32, 'width': 32}
        num_pixel_vars = img_dims['height'] * img_dims['width']
        valid_size = 10000
        input_dir = input_dir + '/svhn/'
        output_dir = output_dir + '/cll_training/svhn/curriculum_training/'
        output_dir += 'patch_sizes_' + '_'.join(str(setting['patch_size']) for setting in curriculum_settings) + '_'
        output_dir += 'gammas_' + '_'.join(str(setting['gamma']) for setting in curriculum_settings)
        output_dir += '/'


    # Set up wandb logging.
    # Create unique identifier thats the same for each run with the same hyperparameters.
    if wandb_online:
        wandb.init(project=wandb_project,
                name=f"Curriculum training: {identifier}", config=config)
    else:
        wandb.init(project=wandb_project,
                name=f"Curriculum training: {identifier}", config=config, mode='dryrun')
        
    # Load dataset.
    train_x, _, test_x, _ = load_data(dataset, input_dir)

    # Use last valid_size (int) images from training set as validation set.
    valid_x = train_x[-valid_size:, :]
    train_x = train_x[:-valid_size, :]

    # Place data on device. Default to GPU when available.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_x = torch.from_numpy(train_x).to(device)
    valid_x = torch.from_numpy(valid_x).to(device)
    test_x = torch.from_numpy(test_x).to(device)
    print("Using device: {}".format(device))

    # Get sizes of the train, valid, and test sets.
    train_N = train_x.shape[0]

    # Create EiNet DAG using PD structure.
    pd_delta = [[img_dims["height"] / delta, img_dims["width"] / delta]
                for delta in pd_deltas]
    graph = Graph.poon_domingos_structure(
        shape=(img_dims["height"], img_dims["width"]), delta=pd_delta)

    args = EinsumNetwork.Args(
        num_var=num_pixel_vars,
        num_dims=1,
        num_classes=1,
        num_sums=k,
        use_em=False,
        num_input_distributions=k,
        exponential_family=EinsumNetwork.CategoricalArray,
        exponential_family_args={'K': 256},
        img_dims=img_dims,
        device=device)

    einet = EinsumNetwork.EinsumNetwork(graph, args)
    einet.initialize()
    einet.to(device)

    # Print the number of trainable parameters in the model.
    num_parameters = sum(p.numel()
                            for p in einet.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_parameters}")

    # Choose optimiser.
    if optimiser_chosen == 'adam':
        optimiser = torch.optim.Adam(einet.parameters(), lr=lr)
    elif optimiser_chosen == 'sgd':
        optimiser = torch.optim.SGD(
            einet.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError('Invalid optimiser.')
    
    # Create directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialise the logging csv file.
    csv_path = output_dir + '/results.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['patch_size', 'gamma', 'train_bpd', 'valid_bpd', 'test_bpd'])

    # Loop through each curriculum setting which gradually increases the patch size finally ending with full MLE training.   
    for idx, setting in enumerate(curriculum_settings):
        # Clear GPU cache betweem runs.
        torch.cuda.empty_cache()

        # If idx > 0, half the learning rate.
        if idx > 0:
            for param_group in optimiser.param_groups:
                param_group['lr'] = param_group['lr'] / 2
            

        # Set up window dimensions for marginalisation.
        marginal_window_dims = {
            "height": setting["patch_size"], "width": setting["patch_size"]}
        print(f"Training using conditional window dimensions of {marginal_window_dims}")
        
        # Train model using early stopping based on avg validation log-likelihood (bpd).
        best_epoch_train_ll_bpd = float('inf')
        best_valid_ll_bpd = float('inf')
        best_epoch_test_ll_bpd = float('inf')

        # MLE training loop using gradient descent.
        for epoch in range(setting_epochs):
            # Training loop using gradient descent on MLE objective.
            einet.train()
            idx_batches = torch.randperm(train_N).split(batch_size)
            for _, idx in enumerate(idx_batches):
                batch_x = train_x[idx, :]
                optimiser.zero_grad()

                cll, _, _ = einet.conditional_composite_ll(
                    batch_x, img_dims["width"], 
                    img_dims["height"], 
                    marginal_window_dims["width"],
                    marginal_window_dims["height"], 
                    patch_prob = 1.0, 
                    grid_sampling=True, 
                    grid_prob = setting["gamma"])
                
                objective = -cll/batch_size
                objective.backward()
                optimiser.step()
                wandb.log({'Avg loss': objective.item()})

            # Evaluate log-likehood of model on training and validation sets.
            with torch.no_grad():
                einet.eval()
                train_ll_bpd = einet.avg_neg_ll_bpd(train_x)
                valid_ll_bpd = einet.avg_neg_ll_bpd(valid_x)

                # Check for model improvements. If so, save model parameters.
                if valid_ll_bpd < best_valid_ll_bpd:
                    best_valid_epoch = 0
                    best_valid_ll_bpd = valid_ll_bpd
                    best_valid_epoch = epoch
                    best_epoch_train_ll_bpd = train_ll_bpd

                    # Save model paramters.
                    if not os.path.exists(output_dir + 'models/'):
                        try:
                            os.makedirs(output_dir + 'models/')
                        except OSError as e:
                            print(e)
                            # Stop run if model directory cannot be created.
                            sys.exit(1)

                    torch.save(einet.state_dict(), output_dir +
                                f'models/einet_model_{setting["patch_size"]}_{setting["gamma"]}.pt')

                # Save training and validation log-likelihoods for each epoch.
                wandb.log({"train_ll (bpd)": train_ll_bpd,
                            "valid_ll (bpd)": valid_ll_bpd})
                print(
                    f'Epoch {epoch}, train ll (bpd): {train_ll_bpd}, valid ll (bpd): {valid_ll_bpd}')

                # Use early stopping if validation log-likelihood has not improved over
                # the last few (defined by patience variable) epochs.
                if epoch - best_valid_epoch >= patience:
                    break

        # Store results for dataset at best validation epoch, including the test log-likelihood.
        einet.load_state_dict(torch.load(
            output_dir + f'models/einet_model_{setting["patch_size"]}_{setting["gamma"]}.pt'))
        best_epoch_test_ll_bpd = einet.avg_neg_ll_bpd(test_x)

        # Print test log-likelihood.
        print(f"Test log-likelihood (bpd): {best_epoch_test_ll_bpd}")
        print("________________________________________________________")

        wandb.log({"test_ll (bpd)": best_epoch_test_ll_bpd})

        # Store the results for this setting in CSV file
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([setting["patch_size"], setting["gamma"],
                            best_epoch_train_ll_bpd, best_valid_ll_bpd, best_epoch_test_ll_bpd])
            
    # Finally, finsh by training using MLE.
    # Clear GPU cache betweem runs.
    torch.cuda.empty_cache()
    
    # Train model using early stopping based on avg validation log-likelihood (bpd).
    best_epoch_train_ll_bpd = float('inf')
    best_valid_ll_bpd = float('inf')
    best_epoch_test_ll_bpd = float('inf')

    # MLE training loop using gradient descent.
    for epoch in range(max_num_epochs):
        # Training loop using gradient descent on MLE objective.
        einet.train()
        idx_batches = torch.randperm(train_N).split(batch_size)
        for _, idx in enumerate(idx_batches):
            batch_x = train_x[idx, :]
            optimiser.zero_grad()
            ll = einet.forward(batch_x).sum()
            objective = -ll/batch_size
            objective.backward()
            optimiser.step()
            wandb.log({'Avg loss': objective.item()})
            
        # Evaluate log-likehood of model on training and validation sets.
        with torch.no_grad():
            einet.eval()
            train_ll_bpd = einet.avg_neg_ll_bpd(train_x)
            valid_ll_bpd = einet.avg_neg_ll_bpd(valid_x)

            # Check for model improvements. If so, save model parameters.
            if valid_ll_bpd < best_valid_ll_bpd:
                best_valid_epoch = 0
                best_valid_ll_bpd = valid_ll_bpd
                best_valid_epoch = epoch
                best_epoch_train_ll_bpd = train_ll_bpd

                # Save model paramters.
                if not os.path.exists(output_dir + 'models/'):
                    try:
                        os.makedirs(output_dir + 'models/')
                    except OSError as e:
                        print(e)
                        # Stop run if model directory cannot be created.
                        sys.exit(1)

                torch.save(einet.state_dict(), output_dir +
                            f'models/einet_model_mle.pt')

            # Save training and validation log-likelihoods for each epoch.
            wandb.log({"train_ll (bpd)": train_ll_bpd,
                        "valid_ll (bpd)": valid_ll_bpd})
            print(
                f'Epoch {epoch}, train ll (bpd): {train_ll_bpd}, valid ll (bpd): {valid_ll_bpd}')

            # Use early stopping if validation log-likelihood has not improved over
            # the last few (defined by patience variable) epochs.
            if epoch - best_valid_epoch >= patience:
                break

    # Store results for dataset at best validation epoch, including the test log-likelihood.
    einet.load_state_dict(torch.load(
        output_dir + f'models/einet_model_mle.pt'))
    best_epoch_test_ll_bpd = einet.avg_neg_ll_bpd(test_x)

    # Print test log-likelihood.
    print(f"Test log-likelihood (bpd): {best_epoch_test_ll_bpd}")
    print("________________________________________________________")

    wandb.log({"test_ll (bpd)": best_epoch_test_ll_bpd})

    # Store the results for this setting in CSV file
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['mle', 'mle',
                        best_epoch_train_ll_bpd, best_valid_ll_bpd, best_epoch_test_ll_bpd])

    # Finish wandb run.
    wandb.finish()

if __name__ == '__main__':
    # Parse args for model.
    args = parse_args()
    # Train baseline models according to arg specifications.
    main(args)