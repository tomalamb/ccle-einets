"""
Training script for training EiNets models for density estimation. Code
builds upon the code from the original EiNets repository, which can be found
at https://github.com/cambridge-mlg/EinsumNetworks.
"""

import torch
from EinsumNetwork import Graph, EinsumNetwork
import argparse
import csv
from datasets import Dataset, load_data
import os
import wandb
import math
import sys


def parse_args() -> argparse.Namespace:
    """Function that gathers the the arguments used for training EiNet models to be trained via
    mle using EM or SGD or for EiNet models trained via ccle.

    Returns:
        argparse.Namespace: Argparser object which stores training arguments for the EiNet model to be trained.
    """
    parser = argparse.ArgumentParser(
        description='EiNet argument parser.')
    parser.add_argument('-i', type=str, default='../data/datasets/', metavar='D',
                        help='directory where the input data is stored (default: ../data/input)')
    parser.add_argument('-o', type=str, default='../data/output/baselines', metavar='D',
                        help='directory where the outputs of training are to be stored (default: ../data/output)')
    parser.add_argument('-K', type=int, default=10, metavar='D',
                        help='number of vectorised distributions in sum nodes and leaf inputs (default: 10)')
    parser.add_argument('--max_num_epochs', type=int, default=64, metavar='D',
                        help='maximum number of training epochs (default: 64)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='D',
                        help='batch size (default: 64)')
    parser.add_argument('--online_em_frequency', type=int, default=1, metavar='D',
                        help='online EM frequency (default: 1)')
    parser.add_argument('--online_em_stepsize', type=float, default=0.05, metavar='D',
                        help='online EM stepsize (default: 0.05)')
    parser.add_argument('--num_runs', type=int, default=1, metavar='D',
                        help='number of runs (default: 1)')
    parser.add_argument('--patience', type=int, default=5, metavar='D',
                        help='patience for early stopping (default: 5)')
    parser.add_argument('--pd_deltas', type=str, default='7,14', metavar='D',
                        help='Poon-Domingos structre delta step sizes for image splits (default: 1)')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
                        help='dataset to train on, (default: mnist, options are mnist, f_mnist, celeba, svhn).')
    parser.add_argument('--optimiser', type=str, default='adam', metavar='D',
                        help='optimiser to use for SGD training (default: adam,, options are adam, sgd).')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='D',
                        help='initial learning rate for optimiser (default: 1e-3)')
    parser.add_argument('--patch_prob', type=float, default=1.0, metavar='D',
                        help='Probability of using ccle for patch of image over the likelihood of the full image. (default: 1.0)')
    parser.add_argument('--patch_size', type=str, default='4,4', metavar='D',
                        help='Window width and height (in this order) for sampled marginal during ccle training.')
    parser.add_argument('--ccle', action='store_true',
                        help='Whether to use ccle training or not. If not provided, the default is False.')
    parser.add_argument('--use_em', action='store_true',
                        help='Determines whether to trainn baseline using SGD or s-EM. If not provided, the default is False.')
    parser.add_argument('--wandb_online', action='store_true',
                        help='Whether to use wandb for logging. If not provided, the default is False.')
    parser.add_argument('--wandb_project', type=str, default='EiNets', metavar='D',
                        help='Wandb project name (default: EiNets).')
    parser.add_argument('--split_patching', action='store_true',
                        help='Whether to use split patching or not. If not provided, the default is False (i.e. no patching).')
    parser.add_argument('--grid_patch', action='store_true',
                        help='Whether to use grid patching or not. If not provided, the default is False')
    parser.add_argument('--grid_prob', type=float, default=0.1, metavar='D',
                        help='Probability of choosing patch within grid region. (default: 1.0)')
    parser.add_argument('--bisection_sampling', action='store_true',
                        help='Whether to use binary cut sampling or not. If not provided, the default is False')
    parser.add_argument('--num_bin_bisections', type=int, default=1, metavar='D',
                        help='Number of binary cut bisections. (default: 1)')
    parser.add_argument('--eval_freq', type=int, default=1, metavar='D',
                        help='Frequency of evaluation of ll on training and validation sets during training. (default: 1)')

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    """Main function used to train EiNet according to the given arguments. Multiple runs can be performed for
    significance testing. These results are then saved to a file.
    Args:
        args (argparse.Namespace): Baseline model training arguments.
    """

    # Parse training arguments to create baseline model.
    input_dir = args.i
    output_dir = args.o
    max_num_epochs = args.max_num_epochs
    batch_size = args.batch_size
    online_em_frequency = args.online_em_frequency
    online_em_stepsize = args.online_em_stepsize
    num_runs = args.num_runs
    patience = args.patience
    pd_deltas = [int(delta) for delta in args.pd_deltas.split(',')]
    dataset_name = args.dataset
    optimiser_chosen = args.optimiser
    patch_size = [int(dim) for dim in args.patch_size.split(',')]
    lr = args.lr
    patch_prob = args.patch_prob
    ccle = args.ccle
    use_em = args.use_em
    wandb_project = args.wandb_project
    wandb_online = args.wandb_online
    split_patching = args.split_patching
    grid_patch = args.grid_patch
    grid_prob = args.grid_prob
    bisection_sampling = args.bisection_sampling
    num_bin_bisections = args.num_bin_bisections
    eval_freq = args.eval_freq

    # Check not using EM and ccle together.
    if use_em and ccle:
        raise ValueError(
            'Cannot use em and ccle together. Choose one or the other.')

    # Main hyperparameters for the baseline model.
    k = args.K

    if ccle:
        # If using ccle, check window dimensions are for ccle training are valid.
        # and generate identifier for run.
        if len(patch_size) > 2 or len(patch_size) < 1:
            raise ValueError(
                'Window dimensions must be a list of length of 1 or 2.')

        # If the patch size is less than 2, assume square window and change patch size to be list with two elements.
        if len(patch_size) < 2:
            patch_size.append(patch_size[0])

        # Create patch window dimensions dictionary.
        patch_dims = {
            "width": patch_size[0], "height": patch_size[1]}

        # Create unique identifier thats the same for each run with the same hyperparameters.
        identifier = f'k={k}_mcle={ccle}_dataset={dataset_name}_optimiser={optimiser_chosen}_lr={lr}_patch_prob={patch_prob}_patch_size={patch_dims["width"]},{patch_dims["height"]}'
    else:
        # If not using ccle, generate identifier for run based on whether using SGD or EM.
        if use_em:
            identifier = f'k={k}_ccle={ccle}_dataset={dataset_name}_em_stepsize_{online_em_stepsize}_use_em={use_em}'
        else:
            identifier = f'k={k}_ccle={ccle}_dataset={dataset_name}_optimiser={optimiser_chosen}_lr={lr}_use_em={use_em}'

    print("Loading data...")
    # Set up dataset and associated parameters.
    if dataset_name == 'mnist':
        dataset = Dataset.MNIST
        img_dims = {'height': 28, 'width': 28}
        wandb_project = wandb_project + '_mnist'
        valid_size = 10000
        input_dir = input_dir + '/mnist/'
        if not ccle:
            if use_em:
                output_dir = output_dir + f'/baseline_training/mnist/use/'
            else:
                output_dir = output_dir + f'/baseline_training/mnist/sgd/'
    elif dataset_name == 'f_mnist':
        dataset = Dataset.F_MNIST
        img_dims = {'height': 28, 'width': 28}
        wandb_project = wandb_project + '_f_mnist'
        valid_size = 10000
        input_dir = input_dir + '/f_mnist/'
        if not ccle:
            if use_em:
                output_dir = output_dir + f'/baseline_training/f_mnist/em/'
            else:
                output_dir = output_dir + f'/baseline_training/f_mnist/sgd/'
    else:
        raise ValueError('Dataset not recognised.')

    # Depending on the type of training, setup wandb logging and output directory.
    if split_patching:
        print('Split patching')
        identifier = identifier + f'_split_patching_{split_patching}'
        patch_dims["width"] = math.floor(
            patch_dims["width"]/2)
        patch_dims["height"] = math.floor(
            patch_dims["height"]/2)
        print("Now have window dimensions:",
              patch_dims["width"], patch_dims["height"])
        wandb_project = wandb_project + '_split_patching'
        output_dir = output_dir + \
            f'/ccle_training/{dataset_name}/patch_size_{str(patch_dims["width"])}_{str(patch_dims["height"])}/patch_prob_{str(patch_prob)}/split_patching/'
    elif grid_patch:
        print(
            f"Using grid patching with grid patch size: {grid_patch} with grid patch probability: {grid_prob}")
        identifier = identifier + f'_grid_patch_{grid_patch}/prob_{grid_prob}'
        wandb_project = wandb_project + '_grid_patch'
        output_dir = output_dir + \
            f'/patch_size_{str(patch_dims["width"])}_{str(patch_dims["height"])}/patch_prob_{str(patch_prob)}/grid_patch/grid_prob_{grid_prob}/'
    elif bisection_sampling:
        print(
            f"Using bisection_sampling sampling with {num_bin_bisections} binary bisection cuts")
        identifier = identifier + f'_bisection_sampling_{num_bin_bisections}'
        wandb_project = wandb_project + '_bisection_sampling'
        output_dir = output_dir + \
            f'/bisection_sampling_full/num_bin_bisections_{num_bin_bisections}/'
    else:
        print("Using uniform random patching")
        output_dir = output_dir + \
            f'/ccle_training/{dataset_name}/patch_size_{str(patch_dims["width"])}_{str(patch_dims["height"])}/patch_prob_{str(patch_prob)}/'

    # Keep track of lls for each epoch of each run.
    run_lls = []
    results = []

    print("Starting training...")
    # Loop over runs for significance testing if required.
    for run in range(num_runs):
        # Clear GPU cache betweem runs.
        torch.cuda.empty_cache()

        # Set up wandb logging.
        config = vars(args)
        config['run'] = run
        config['identifier'] = identifier
        if wandb_online:
            if ccle:
                wandb.init(project=wandb_project,
                           name=f"ccle run {identifier} {run}", config=vars(args))
            else:
                wandb.init(project=wandb_project,
                           name=f"Baseline run {identifier} {run}", config=vars(args))
        else:
            if ccle:
                wandb.init(project=wandb_project,
                           name=f"ccle run {identifier} {run}", config=vars(args), mode='dryrun')
            else:
                wandb.init(project=wandb_project,
                           name=f"Baseline run {identifier} {run}", config=vars(args), mode='dryrun')

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

        # Get sizes of the training set.
        train_N = train_x.shape[0]

        # Create EiNet DAG using PD structure.
        pd_delta = [[img_dims["height"] / delta, img_dims["width"] / delta]
                    for delta in pd_deltas]
        print("Using PD deltas:", pd_delta)
        graph = Graph.poon_domingos_structure(
            shape=(img_dims["height"], img_dims["width"]), delta=pd_delta)

        # Create EiNet using categorical variables and the passed arguments.
        if ccle:
            args = EinsumNetwork.Args(
                num_var=train_x.shape[1],
                num_dims=1,
                num_classes=1,
                num_sums=k,
                use_em=False,
                num_input_distributions=k,
                exponential_family=EinsumNetwork.CategoricalArray,
                exponential_family_args={'K': 256},
                img_dims=img_dims,
                device=device)
        else:
            args = EinsumNetwork.Args(
                num_var=train_x.shape[1],
                num_dims=1,
                num_classes=1,
                num_sums=k,
                use_em=use_em,
                num_input_distributions=k,
                exponential_family=EinsumNetwork.CategoricalArray,
                exponential_family_args={'K': 256},
                online_em_frequency=online_em_frequency,
                online_em_stepsize=online_em_stepsize,
                img_dims=img_dims,
                device=device)

        einet = EinsumNetwork.EinsumNetwork(graph, args)
        einet.initialize()
        einet.to(device)

        # Print the number of trainable parameters in the model.
        num_parameters = sum(p.numel()
                             for p in einet.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_parameters}")

        # If not using EM, setup optimiser for training.
        if not use_em:
            optimiser = torch.optim.Adam(einet.parameters(), lr=lr)

        # Train model using early stopping based on avg validation nll (bpd).
        best_epoch_train_nll_bpd = float('inf')
        best_valid_nll_bpd = float('inf')
        best_epoch_test_nll_bpd = float('inf')
        best_valid_epoch = 0

        if not use_em:
            # ccle training loop.
            if ccle:
                for epoch in range(max_num_epochs):
                    # Training loop using gradient descent on ccle objective.
                    idx_batches = torch.randperm(train_N).split(batch_size)
                    einet.train()
                    for _, idx in enumerate(idx_batches):
                        batch_x = train_x[idx, :]
                        optimiser.zero_grad()
                        ccll, _, _ = einet.ccll(
                            batch_x, img_dims["width"],
                            img_dims["height"],
                            patch_dims["width"],
                            patch_dims["height"],
                            patch_prob,
                            grid_sampling=grid_patch,
                            grid_prob=grid_prob,
                            bisection_sampling=bisection_sampling,
                            num_bin_bisections=num_bin_bisections)

                        objective = 1/batch_size * -ccll
                        objective.backward()
                        optimiser.step()

                        wandb.log({'Avg nccll loss': objective.item()})

                    if epoch % eval_freq == 0:
                        # Evaluate log-likehood of model on full training and validation sets.
                        with torch.no_grad():
                            einet.eval()
                            train_nll_bpd = einet.avg_neg_ll_bpd(train_x)
                            valid_nll_bpd = einet.avg_neg_ll_bpd(valid_x)

                            # Check for model improvements. If so, save model parameters.
                            if valid_nll_bpd < best_valid_nll_bpd:
                                best_valid_nll_bpd = valid_nll_bpd
                                best_valid_epoch = epoch
                                best_epoch_train_nll_bpd = train_nll_bpd

                                # Save model paramters.
                                if not os.path.exists(output_dir + 'models/'):
                                    try:
                                        os.makedirs(output_dir + 'models/')
                                    except OSError as e:
                                        print(e)
                                        # Stop run if model directory cannot be created.
                                        sys.exit(1)

                                torch.save(einet.state_dict(), output_dir +
                                           f'models/einet_model_{run}.pt')

                            # Save training and validation nlls for each epoch.
                            run_lls.append(
                                (run, epoch, train_nll_bpd, valid_nll_bpd))
                            wandb.log({"train_nll (bpd)": train_nll_bpd,
                                       "valid_nll (bpd)": valid_nll_bpd})
                            print(
                                f'Epoch {epoch}, train nll (bpd): {train_nll_bpd}, valid nll (bpd): {valid_nll_bpd}')

                            # Use early stopping if validation nll has not improved over
                            # the last few (defined by patience variable) epochs.
                            if epoch - best_valid_epoch >= patience:
                                break
            else:
                # Baseline MLE training loop using gradient descent.
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
                        wandb.log({'Avg nll loss': objective.item()})

                    # Evaluate log-nll of model on training and validation sets.
                    if epoch % eval_freq == 0:
                        with torch.no_grad():
                            einet.eval()
                            train_nll_bpd = einet.avg_neg_ll_bpd(train_x)
                            valid_nll_bpd = einet.avg_neg_ll_bpd(valid_x)

                            # Check for model improvements. If so, save model parameters.
                            if valid_nll_bpd < best_valid_nll_bpd:
                                best_valid_nll_bpd = valid_nll_bpd
                                best_valid_epoch = epoch
                                best_epoch_train_nll_bpd = train_nll_bpd

                                # Save model paramters.
                                if not os.path.exists(output_dir + 'models/'):
                                    try:
                                        os.makedirs(output_dir + 'models/')
                                    except OSError as e:
                                        print(e)
                                        # Stop run if model directory cannot be created.
                                        sys.exit(1)

                                torch.save(einet.state_dict(), output_dir +
                                           f'models/einet_model_{run}.pt')

                            # Save training and validation nll for each epoch.
                            run_lls.append(
                                (run, epoch, train_nll_bpd, valid_nll_bpd))
                            wandb.log({"train_nll (bpd)": train_nll_bpd,
                                       "valid_nll (bpd)": valid_nll_bpd})
                            print(
                                f'Epoch {epoch}, train nll (bpd): {train_nll_bpd}, valid nll (bpd): {valid_nll_bpd}')

                            # Use early stopping if validation nll has not improved over
                            # the last few (defined by patience variable) epochs.
                            if epoch - best_valid_epoch >= patience:
                                break
        else:
            # EM training loop.
            for epoch in range(max_num_epochs):
                # Training loop using EM algorithm.
                idx_batches = torch.randperm(train_N).split(batch_size)
                einet.train()
                for _, idx in enumerate(idx_batches):
                    batch_x = train_x[idx, :]
                    outputs = einet.forward(batch_x)
                    log_likelihood = outputs.sum()
                    log_likelihood.backward()
                    wandb.log({'Avg NLL loss': -
                               log_likelihood.item()/batch_size})

                    einet.em_process_batch()

                einet.em_update()

                # Evaluate nll of model on training and validation sets.
                if epoch % eval_freq == 0:
                    with torch.no_grad():
                        einet.eval()
                        train_nll_bpd = einet.avg_neg_ll_bpd(train_x)
                        valid_nll_bpd = einet.avg_neg_ll_bpd(valid_x)

                        # Check for model improvements. If so, save model parameters.
                        if valid_nll_bpd < best_valid_nll_bpd:
                            best_valid_nll_bpd = valid_nll_bpd
                            best_valid_epoch = epoch
                            best_epoch_train_nll_bpd = train_nll_bpd

                            # Save model paramters.
                            if not os.path.exists(output_dir + 'models/'):
                                try:
                                    os.makedirs(output_dir + 'models/')
                                except OSError as e:
                                    print(e)
                                    # Stop run if model directory cannot be created.
                                    sys.exit(1)

                            torch.save(einet.state_dict(), output_dir +
                                       f'models/einet_model_{run}.pt')

                        # Save training and validation nll for each epoch.
                        run_lls.append(
                            (run, epoch, train_nll_bpd, valid_nll_bpd))
                        wandb.log({"train_nll (bpd)": train_nll_bpd,
                                   "valid_nll (bpd)": valid_nll_bpd})
                        print(
                            f'Epoch {epoch}, train nll (bpd): {train_nll_bpd}, valid nll (bpd): {valid_nll_bpd}')

                        # Use early stopping if validation NLL has not improved over
                        # the last few (defined by patience variable) epochs.
                        if epoch - best_valid_epoch >= patience:
                            break

        # Store results for dataset at best validation epoch, including the test nll.
        einet.load_state_dict(torch.load(
            output_dir + f'models/einet_model_{run}.pt'))
        best_epoch_test_nll_bpd = einet.avg_neg_ll_bpd(test_x)
        wandb.log({"test_nll (bpd)": best_epoch_test_nll_bpd})
        print(
            f'Best epoch train nll: {best_epoch_train_nll_bpd}, Best epoch valid nll: {best_valid_nll_bpd}, Best epoch test nll: {best_epoch_test_nll_bpd}')
        results.append((run, best_epoch_train_nll_bpd, best_valid_nll_bpd,
                       best_epoch_test_nll_bpd, best_valid_epoch))

        # Finish wandb run.
        wandb.finish()

    # Store lls for each epoch of each run in a csv file.
    file_name = f'training_lls.csv'
    with open(output_dir + file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['run', 'epoch', 'train_ll (bpd)', 'valid_ll (bpd)'])
        writer.writerows(run_lls)

    # Calculate mean and standard deviation of results over the different dataset runs.
    results_runs = torch.tensor(results)
    mean_runs = torch.mean(results_runs, dim=0).tolist()
    std_runs = torch.std(results_runs, dim=0).tolist()

    # Save results to csv file.
    file_name = f'training_summary.csv'
    with open(output_dir + file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['run', 'train_ll (bpd)', 'valid_ll (bpd)',
                        'test_ll (bpd)', 'best_valid_epoch', ])
        writer.writerows(results)
        writer.writerow(['mean', mean_runs[1], mean_runs[2],
                        mean_runs[3], mean_runs[4]])
        writer.writerow(['std', std_runs[1], std_runs[2],
                        std_runs[3], std_runs[4]])


if __name__ == '__main__':
    # Parse args for model.
    args = parse_args()
    # Train baseline models according to arg specifications.
    main(args)
