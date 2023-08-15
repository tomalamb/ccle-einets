import torch
from EinsumNetwork import Graph, EinsumNetwork
import argparse
import csv
import numpy as np
from datasets import Dataset, load_data
import os
import wandb
import sys
import re


def parse_args() -> argparse.Namespace:
    """Function that gathers the the arguments used for training
    a baseline EiNet model.

    Returns:
        argparse.Namespace: Argparser object which stores training arguments for the EiNet model to be trained.
    """
    parser = argparse.ArgumentParser(
        description='EiNet argument parser.')
    parser.add_argument('--data_i', type=str, default='data/datasets', metavar='D',
                        help='directory where the input data is stored (default: data/input/datasets')
    parser.add_argument('--model_i', type=str, default='data/output', metavar='D',
                        help='directory where the EiNet models to be evaluated are stored (default: data/output)')
    parser.add_argument('-o', type=str, default='data/output', metavar='D',
                        help='directory where the outputs of training are to be stored (default: data/output')
    parser.add_argument('-K', type=int, default=10, metavar='D',
                        help='number of vectorised distributions in sum nodes and leaf inputs (default: 10)')
    parser.add_argument('--max_num_epochs', type=int, default=64, metavar='D',
                        help='maximum number of training epochs (default: 64)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='D',
                        help='batch size (default: 32)')
    parser.add_argument('--patience', type=int, default=8, metavar='D',
                        help='patience for early stopping (default: 8)')
    parser.add_argument('--num_runs', type=int, default=1, metavar='D',
                        help='number of runs (default: 1)')
    parser.add_argument('--pd_deltas', type=str, default='7,14', metavar='D',
                        help='Poon-Domingos structre delta step sizes for image splits (default: 1)')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
                        help='dataset to train on, (default: mnist, options are mnist, f_mnist).')
    parser.add_argument('--optimiser', type=str, default='adam', metavar='D',
                        help='optimise (default: adam,, options are adam, sgd).')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='D',
                        help='initial learning rate for optimiser (default: 1e-3)')
    parser.add_argument('--bisection_sampling', action='store_true',
                        help='Whether to finetune bisection model.')
    parser.add_argument('--grid_sampling', action='store_true',
                        help='Whether to finetune bisection model.')
    parser.add_argument('--patch_prob', type=float, default=1.0, metavar='D',
                        help='Probability of calcualting ccll for patch of image over the likelihood of the full image. (default: 1.0)')
    parser.add_argument('--patch_size', type=str, default='4,4', metavar='D',
                        help='Window width and height (in this order) for sampled marginal during ccle training.')
    parser.add_argument('--wandb_online', action='store_true',
                        help='Whether to use wandb for logging. If not provided, the default is False.')
    parser.add_argument('--wandb_project', type=str, default='EiNets - MLE Finetuning', metavar='D',
                        help='Wandb project name (default: EiNets).')
    parser.add_argument('--runs', type=int, default=1, metavar='D',
                        help='Number of runs to perform. If not provided, the default is 1.')
    parser.add_argument('--num_bin_bisections', type=int, default=1, metavar='D',
                        help='Number of bisections to use when splitting image into patches. If not provided, the default is 1.')
    parser.add_argument('--grid_prob', type=float, default=1.0, metavar='D',
                        help='Probability of choosing patch within grid region.  (default: 1.0)')
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    """Main function used to MLE finetune CCLE trained-EiNets according to the given arguments. Results are saved to a file.
    Args:
        args (argparse.Namespace): MLE fine-tuning training arguments.
    """
    print("Staring MLE finetuning training script...")

    # Parse training arguments to create baseline model.
    data_input_dir = args.data_i
    model_input_dir = args.model_i
    output_dir = args.o
    num_runs = args.num_runs
    max_num_epochs = args.max_num_epochs
    batch_size = args.batch_size
    patience = args.patience
    pd_deltas = [int(delta) for delta in args.pd_deltas.split(',')]
    dataset_name = args.dataset
    patch_size = [int(dim) for dim in args.patch_size.split(',')]
    lr = args.lr
    patch_prob = args.patch_prob
    wandb_project = args.wandb_project
    wandb_online = args.wandb_online
    num_bin_bisections = args.num_bin_bisections
    grid_prob = args.grid_prob
    bisection_sampling = args.bisection_sampling
    grid_sampling = args.grid_sampling

    # Main hyperparameters for the baseline model.
    k = args.K

    # Check window dimensions are for CLL training are valid
    # and generate identifier for run.
    if len(patch_size) > 2 or len(patch_size) < 1:
        raise ValueError(
            'Window dimensions must be a list of length of 1 or 2.')

    # If the patch size is less than 2, assume square window and change patch size to be list with two elements.
    if len(patch_size) < 2:
        patch_size.append(patch_size[0])

    # Create patch window dimensions dictionary.
    patch_window_dims = {
        "width": patch_size[0], "height": patch_size[1]}

    identifier = f'mle_finetuning_k={k}_dataset={dataset_name}_lr={lr}'

    # Set up dataset and associated parameters.
    if dataset_name == 'mnist':
        dataset = Dataset.MNIST
        img_dims = {'height': 28, 'width': 28}
        num_pixel_vars = img_dims['height'] * img_dims['width']
        valid_size = 10000
        data_input_dir = data_input_dir + '/mnist/'
    elif dataset_name == 'f_mnist':
        dataset = Dataset.F_MNIST
        img_dims = {'height': 28, 'width': 28}
        num_pixel_vars = img_dims['height'] * img_dims['width']
        valid_size = 10000
        data_input_dir = data_input_dir + '/f_mnist/'
    else:
        raise ValueError('Dataset not supported.')

    # Choose model to finetune.
    # Setup input and output directories.
    if grid_sampling:
        # Grid patching evaluation.
        print(
            f'MLE-finetuning grid patching model with grid prob {grid_prob} and patch size {str(patch_window_dims["width"])}, {str(patch_window_dims["height"])}')
        model_input_dir = model_input_dir + \
            f'/ccle_training/{dataset_name}/patch_size_{str(patch_window_dims["width"])}_{str(patch_window_dims["height"])}/patch_prob_{patch_prob}/grid_patch/grid_prob_{grid_prob}/models/'
        output_dir = output_dir + \
            f'/mle_finetuning/{dataset_name}/patch_size_{str(patch_window_dims["width"])}_{str(patch_window_dims["height"])}/patch_prob_{patch_prob}/grid_patch/grid_prob_{grid_prob}/'
        identifier = identifier + \
            f'_grid_sampling_{grid_sampling}_grid_prob_{grid_prob}'
    elif bisection_sampling:
        # Bisection sampling.
        print(
            f"MLE-finetuning bisection sampling model with {num_bin_bisections} bisections.")
        model_input_dir = model_input_dir + \
            f'/ccle_training/{dataset_name}/bisection_sampling/num_bin_bisections_{num_bin_bisections}/models/'
        output_dir = output_dir + \
            f'/mle_finetuning/{dataset_name}/bisection_sampling/num_bin_bisections_{num_bin_bisections}/'
        identifier = identifier + f'_bisection_sampling_{num_bin_bisections}'
    else:
        # Uniform random sampling evaluation.
        print(
            f"MLE-finetuning uniform random sampling model with patch size {str(patch_window_dims['width'])}, {str(patch_window_dims['height'])}")
        model_input_dir = model_input_dir + \
            f'/ccle_training/{dataset_name}/patch_size_{str(patch_window_dims["width"])}_{str(patch_window_dims["height"])}/patch_prob_{patch_prob}/models/'
        output_dir = output_dir + \
            f'/mle_finetuning/{dataset_name}/patch_size_{str(patch_window_dims["width"])}_{str(patch_window_dims["height"])}/patch_prob_{patch_prob}/'
        identifier = identifier + \
            f'_patch_prob_{str(patch_prob)}_patch_size_{str(patch_window_dims["width"])}_{str(patch_window_dims["height"])}'

    # Keep track of lls for each epoch of each run.
    run_lls = []
    results = []

    # Load dataset.
    train_x, _, test_x, _ = load_data(dataset, data_input_dir)

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

    # Get specfied run file in model directory.
    model_file = os.listdir(model_input_dir)[0]

    print("Starting fine-tuning...")
    # Loop over runs for significance testing if required.
    for run in range(num_runs):
        # Clear GPU cache betweem runs.
        torch.cuda.empty_cache()
        # Set up wandb logging.
        config = vars(args)
        config['run'] = run
        config['identifier'] = identifier

        if wandb_online:
            wandb.init(project=wandb_project,
                       name=f"MLE finetune run {identifier} {run}", config=config)
        else:
            wandb.init(project=wandb_project,
                       name=f"MLE finetune run {identifier} {run}", config=config, mode='dryrun')

        # Clear GPU cache betweem runs.
        torch.cuda.empty_cache()

        # Create EiNet DAG using PD structure.
        pd_delta = [[img_dims["height"] / delta, img_dims["width"] / delta]
                    for delta in pd_deltas]
        graph = Graph.poon_domingos_structure(
            shape=(img_dims["height"], img_dims["width"]), delta=pd_delta)

        # Create EiNet model using the arguments passed.
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

        # Load trained model.
        print(f"Loading model ...", model_input_dir + model_file)
        einet = EinsumNetwork.EinsumNetwork(graph, args)
        einet.initialize()
        einet.to(device)
        einet.load_state_dict(torch.load(
            model_input_dir + model_file))

        # Print the number of trainable parameters in the model.
        num_parameters = sum(p.numel()
                             for p in einet.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_parameters}")

        # Initialise optimiser.
        optimiser = torch.optim.Adam(einet.parameters(), lr=lr)

        # Evaluate pre-trained model's performance on training, validation, and test sets.
        einet.eval()
        with torch.no_grad():
            initial_train_ll_bpd = einet.avg_neg_ll_bpd(train_x)
            initial_valid_ll_bpd = einet.avg_neg_ll_bpd(valid_x)
            initial_test_ll_bpd = einet.avg_neg_ll_bpd(test_x)
            print(
                f"CLL pretrained model's: train log-likelihood: {initial_train_ll_bpd}, valid log-likelihood: {initial_valid_ll_bpd}, test log-likelihood: {initial_test_ll_bpd}")

            run_lls.append([-1, initial_train_ll_bpd, initial_valid_ll_bpd])

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
                wandb.log({'Avg negative LL loss': objective.item()})

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
                               f'models/einet_model_{run}.pt')

                # Save training and validation log-likelihoods for each epoch.
                run_lls.append(
                    (run, epoch, train_ll_bpd, valid_ll_bpd))
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
            output_dir + f'models/einet_model_{run}.pt'))
        best_epoch_test_ll_bpd = einet.avg_neg_ll_bpd(test_x)
        wandb.log({"test_ll (bpd)": best_epoch_test_ll_bpd})

        # Compute differences in log-likelihoods as a result of finetuning.
        train_differs = best_epoch_train_ll_bpd - initial_train_ll_bpd
        valid_differs = best_valid_ll_bpd - initial_valid_ll_bpd
        test_differs = best_epoch_test_ll_bpd - initial_test_ll_bpd

        results.append((run, initial_train_ll_bpd, best_epoch_train_ll_bpd, train_differs, initial_valid_ll_bpd, best_valid_ll_bpd, valid_differs,
                        initial_test_ll_bpd, best_epoch_test_ll_bpd, test_differs,  best_valid_epoch))

        # Log differences in pre-trained models and best models.
        wandb.log({"train_ll_diff": best_epoch_train_ll_bpd - initial_train_ll_bpd,
                   "valid_ll_diff": best_valid_ll_bpd - initial_valid_ll_bpd,
                   "test_ll_diff": best_epoch_test_ll_bpd - initial_test_ll_bpd})

        print(
            f'Best epoch train ll: {best_epoch_train_ll_bpd}, Best epoch valid ll: {best_valid_ll_bpd}, Best epoch test ll: {best_epoch_test_ll_bpd}')

        print(
            f"Train log-likelihood difference: {train_differs}, Valid log-likelihood difference: {valid_differs}, Test log-likelihood difference: {test_differs}")

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
        writer.writerow(['run', 'pre_trained train_ll (bpd)', 'train_ll (bpd)', 'train_ll_difference  (bpd)', 'pre_trained_valid_ll (bpd)',
                        'valid_ll (bpd)', 'valid_ll_difference (bpd)', 'pre_trained_test_ll (bpd)', 'test_ll (bpd)', 'test_ll_difference (bpd)', 'best_valid_epoch'])
        writer.writerows(results)
        writer.writerow(['mean', mean_runs[1], mean_runs[2], mean_runs[3], mean_runs[4],
                        mean_runs[5], mean_runs[6], mean_runs[7], mean_runs[8], mean_runs[9], mean_runs[10]])
        writer.writerow(['std', std_runs[1], std_runs[2], std_runs[3], std_runs[4],
                        std_runs[5], std_runs[6], std_runs[7], std_runs[8], std_runs[9], std_runs[10]])


if __name__ == '__main__':
    # Parse args for model.
    args = parse_args()
    # Train baseline models according to arg specifications.
    main(args)
