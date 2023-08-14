import argparse
import csv
import torch
from EinsumNetwork import Graph, EinsumNetwork
from datasets import Dataset, load_data
import os
import numpy as np
import matplotlib.pyplot as plt
from pytorch_fid import fid_score
from EinsumNetwork.EinsumNetwork import window_sample
import pickle as pkl
import random


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation of EiNet models."""
    parser = argparse.ArgumentParser(
        description='EiNet CLL test evaluation argument parser.')
    parser.add_argument('--data_i', type=str, default='../data/datasets/', metavar='D',
                        help='directory where the input data is stored (default: ../data/input/datasets/)')
    parser.add_argument('--model_i', type=str, default='../data/output/cll_training/', metavar='D',
                        help='directory where the EiNet models to be evaluated are stored (default: ../data/output/cll_training/)')
    parser.add_argument('-K', type=int, default=32, metavar='D',
                        help='number of vectorised distributions in sum nodes and leaf inputs (default: 32)')
    parser.add_argument('-o', type=str, default='../data/output/cll_evaluation/', metavar='D',
                        help='directory where the outputs of training are to be stored (default: ../data/output/cll_evaluation/)')
    parser.add_argument("--test_samples_per_img", type=int, default=3,
                        help="Number of samples to use for CLL testing per img (default: 3).")
    parser.add_argument('--pd_deltas', type=str, default='7,28', metavar='D',
                        help='Poon-Domingos structre delta step sizes for image splits (default: 7,28)')
    parser.add_argument("--dataset", type=str, default="mnist",
                        help="Dataset to use (default: mnist).")
    parser.add_argument("--patch_prob", type=str, default=1.0,
                        help="Patching probability used for CLL training (default: 1.0).")
    parser.add_argument('--patch_size', type=str, default='4,4', metavar='D',
                        help='Window width and height (in this order) for sampled marginal during CLL training.')
    parser.add_argument('--sgd', action='store_true',
                        help='Whether to evaluate FID score on MLE sgd-trained models')
    parser.add_argument('--em', action='store_true',
                        help='Whether to evaluate FID score on EM trained models')
    parser.add_argument("--ccll_test", action='store_true',
                        help='Boolean to indicate if want to evaluate ccll on test set.')
    parser.add_argument("--fid", action='store_true',
                        help='Boolean to indicate if want to evaluate fid of models.')
    parser.add_argument("--fid_inpaint", action='store_true',
                        help='Boolean to indicate if want to evaluate fid on test set for inpainting evaluation.')
    parser.add_argument("--grid_prob", type=float, default=0.1,
                        help='Probability used in selecting patches in grid patching.')
    parser.add_argument("--num_bin_bisections", type=int, default=3,
                        help='Number of binary bisections used in bisection sampling.')
    parser.add_argument("--lr", type=float, default=0.01,
                        help='Learning rate used for MLE finetuning.')
    parser.add_argument("--grid_patching", action='store_true',
                        help='Boolean to indicate if want to evaluate grid patching.')
    parser.add_argument("--bisection_sampling", action='store_true',
                        help='Boolean to indicate if want to evaluate bisection sampling.')

    return parser.parse_args()


def cll_evaluation(args) -> None:
    """Script that evaluates CCLLs of EiNet on the test set of a chosen dataset, results are saved in a csv file."""
    # Load command line arguments.
    data_input_dir = args.data_i
    model_input_dir = args.model_i
    output_dir = args.o
    pd_deltas = [int(delta) for delta in args.pd_deltas.split(',')]
    k = args.K
    test_samples_per_img = args.test_samples_per_img
    dataset_name = args.dataset
    patch_prob = args.patch_prob
    patch_size = [int(dim) for dim in args.patch_size.split(',')]
    grid_prob = args.grid_prob
    sgd = args.sgd
    em = args.em
    num_bin_bisections = args.num_bin_bisections
    grid_patching = args.grid_patching
    bisection_sampling = args.bisection_sampling

    # Argument checking.
    if len(patch_size) > 2 or len(patch_size) < 1:
        raise ValueError(
            'Window dimensions must be a list of length of 1 or 2.')
    if len(patch_size) < 2:
        patch_size.append(patch_size[0])

    # Create patch window dimensions dictionary.
    patch_window_dims = {
        "width": patch_size[0], "height": patch_size[1]}

    # Load dataset.
    if dataset_name == "mnist":
        dataset = Dataset.MNIST
        img_dims = {'height': 28, 'width': 28}
        num_pixel_vars = img_dims['height'] * img_dims['width']
        data_input_dir = data_input_dir + 'mnist/'
    elif dataset_name == 'f_mnist':
        dataset = Dataset.F_MNIST
        img_dims = {'height': 28, 'width': 28}
        num_pixel_vars = img_dims['height'] * img_dims['width']
        data_input_dir = data_input_dir + 'f_mnist/'

    # Setup input and output directories.
    if grid_patching:
        # Grid patching evaluation.
        print(f'Evaluating test CCLL of grid patching model with grid prob {grid_prob} and patch size {str(patch_window_dims["width"])}, {str(patch_window_dims["height"])}')
        model_input_dir = model_input_dir + \
            f'cll_training/{dataset_name}/patch_size_{str(patch_window_dims["width"])}_{str(patch_window_dims["height"])}/patch_prob_{patch_prob}/grid_patch/grid_prob_{grid_prob}/models/'
        output_dir = output_dir + \
            f'cll_evaluation/{dataset_name}/patch_size_{str(patch_window_dims["width"])}_{str(patch_window_dims["height"])}/patch_prob_{patch_prob}/grid_patch/grid_prob_{grid_prob}/'
    elif bisection_sampling:
        # Bisection sampling.
        print(f"Evaluating test CCLL of bisection sampling model with {num_bin_bisections} bisections.")
        model_input_dir = model_input_dir + \
            f'cll_training/{dataset_name}/patch_prob_{patch_prob}/bisection_sampling/num_bin_bisections_{num_bin_bisections}/models/'
        output_dir = output_dir + \
            f'cll_evaluation/{dataset_name}/patch_prob_{patch_prob}/bisection_sampling/num_bin_bisections_{num_bin_bisections}/'
    elif sgd:
        # SGD MLE baseline evaluation.
        print(f"Evaluating test CCLL of SGD MLE baseline model.")
        model_input_dir = model_input_dir + f'baselines/{dataset_name}/sgd/models/'
        output_dir = output_dir + f'baseline_evaluation/{dataset_name}/sgd/'
    elif em:
        # EM baseline evaluation.
        print(f"Evaluating test CCLL of EM baseline model.")
        model_input_dir = model_input_dir + f'baselines/{dataset_name}/em/models/'
        output_dir = output_dir + f'baseline_evaluation/{dataset_name}/em/'
    else:
        # Uniform random sampling evaluation.
        print(f"Evaluating test CCLL of uniform random sampling model with patch size {str(patch_window_dims['width'])}, {str(patch_window_dims['height'])}")
        model_input_dir = model_input_dir + \
            f'cll_training/{dataset_name}/patch_size_{str(patch_window_dims["width"])}_{str(patch_window_dims["height"])}/patch_prob_{patch_prob}/models/'
        output_dir = output_dir + \
            f'cll_evaluation/{dataset_name}/patch_size_{str(patch_window_dims["width"])}_{str(patch_window_dims["height"])}/patch_prob_{patch_prob}/'

    # Load test data.
    _, _, test_x, _ = load_data(dataset, data_input_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_x = torch.from_numpy(test_x).to(device)

    # First check that the input directory exists and has files in it.
    if not os.path.exists(model_input_dir) or not os.listdir(model_input_dir):
        raise ValueError(f'Models specfied at directory {model_input_dir} for evaluation do not exist.')

    # Loop over models, loading each one and computing CCLL on test set.
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

    # Get file name in model_input_dir directory, if more than one, just takes the first.
    model_file = os.listdir(model_input_dir)[0]

    # Load trained model.
    print(f"Loading model from dir: {model_input_dir + model_file}")
    einet = EinsumNetwork.EinsumNetwork(graph, args)
    einet.initialize()
    einet.to(device)
    einet.load_state_dict(torch.load(
        model_input_dir + model_file))
    einet.eval()

    # Patch sizes for which to compute the CLL of the loaded model over the test set.
    test_sample_windows = [(4, 4), (8, 8), (12, 12), (4, 12)]

    # Keep track of results for each test window as well as the average over all test windows.
    test_tracker = []
    window_test_tracker = []
    avg_cll_total = 0.0

    print("Running CCLL evaluation on test set...")
    # Loop over test windows.
    for test_window in test_sample_windows:
        # Compute ccll on test set for each test window.
        avg_ccll, ccll_bpd_std, ccll_tracker = einet.eval_test_avg_ccll_bpd(
            data_input_dir, dataset, img_dims["width"], img_dims["height"], test_window[0], test_window[1], test_samples_per_img)
        print(
            f'Avg test ccll (bpd) for test window {test_window[0]} x {test_window[1]}: {avg_ccll} bpd')
        window_test_tracker.append(
            {"patch_dims": test_window, "ccll_bpd": avg_ccll, "std dev": ccll_bpd_std})

        # Increment average cll over all test windows.
        test_tracker.extend(ccll_tracker)

    # Make output directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Compute average cll over all test windows with associated standard deviation.
    avg_ccll_total = np.mean(test_tracker)
    std_dev_total = np.std(test_tracker)

    print(
        f'Avg ccll (bpd) for all test windows: {avg_ccll_total} bpd, std dev: {std_dev_total}')

    # Save test cll results for each test window to csv file.
    file_name = 'test_cclls.csv'
    with open(output_dir + file_name, 'w', newline='') as csvfile:
        headings = ['patch_dims', 'ccll_bpd', 'std dev']
        writer = csv.DictWriter(csvfile, fieldnames=headings)
        writer.writeheader()
        for data in window_test_tracker:
            writer.writerow(data)

    # Save average cll over all test windows to csv file.
    file_name = 'avg_test_ccll.csv'
    with open(output_dir + file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['avg_test_ccll', 'std dev'])
        writer.writerow([avg_cll_total, std_dev_total])


def fid_evaluation(args):
    """Calculate the FID scores for the EiNet models over the test of a 
    give dataset.
    """
    data_input_dir = args.data_i
    model_input_dir = args.model_i
    output_dir = args.o
    pd_deltas = [int(delta) for delta in args.pd_deltas.split(',')]
    k = args.K
    dataset_name = args.dataset
    patch_prob = args.patch_prob
    sgd = args.sgd
    patch_size = [int(dim) for dim in args.patch_size.split(',')]
    grid_prob = args.grid_prob
    num_bin_bisections = args.num_bin_bisections
    em = args.em
    grid_patching = args.grid_patching
    bisection_sampling = args.bisection_sampling

    # Argument checking.
    if len(patch_size) > 2 or len(patch_size) < 1:
        raise ValueError(
            'Window dimensions must be a list of length of 1 or 2.')
    if len(patch_size) < 2:
        patch_size.append(patch_size[0])

    # Create patch window dimensions dictionary.
    patch_window_dims = {
        "width": patch_size[0], "height": patch_size[1]}

    # Load dataset.
    if dataset_name == "mnist":
        dataset = Dataset.MNIST
        img_dims = {'height': 28, 'width': 28}
        num_pixel_vars = img_dims['height'] * img_dims['width']
        data_input_dir = data_input_dir + 'mnist/'
    elif dataset_name == 'f_mnist':
        dataset = Dataset.F_MNIST
        img_dims = {'height': 28, 'width': 28}
        num_pixel_vars = img_dims['height'] * img_dims['width']
        data_input_dir = data_input_dir + 'f_mnist/'
    
    # Setup input and output directories.
    if grid_patching:
        # Grid patching evaluation.
        print(f'Computing FID scores for grid patching model with grid prob {grid_prob} and patch size {str(patch_window_dims["width"])}, {str(patch_window_dims["height"])}')
        model_input_dir = model_input_dir + \
            f'cll_training/{dataset_name}/patch_size_{str(patch_window_dims["width"])}_{str(patch_window_dims["height"])}/patch_prob_{patch_prob}/grid_patch/grid_prob_{grid_prob}/models/'
        output_dir = output_dir + \
            f'cll_evaluation/{dataset_name}/patch_size_{str(patch_window_dims["width"])}_{str(patch_window_dims["height"])}/patch_prob_{patch_prob}/grid_patch/grid_prob_{grid_prob}/'
    elif bisection_sampling:
        # Bisection sampling.
        print(f"Computing FID scores for bisection sampling model with {num_bin_bisections} bisections.")
        model_input_dir = model_input_dir + \
            f'cll_training/{dataset_name}/patch_prob_{patch_prob}/bisection_sampling/num_bin_bisections_{num_bin_bisections}/models/'
        output_dir = output_dir + \
            f'cll_evaluation/{dataset_name}/patch_prob_{patch_prob}/bisection_sampling/num_bin_bisections_{num_bin_bisections}/'
    elif sgd:
        # SGD MLE baseline evaluation.
        print(f"Computing FID scores for SGD MLE baseline model.")
        model_input_dir = model_input_dir + f'baselines/{dataset_name}/sgd/models/'
        output_dir = output_dir + f'baseline_evaluation/{dataset_name}/sgd/'
    elif em:
        # EM baseline evaluation.
        print(f"Computing FID scores for EM baseline model.")
        model_input_dir = model_input_dir + f'baselines/{dataset_name}/em/models/'
        output_dir = output_dir + f'baseline_evaluation/{dataset_name}/em/'
    else:
        # Uniform random sampling evaluation.
        print(f"Computing FID scores for uniform random sampling model with patch size {str(patch_window_dims['width'])}, {str(patch_window_dims['height'])}")
        model_input_dir = model_input_dir + \
            f'cll_training/{dataset_name}/patch_size_{str(patch_window_dims["width"])}_{str(patch_window_dims["height"])}/patch_prob_{patch_prob}/models/'
        output_dir = output_dir + \
            f'cll_evaluation/{dataset_name}/patch_size_{str(patch_window_dims["width"])}_{str(patch_window_dims["height"])}/patch_prob_{patch_prob}/'

    # Load test data.
    _, _, test_x, _ = load_data(dataset, data_input_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_x = torch.from_numpy(test_x)

    # First check that the input directory exists and has files in it.
    if not os.path.exists(model_input_dir) or not os.listdir(model_input_dir):
        raise ValueError(f'Models specfied at directory {model_input_dir} for evaluation do not exist.')

    # Check if directory exists and is non-empty. If no, then save test set images.
    if not os.path.exists(data_input_dir + 'images'):
        os.makedirs(data_input_dir + 'images')
        for i in range(test_x.shape[0]):
            sample = test_x[i, :].reshape(
                (img_dims['height'], img_dims['width']))
            plt.imsave(data_input_dir +
                       f'images/sample_{i}.png', sample, cmap='gray')
    else:
        if not os.listdir(data_input_dir + 'images'):
            for i in range(test_x.shape[0]):
                sample = test_x[i].reshape(
                    (img_dims['height'], img_dims['width']))
                plt.imsave(data_input_dir +
                           f'images/sample_{i}.png', sample, cmap='gray')

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
    
    # Get file name in model_input_dir directory
    model_file = os.listdir(model_input_dir)[0]

    # Load model.
    einet = EinsumNetwork.EinsumNetwork(graph, args)
    einet.initialize()
    einet.to(device)
    einet.load_state_dict(torch.load(
        model_input_dir + model_file))
    einet.eval()

    # Save samples to file.
    output_dir_samples = output_dir + f'fid_samples/'
    if not os.path.exists(output_dir_samples):
        os.makedirs(output_dir_samples)

    # Sample images from trained
    print(f"Sampling images from model...{model_input_dir + model_file}")
    # Samples images in batches of 100 to avoid memory issues.
    for i in range(0, test_x.shape[0], 100):
        print(f"Sampling images {i} to {i+100}...")
        samples = einet.sample(100).cpu().numpy().reshape((-1, 28, 28))

        # Save images from batch.
        for j in range(samples.shape[0]):
            idx = i + j
            sample = samples[j, :, :]
            plt.imsave(output_dir_samples +
                       f'sample_{idx}.png', sample, cmap='gray')

    # Calculate FID score.
    print("Calculating FID score...")
    fid = fid_score.calculate_fid_given_paths(
        [data_input_dir + 'images/', output_dir_samples], 100, device, 2048)
    print(f"FID score: {fid}")


    # Save FID scores to csv file along with average and standard deviation.
    print("Saving FID scores to csv file...")
    file_name = 'fid_scores.csv'
    with open(output_dir + file_name, 'w', newline='') as csvfile:
        headings = ['fid']
        writer = csv.DictWriter(csvfile, fieldnames=headings)
        writer.writeheader()
        writer.writerow(fid)


def fid_inpainting_evaluation(args):
    """
    Evaluates FID score of trained models on inpainting tasks.
    """
    data_input_dir = args.data_i
    model_input_dir = args.model_i
    output_dir = args.o
    pd_deltas = [int(delta) for delta in args.pd_deltas.split(',')]
    k = args.K
    dataset_name = args.dataset
    patch_prob = args.patch_prob
    sgd = args.sgd
    patch_size = [int(dim) for dim in args.patch_size.split(',')]
    grid_prob = args.grid_prob
    num_bin_bisections = args.num_bin_bisections
    em = args.em
    grid_patching = args.grid_patching
    bisection_sampling = args.bisection_sampling

    # Uses the same window patches as for test CCLL evaluation.
    test_sample_windows = [(4, 4), (8, 8), (12, 12), (4, 12)]

    # Argument checking.
    if len(patch_size) > 2 or len(patch_size) < 1:
        raise ValueError(
            'Window dimensions must be a list of length of 1 or 2.')
    if len(patch_size) < 2:
        patch_size.append(patch_size[0])

    # Create patch window dimensions dictionary.
    patch_window_dims = {
        "width": patch_size[0], "height": patch_size[1]}
    
    # Load dataset.
    if dataset_name == "mnist":
        dataset = Dataset.MNIST
        img_dims = {'height': 28, 'width': 28}
        num_pixel_vars = img_dims['height'] * img_dims['width']
        data_input_dir = data_input_dir + 'mnist/'
    elif dataset_name == 'f_mnist':
        dataset = Dataset.F_MNIST
        img_dims = {'height': 28, 'width': 28}
        num_pixel_vars = img_dims['height'] * img_dims['width']
        data_input_dir = data_input_dir + 'f_mnist/'

    # Setup input and output directories.
    if grid_patching:
        # Grid patching evaluation.
        print(f'Computing FID inpainting scores for grid patching model with grid prob {grid_prob} and patch size {str(patch_window_dims["width"])}, {str(patch_window_dims["height"])}')
        model_input_dir = model_input_dir + \
            f'cll_training/{dataset_name}/patch_size_{str(patch_window_dims["width"])}_{str(patch_window_dims["height"])}/patch_prob_{patch_prob}/grid_patch/grid_prob_{grid_prob}/models/'
        output_dir = output_dir + \
            f'cll_evaluation/{dataset_name}/patch_size_{str(patch_window_dims["width"])}_{str(patch_window_dims["height"])}/patch_prob_{patch_prob}/grid_patch/grid_prob_{grid_prob}/'
    elif bisection_sampling:
        # Bisection sampling.
        print(f"Computing FID inpainting scores for bisection sampling model with {num_bin_bisections} bisections.")
        model_input_dir = model_input_dir + \
            f'cll_training/{dataset_name}/patch_prob_{patch_prob}/bisection_sampling/num_bin_bisections_{num_bin_bisections}/models/'
        output_dir = output_dir + \
            f'cll_evaluation/{dataset_name}/patch_prob_{patch_prob}/bisection_sampling/num_bin_bisections_{num_bin_bisections}/'
    elif sgd:
        # SGD MLE baseline evaluation.
        print(f"Computing FID inpainting scores for SGD MLE baseline model.")
        model_input_dir = model_input_dir + f'baselines/{dataset_name}/sgd/models/'
        output_dir = output_dir + f'baseline_evaluation/{dataset_name}/sgd/'
    elif em:
        # EM baseline evaluation.
        print(f"Computing FID inpainting scores for EM baseline model.")
        model_input_dir = model_input_dir + f'baselines/{dataset_name}/em/models/'
        output_dir = output_dir + f'baseline_evaluation/{dataset_name}/em/'
    else:
        # Uniform random sampling evaluation.
        print(f"Computing FID inpainting scores for uniform random sampling model with patch size {str(patch_window_dims['width'])}, {str(patch_window_dims['height'])}")
        model_input_dir = model_input_dir + \
            f'cll_training/{dataset_name}/patch_size_{str(patch_window_dims["width"])}_{str(patch_window_dims["height"])}/patch_prob_{patch_prob}/models/'
        output_dir = output_dir + \
            f'cll_evaluation/{dataset_name}/patch_size_{str(patch_window_dims["width"])}_{str(patch_window_dims["height"])}/patch_prob_{patch_prob}/'

    # Load test data.
    _, _, test_x, _ = load_data(dataset, data_input_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_x = torch.from_numpy(test_x).to(device)

    # Check if if patch_sample_windows.pkl exists containg patch locations for inpainting
    # If not, then create it.
    if not os.path.exists(data_input_dir + 'inpaint_samples/patch_sample_windows.pkl'):
        if not os.path.exists(data_input_dir + 'inpaint_samples/'):
            os.makedirs(data_input_dir + 'inpaint_samples/')
        print("Saving test set patch samples...")
        patch_sample_idxs = []
        for i in range(test_x.shape[0]):
            # Sample patch from image, reshaping to inpatch_patch_size x inpaint_patch_size and then saving.
            # Randomly choose patch size for patch sample from test_sample_windows.
            inpaint_patch_size = random.choice(test_sample_windows)
            sample = window_sample(img_dims["height"], img_dims["width"],
                                   inpaint_patch_size[0], inpaint_patch_size[1], device=device).cpu().numpy()
            patch_sample_idxs.append(sample)

        # Save patch sample idxs to pickle file.
        with open(data_input_dir + 'inpaint_samples/patch_sample_windows.pkl', 'wb') as f:
            pkl.dump(patch_sample_idxs, f)
    else:
        print("Loading test set patch samples...")
        with open(data_input_dir + 'inpaint_samples/patch_sample_windows.pkl', 'rb') as f:
            patch_sample_idxs = pkl.load(f)

    # Get file name in model_input_dir directory
    model_file = os.listdir(model_input_dir)[0]

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

    # Load model.
    einet = EinsumNetwork.EinsumNetwork(graph, args)
    einet.initialize()
    einet.to(device)
    einet.load_state_dict(torch.load(
        model_input_dir + model_file))
    einet.eval()

    # Create directory to save examples if it doesn't already exist.
    output_dir_samples = output_dir + f'fid_inpaint_samples/'
    if not os.path.exists(output_dir_samples):
        os.makedirs(output_dir_samples)

    # Sample images from trained model.
    print(f"Sampling images from model...{model_input_dir + model_file}")

    samples = []
    # counter = 0
    for i in range(test_x.shape[0]):
        if i % 500 == 0:
            print(f"Sampling image {i}")
        inpaint_patch = patch_sample_idxs[i]
        # Perform MAP inference to inpaint patch location.
        einet.remove_marginalization_idx()
        einet.set_marginalization_idx(inpaint_patch)
        inpainted_image = einet.mpe(
            x=test_x[i, :].unsqueeze(0)).cpu().numpy().squeeze()
        samples.append(inpainted_image)

    # Save samples to file.
    for idx, sample in enumerate(samples):
        plt.imsave(output_dir_samples + f'sample_{idx}.png', sample.reshape(
            (img_dims["height"], img_dims["width"])), cmap='gray')

    # Calculate FID score.
    print("Calculating FID score...")
    fid_inp = fid_score.calculate_fid_given_paths(
        [data_input_dir + 'images/', output_dir_samples], 100, device, 2048)
    print(f"FID_inp score: {fid_inp}")

    # Save FID scores to csv file along with average and standard deviation.
    print("Saving FID scores to csv file...")
    file_name = 'fid_inpaint_scores.csv'
    with open(output_dir + file_name, 'w', newline='') as csvfile:
        headings = ['fid_inp']
        writer = csv.DictWriter(csvfile, fieldnames=headings)
        writer.writeheader()
        writer.writerow(fid_inp)


if __name__ == '__main__':
    args = parse_args()
    if args.ccll_test:
        cll_evaluation(args)
    elif args.fid:
        fid_evaluation(args)
    elif args.fid_inpaint:
        fid_inpainting_evaluation(args)
