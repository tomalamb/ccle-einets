"""
Class for creating Einsum Networks (EiNets). Code is based on the original
implementation by Peharz et al. (2020) whose repository can be found at
https://github.com/cambridge-mlg/EinsumNetworks.
"""

from torch.nn.parameter import Parameter
from typing import Iterator
from EinsumNetwork import Graph
from EinsumNetwork.FactorizedLeafLayer import *
from EinsumNetwork.SumLayer import *
from torch.utils.checkpoint import checkpoint
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import pickle as pkl
from datasets import load_data


class Args(object):
    """
    Arguments for EinsumNetwork class.

    num_var: number of random variables (RVs). An RV might be multidimensional though -- see num_dims.
    num_dims: number of dimensions per RV. E.g. you can model an 32x32 RGB image as an 32x32 array of three dimensional
              RVs.
    num_input_distributions: number of distributions per input region (K in the paper).
    num_sums: number of sum nodes per internal region (K in the paper).
    num_classes: number of outputs of the PC.
    exponential_family: which exponential family to use; (sub-class ExponentialFamilyTensor).
    exponential_family_args: arguments for the exponential family, e.g. trial-number N for Binomial.
    use_em: determines if the internal em algorithm shall be used; otherwise you might use e.g. SGD.
    online_em_frequency: how often shall online be triggered in terms, of batches? 1 means after each batch, None means
                         batch EM. In the latter case, EM updates must be triggered manually after each epoch.
    online_em_stepsize: stepsize for inline EM. Only relevant if online_em_frequency not is None.
    """

    def __init__(self,
                 num_var=20,
                 num_dims=1,
                 num_input_distributions=10,
                 num_sums=10,
                 num_classes=1,
                 exponential_family=NormalArray,
                 exponential_family_args=None,
                 use_em=True,
                 online_em_frequency=1,
                 online_em_stepsize=0.05,
                 img_dims=None,
                 num_conditionals=1,
                 device='cpu'):
        self.num_var = num_var
        self.num_dims = num_dims
        self.num_input_distributions = num_input_distributions
        self.num_sums = num_sums
        self.num_classes = num_classes
        self.exponential_family = exponential_family
        if exponential_family_args is None:
            exponential_family_args = {}
        self.exponential_family_args = exponential_family_args
        self.use_em = use_em
        self.online_em_frequency = online_em_frequency
        self.online_em_stepsize = online_em_stepsize
        self.img_dims = img_dims
        self.device = device
        self.num_conditionals = num_conditionals


class EinsumNetwork(torch.nn.Module):
    """
    Implements Einsum Networks (EiNets).

    The basic philosophy of EiNets is to summarize many PC nodes in monolithic GPU-friendly parallel operations.
    An EiNet can be seen as a special layered feed-forward neural network, consisting of a sequence of layers. Each
    layer can in principle get input from all layers before.

    As a general design principle, each layer in EinsumNetworks produces a tensor of log-densities in the forward pass,
    of generic shape
            (batch_size, vector_length, num_nodes)
    where
        batch_size is the number of samples in a mini-batch.
        vector_length is the length of the vectorized operations; this is called K in the paper -- in the paper we
                      assumed this constant over the whole EiNet, but this can be partially relaxed.
        num_nodes is the number of nodes which are realized in parallel using this layer.
    Thus, in classical PCs, we would interpret the each layer as a collection of vector_length * num_nodes PC nodes.

    The class EinsumNetork mainly governs the layer-wise layout, initialization, forward() calls, EM learning, etc.
    """

    def __init__(self, graph, args=None):
        """Make an EinsumNetwork."""
        super(EinsumNetwork, self).__init__()

        check_flag, check_msg = Graph.check_graph(graph)
        if not check_flag:
            raise AssertionError(check_msg)
        self.graph = graph

        self.args = args if args is not None else Args()

        if len(Graph.get_roots(self.graph)) != 1:
            raise AssertionError(
                "Currently only EinNets with single root node supported.")

        root = Graph.get_roots(self.graph)[0]
        if tuple(range(self.args.num_var)) != root.scope:
            raise AssertionError(
                "The graph should be over tuple(range(num_var)).")

        for node in Graph.get_leaves(self.graph):
            node.num_dist = self.args.num_input_distributions

        for node in Graph.get_sums(self.graph):
            if node is root:
                node.num_dist = self.args.num_classes
            else:
                node.num_dist = self.args.num_sums

        # Algorithm 1 in the paper -- organize the PC in layers
        self.graph_layers = Graph.topological_layers(self.graph)

        # input layer
        einet_layers = [FactorizedLeafLayer(self.graph_layers[0],
                                            self.args.num_var,
                                            self.args.num_dims,
                                            self.args.exponential_family,
                                            self.args.exponential_family_args,
                                            use_em=self.args.use_em)]

        # internal layerss
        for c, layer in enumerate(self.graph_layers[1:]):
            if c % 2 == 0:   # product layer
                einet_layers.append(EinsumLayer(
                    self.graph, layer, einet_layers, use_em=self.args.use_em))
            else:     # sum layer
                # the Mixing layer is only for regions which have multiple partitions as children.
                multi_sums = [n for n in layer if len(graph.succ[n]) > 1]
                if multi_sums:
                    einet_layers.append(EinsumMixingLayer(
                        graph, multi_sums, einet_layers[-1], use_em=self.args.use_em))

        self.einet_layers = torch.nn.ModuleList(einet_layers)
        self.em_set_hyperparams(
            self.args.online_em_frequency, self.args.online_em_stepsize)

    def initialize(self, init_dict=None):
        """
        Initialize layers.

        :param init_dict: None; or
                          dictionary int->initializer; mapping layer index to initializers; or
                          dictionary layer->initializer;
                          the init_dict does not need to have an initializer for all layers
        :return: None
        """
        if init_dict is None:
            init_dict = dict()
        if all([type(k) == int for k in init_dict.keys()]):
            init_dict = {self.einet_layers[k]: init_dict[k]
                         for k in init_dict.keys()}
        for layer in self.einet_layers:
            layer.initialize(init_dict.get(layer, 'default'))

    def set_marginalization_idx(self, idx):
        """Set indices of marginalized variables."""
        self.einet_layers[0].set_marginalization_idx(idx)

    def remove_marginalization_idx(self):
        """Remove marginalized variables."""
        self.einet_layers[0].remove_marginalization_idx()

    def get_marginalization_idx(self):
        """Get indices of marginalized variables."""
        return self.einet_layers[0].get_marginalization_idx()

    def forward(self, x):
        """Evaluate the EinsumNetwork feed forward."""

        input_layer = self.einet_layers[0]
        input_layer(x=x)
        for einsum_layer in self.einet_layers[1:]:
            einsum_layer()
        return self.einet_layers[-1].prob[:, :, 0]

    def backtrack(self, num_samples=1, class_idx=0, x=None, mode='sampling', **kwargs):
        """
        Perform backtracking; for sampling or MPE approximation.
        """

        sample_idx = {l: [] for l in self.einet_layers}
        dist_idx = {l: [] for l in self.einet_layers}
        reg_idx = {l: [] for l in self.einet_layers}

        root = self.einet_layers[-1]

        if x is not None:
            self.forward(x)
            num_samples = x.shape[0]

        sample_idx[root] = list(range(num_samples))
        dist_idx[root] = [class_idx] * num_samples
        reg_idx[root] = [0] * num_samples

        for layer in reversed(self.einet_layers):

            if not sample_idx[layer]:
                continue

            if type(layer) == EinsumLayer:

                ret = layer.backtrack(dist_idx[layer],
                                      reg_idx[layer],
                                      sample_idx[layer],
                                      use_evidence=(x is not None),
                                      mode=mode,
                                      **kwargs)
                dist_idx_left, dist_idx_right, reg_idx_left, reg_idx_right, layers_left, layers_right = ret

                for c, layer_left in enumerate(layers_left):
                    sample_idx[layer_left].append(sample_idx[layer][c])
                    dist_idx[layer_left].append(dist_idx_left[c])
                    reg_idx[layer_left].append(reg_idx_left[c])

                for c, layer_right in enumerate(layers_right):
                    sample_idx[layer_right].append(sample_idx[layer][c])
                    dist_idx[layer_right].append(dist_idx_right[c])
                    reg_idx[layer_right].append(reg_idx_right[c])

            elif type(layer) == EinsumMixingLayer:

                ret = layer.backtrack(dist_idx[layer],
                                      reg_idx[layer],
                                      sample_idx[layer],
                                      use_evidence=(x is not None),
                                      mode=mode,
                                      **kwargs)
                dist_idx_out, reg_idx_out, layers_out = ret

                for c, layer_out in enumerate(layers_out):
                    sample_idx[layer_out].append(sample_idx[layer][c])
                    dist_idx[layer_out].append(dist_idx_out[c])
                    reg_idx[layer_out].append(reg_idx_out[c])

            elif type(layer) == FactorizedLeafLayer:

                unique_sample_idx = sorted(list(set(sample_idx[layer])))
                if unique_sample_idx != sample_idx[root]:
                    raise AssertionError("This should not happen.")

                dist_idx_sample = []
                reg_idx_sample = []
                for sidx in unique_sample_idx:
                    dist_idx_sample.append(
                        [dist_idx[layer][c] for c, i in enumerate(sample_idx[layer]) if i == sidx])
                    reg_idx_sample.append(
                        [reg_idx[layer][c] for c, i in enumerate(sample_idx[layer]) if i == sidx])

                samples = layer.backtrack(
                    dist_idx_sample, reg_idx_sample, mode=mode, **kwargs)

                if self.args.num_dims == 1:
                    samples = torch.squeeze(samples, 2)

                if x is not None:
                    marg_idx = layer.get_marginalization_idx()
                    keep_idx = [i for i in range(
                        self.args.num_var) if i not in marg_idx]
                    samples[:, keep_idx] = x[:, keep_idx]

                return samples

    def sample(self, num_samples=1, class_idx=0, x=None, **kwargs):
        return self.backtrack(num_samples=num_samples, class_idx=class_idx, x=x, mode='sample', **kwargs)

    def mpe(self, num_samples=1, class_idx=0, x=None, **kwargs):
        return self.backtrack(num_samples=num_samples, class_idx=class_idx, x=x, mode='argmax', **kwargs)

    def em_set_hyperparams(self, online_em_frequency, online_em_stepsize, purge=True):
        for l in self.einet_layers:
            l.em_set_hyperparams(online_em_frequency,
                                 online_em_stepsize, purge)

    def em_process_batch(self):
        for l in self.einet_layers:
            l.em_process_batch()

    def em_update(self):
        for l in self.einet_layers:
            l.em_update()

    def ccll(self, x_batch, img_width, img_height, marginal_window_width, marginal_window_height,
             patch_prob, sample=None,  grid_sampling=False, grid_prob=1.0, bisection_sampling=False, num_bin_bisections=1) -> tuple:
        """Compute the conditional composite log-likelihood (ccll) of the model for each image within x_batch given the rest of the pixels within the images
        with probability patch_prob. Otherwise, return the full log-likelihood. Set patch_prob to 1.0 to always compute the conditional composite log-likelihood.
        Can choose to sample a window for ccll training using uniform random, grid or bisection sampling.

        Args:
            x_batch (_type_): A batch of images.
            img_width (int): Input images widths.
            img_height (int): Input images heights.
            marginal_window_width (int): Window width for cll window.
            marginal_window_height (int): Window height for cll window.
            patch_prob (float): Probability of choosing a window for which to compute the conditional
                log-likelihood or just computing the full log-likelihood.
            sample (torch.Tensor, optional): A tensor of pixel indices for the sampled window we wished to compte
                 the ccll for. Defaults to None - where we randomly sample a window.
            grid_sampling (bool, optional): Whether to use grid sampling. Defaults to False.
            grid_prob (float, optional): Grid probablility for grid sampling. Defaults to 1.0.
            bisection_sampling (bool, optional): Whether to use bisection sampling. Defaults to False.
            num_bin_bisections (int, optional): Number of bisections to use for bisection sampling. Defaults to 1.

        Returns:
            tuple: A tuple containing the conditional log-likelihood (or the full log-likelihood) and the 
            sampled window (if patch_prob < 1.0).
        """
        # Compute the full log-likelihood of the batch.
        lls = self.forward(x_batch)
        ll = lls.sum()

        # If not passed patch window sample, then sample one.
        if sample is None:
            # Choose random window with probability self.args.patch_prob for which we compute the
            # conditional log-likelihood.
            if torch.rand((1,), device=self.args.device) <= patch_prob:
                if grid_sampling:
                    sample = grid_window_sampling(
                        img_width, img_height, marginal_window_width, marginal_window_height, grid_prob, device=self.args.device)
                elif bisection_sampling:
                    sample = bisection_sampling(
                        img_width, img_height, n_bisects=num_bin_bisections, device=self.args.device)
                else:
                    sample = uniform_random_sampling(
                        img_width, img_height, marginal_window_width, marginal_window_height, device=self.args.device)

                # Set marginalization indices to those of the sampled window.
                self.set_marginalization_idx(sample)
                marginal_lls = self.forward(x_batch)
                self.remove_marginalization_idx()
                # Compute ccll of the sampled window given the rest of the
                # pixels within the image.
                ccll = (lls - marginal_lls).sum()

            else:
                # Return the full log-likelihood if no window is chosen.
                # Bit of an abuse of notation here, but can think of this as cll with a patch window of size 0.
                ccll = lls.sum()
                ll = ccll
        else:
            # Otherwise, if we have passed a sample, then compute the ccll for that sample.
            # Set marginalization indices to those of the sampled window(s).
            self.set_marginalization_idx(sample)
            marginal_lls = self.forward(x_batch)
            self.remove_marginalization_idx()
            # Compute ccll of the sampled window given the rest of the
            # pixels within the image.
            ccll = (lls - marginal_lls).sum()

        return ccll, sample, ll

    def eval_avg_ll(self, x, batch_size=100) -> float:
        """Evaluate avg log-likelihood of model on given data x in a batched way (for
        efficient computation over larger datasets).

        Args:
            x (torch.tensor): Data for which we compute the log-likelihood of the model.
            batch_size (int, optional): Batch size for log-likeihood computations. Defaults to 100.
        Returns:
            float: Return log-likeihood of model for data x.
        """
        with torch.no_grad():
            # Generate indices for batches.
            idx_batches = torch.arange(
                0, x.shape[0], dtype=torch.int32, device=self.args.device).split(batch_size)
            ll_total = 0.0
            for _, idx in enumerate(idx_batches):
                batch_x = x[idx, :].to(self.args.device)
                # Compute log-likelihood for the batch and add to the total log-likelihood.
                ll_total += self.forward(batch_x).sum().item()

            return ll_total

    def avg_neg_ll_bpd(self, x, batch_size=100) -> float:
        """Evaluate the avg neg log-likelihood of model in bits per dimension on given data x
        in a batched way (for efficient computation over larger datasets).

        Args:
            x (torch.tensor): Data for which we compute the log-likelihood of the model.
            batch_size (int, optional): Batch size for log-likeihood computations. Defaults to 100.
        Returns:
            float: Return negative log-likeihood of model for data x measured in bits per dimension.
        """
        with torch.no_grad():
            # Generate indices for batches.
            idx_batches = torch.arange(
                0, x.shape[0], dtype=torch.int32, device=self.args.device).split(batch_size)
            ll_total = 0.0
            for _, idx in enumerate(idx_batches):
                batch = x[idx, :].to(self.args.device)
                # Compute log-likelihood for the batch and add to the total log-likelihood.
                ll_total += self.forward(batch).sum().item()

            # Convert average log-likelihood to bits per dimension.
            ll_total /= (x.shape[0] * x.shape[1] * np.log(2.0))

            return -ll_total

    def eval_test_avg_ccll_bpd(self, data_input_dir, dataset, img_width, img_height, patch_window_width, patch_window_height, samples_per_img) -> tuple:
        """Evaluate the avg CCLL of the model in bits per dimension over a given test set for a given patch window size.

        Args:
            data_input_dir (str): directory containing the test set for evaluation.
            dataset (Dataset): Dataset object for the test set.
            img_width (int): input images widths.
            img_height (int): input images heights.
            patch_window_width (int): test patch window width for ccll window.
            patch_window_height (int): test patch height for ccll window.
            samples_per_img (int): Number of samples to take for each image in the test set.

        Returns:
            tuple: A tuple containing the avg CCLL and its standard deviation in bits per dimension
            and a tracker containing the CCLL for each image in the test set stored as a list.
        """
        with torch.no_grad():
            # Load test set.
            _, _, x_test, _ = load_data(dataset, data_input_dir)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            x_test = torch.from_numpy(x_test).to(device)

            # First check if samples patch locations for each test set image already exist for this window setting saved as pickle files.
            # If so, load them.
            if os.path.exists(data_input_dir + f"test_set_samples/{patch_window_width}_{patch_window_height}.pkl"):
                print("Loading samples from file.")
                samples = pkl.load(
                    open(data_input_dir + f"test_set_samples/{patch_window_width}_{patch_window_height}.pkl", "rb"))

                ccll_tracker = []
                # Loop over all images in the test set.
                for idx in range(x_test.shape[0]):
                    if idx % 1000 == 0:
                        print("Processing image: ", idx)
                    for i in range(samples_per_img):
                        # Compute the CCLL for a randomly selected window for the image.
                        sample = torch.tensor(samples[idx][i])
                        ccll, sample, _ = self.conditional_composite_ll(
                            x_test[idx, :].unsqueeze(0), img_width, img_height, patch_window_width, patch_window_height, patch_prob=1.0, sample=sample)
                        # Convert to bits per dimension (scaling by number of pixels in patch instead of total number of pixels in image)
                        ccll_bpd = -ccll / (sample.shape[0] * np.log(2.0))
                        # Append data to tracker for this image.
                        ccll_tracker.append(ccll_bpd.item())
            else:
                # If the sample locations don't exist for this window setting, then sample them and save them to file.
                print("Sampling windows.")
                # Sample samples_per_img windows for each image in the test set.
                cll_tracker = []
                samples = []
                # Loop over all images in the test set repeating the sampling process for each image samples_per_img times.
                for idx in range(x_test.shape[0]):
                    if idx % 1000 == 0:
                        print("Processing image: ", idx)
                    img_samples = []
                    for _ in range(samples_per_img):
                        # Compute the CLL for a randomly selected window for the image.
                        ccll, sample, _ = self.ccll(
                            x_test[idx, :].unsqueeze(0), img_width, img_height, patch_window_width, patch_window_height, patch_prob=1.0)
                        # Convert to bits per dimension (scaling by number of pixels in patch instead of total number of pixels in image)
                        ccll_bpd = -ccll / (sample.shape[0] * np.log(2.0))
                        # Append data to tracker for this image.
                        sample = tuple(sample.cpu().numpy())
                        img_samples.append(sample)
                        cll_tracker.append(ccll_bpd.item())
                    samples.append(img_samples)

                # Check if directory exists, if not create it.
                if not os.path.exists(data_input_dir + "test_set_samples/"):
                    os.makedirs(data_input_dir + "test_set_samples/")

                # Save samples to file.
                with open(data_input_dir + f"test_set_samples/{patch_window_width}_{patch_window_height}.pkl", 'wb') as file:
                    pkl.dump(samples, file)

            # Compute average CLL in bpd over all images in the test set.
            avg_cll = np.mean(cll_tracker)

            # Calculate the standard deviation of the CLL over the test set using the tracker.
            cll_bpd_std = np.std(cll_tracker)

        return avg_cll, cll_bpd_std, cll_tracker


def uniform_random_sampling(img_width, img_height, marginal_window_width, marginal_window_height, device='cuda') -> torch.Tensor:
    """
    Uniformly sample a window (rectangular subset of pixels) from an image with dimensions img_width x img_height whose
    pixels are ordered linearly across rows Also can generate tensor of indices for a batch of images for test set evaluation.

    Args:
        img_width (int): width of the input image.
        img_height (int): height of the input image.
        marginal_window_width (int): width of the patch window to be sampled.
        marginal_window_height (int): height of the window of the patch window to be sampled.
        device (str, optional): Device. defaults to 'cuda'.

    Returns:
        torch.Tensor: A tensor of pixel indices for the randomly sampled window.
    """
    # If width and height of window are not equal, then randomly swap them to
    # ensure sampling can be both horizontal and vertical rectangles.
    if marginal_window_width != marginal_window_height:
        if torch.rand(1) > 0.5:
            marginal_window_width, marginal_window_height = marginal_window_height, marginal_window_width

    # Sample top left corner of window.
    row = img_width * \
        torch.randint(0, img_height - marginal_window_height,
                      (1,), device=device)
    top_left = torch.randint(
        0, img_width - marginal_window_width, (1,), device=device) + row

    # Sample rest of window relative to top left corner and the provided dimensions.
    sample = torch.tensor([j for i in range(marginal_window_height)
                           for j in range(top_left + i * img_width, top_left + i * img_width + marginal_window_width)], device=device)

    return sample


def grid_window_sampling(img_width, img_height, marginal_window_width, marginal_window_height, gamma=1.0, border=0, device='cuda') -> torch.Tensor:
    """Method of sampling a patch window for ccl training using grid patching. This splits an image into a grid of patches, which
    are then sampled with probability gamma. The inital top left corner of the grid is sampled uniformly from the top left
    corner of the image with patch dimensions around index (0,0), which allows for random translations of the grid.

    Args:
        img_width (_type_): input image width.
        img_height (_type_): input image height.
        marginal_window_width (_type_): grid patch width.
        marginal_window_height (_type_): grid patch height.
        gamma (float, optional): grid probability. Defaults to 0.8.
        device (str, optional): device for training. Defaults to 'cuda'.

    Returns:
        torch.Tensor: torch tensor of pixel indices for the sampled window.
    """

    # Sample top left pixel by sampling uniformly with patch dimentisons around index (0,0).
    top_left_pixel_x = torch.randint(-marginal_window_width+1,
                                     1, (1,), device=device).item()
    top_left_pixel_y = torch.randint(-marginal_window_height+1,
                                     1, (1,), device=device).item()

    # If width and height of window are not equal, then randomly swap them to
    # ensure sampling can be both horizontal and vertical rectangles.
    if marginal_window_width != marginal_window_height:
        if torch.rand(1) > 0.5:
            marginal_window_width, marginal_window_height = marginal_window_height, marginal_window_width

    # Grid patches
    grid_patches = []

    # Based on the top left corner of the grid, create list of possible patches that can be sampled.
    for i in range(top_left_pixel_x, img_width, marginal_window_width):
        for j in range(top_left_pixel_y, img_height, marginal_window_height):
            # If top left corner is outside of image, skip.
            if i < 0 or j < 0 or i + marginal_window_width > img_width or j + marginal_window_height > img_height:
                continue

            # Sample rest of window relative to top left corner and the provided dimensions.
            grid_patch = torch.tensor([k for x in range(marginal_window_width) for y in range(marginal_window_height)
                                       for k in range((i + x) * img_width + (j + y),
                                                      (i + x) * img_width + (j + y) + 1)], device=device)
            # Append to list of possible patches.
            grid_patches.append(grid_patch)

    # Select patches with probability gamma. To avoid the usually unlikely case of no patches being selected, we
    # resample until we have at least one patch.
    sampled_grid_patches = []
    while len(sampled_grid_patches) == 0:
        sampled_grid_patches = [
            patch for patch in grid_patches if np.random.uniform() < gamma]

    # Combine sampled patches into a overall patch.
    sample = torch.cat(sampled_grid_patches)

    return sample


def bisection_sampling(img_width, img_height, n_bisects, device='cuda') -> torch.Tensor:
    """Function that samples a patch window for ccl training using bisection sampling. This splits an image into a set of
    symmetric binary bisections, which are then sampled uniformly. The number of bisections is given by n_bisects.

    Args:
        img_width (int): width of the input image.
        img_height (int): height of the input image.
        n_bisects (int): number of symmetric binary bisections to make.
        device (str, optional): computation device. Defaults to 'cuda'.

    Returns:
        torch.Tensor: A tensor of indices of the sampled pixels from the selected bisected half.
    """

    # Calculate the centre of the image alongisde the number of bisctions and the angle of each bisecting line. Note
    # here we are using cartesian coordinates.
    center_x, center_y = (img_width-1) // 2, (img_height - 1) // 2
    n__bin_segments = 2 ** n_bisects
    segment_angle = np.pi / n__bin_segments
    indices_set = []

    # Create grid of indicies for the image.
    x, y = torch.meshgrid(torch.arange(img_width), torch.arange(img_height))
    x, y = x.to(device), y.to(device)

    # Loop through each bisecting line and append the pixels corresponding to the two halves of the image.
    for segment_index in range(int(np.pi // segment_angle)):
        upper_half_indices = []
        lower_half_indices = []
        theta = segment_angle * segment_index

        # Randomly decide which half contains the bisecting line.
        line_in_upper_half = torch.rand(1).item() < 0.5

        # If the angle is \pi/2 then the line is veritcal.
        if theta == np.pi / 2:
            line_x = center_x

            # Create binary mask for the upper half of the image with repect to the bisecting line.
            upper_half_mask = (x <= line_x) == line_in_upper_half

            # Gather the indices of the upper and lower half of the image.
            upper_half_indices = torch.arange(
                img_width * img_height, device=device)[upper_half_mask.view(-1)]
            lower_half_indices = torch.arange(
                img_width * img_height, device=device)[~upper_half_mask.view(-1)]
        else:
            # Else, return y-coordinates of the bisecting line given the x-coordinates.
            def line(x):
                return torch.tan(torch.tensor(theta)) * (x - center_x) + center_y

            # Create binary mask for the upper half of the image with repect to the bisecting line.
            upper_half_mask = (y <= line(x)) == line_in_upper_half

            # Gather the indices of the upper and lower half of the image.
            upper_half_indices = torch.arange(
                img_width * img_height, device=device)[upper_half_mask.view(-1)]
            lower_half_indices = torch.arange(
                img_width * img_height, device=device)[~upper_half_mask.view(-1)]

        # Append regions to the list of bisected halves.
        indices_set.append(upper_half_indices)
        indices_set.append(lower_half_indices)

    # Uniformally sample a bisected half of indicies from the list of all possible bisected halves.
    sample = np.random.randint(0, len(indices_set))

    return indices_set[sample]
