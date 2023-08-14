"""
Code containing various utlity functions. Code builds upon the code from the original 
EiNets repository, which can be found at https://github.com/cambridge-mlg/EinsumNetworks.
"""

import numpy as np
import os
import torch
import errno
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from scipy.optimize import fsolve


def mkdir_p(path):
    """Linux mkdir -p"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def one_hot(x, K, dtype=torch.float):
    """One hot encoding"""
    with torch.no_grad():
        ind = torch.zeros(x.shape + (K,), dtype=dtype, device=x.device)
        ind.scatter_(-1, x.unsqueeze(-1), 1)
        return ind


def save_image_stack(samples, num_rows, num_columns, filename, margin=5, margin_gray_val=1., frame=0, frame_gray_val=0.0):
    """Save image stack in a tiled image"""

    # for gray scale, convert to rgb
    if len(samples.shape) == 3:
        samples = np.stack((samples,) * 3, -1)

    height = samples.shape[1]
    width = samples.shape[2]

    samples -= samples.min()
    samples /= samples.max()

    img = margin_gray_val * \
        np.ones((height*num_rows + (num_rows-1)*margin,
                width*num_columns + (num_columns-1)*margin, 3))
    for h in range(num_rows):
        for w in range(num_columns):
            img[h*(height+margin):h*(height+margin)+height, w*(width+margin):w*(width+margin)+width, :] = samples[h*num_columns + w, :]

    framed_img = frame_gray_val * \
        np.ones((img.shape[0] + 2*frame, img.shape[1] + 2*frame, 3))
    framed_img[frame:(frame+img.shape[0]), frame:(frame+img.shape[1]), :] = img

    img = Image.fromarray(np.round(framed_img * 255.).astype(np.uint8))

    img.save(filename)


def sample_matrix_categorical(p):
    """Sample many Categorical distributions represented as rows in a matrix."""
    with torch.no_grad():
        cp = torch.cumsum(p[:, 0:-1], -1)
        rand = torch.rand((cp.shape[0], 1), device=cp.device)
        rand_idx = torch.sum(rand > cp, -1).long()
        return rand_idx


def grid_prob_solver(img_height, img_width, patch_height, patch_width, num_windows, root_est=0.5) -> float:
    """Finds the grid probability, \gamma, that will result in the expected number of windows sampled
    being equal to num_windows when using grid sampling for CLLE training.

    Args:
        img_height (int): Height of the image.
        img_width (int): Width of the image.
        patch_height (int): Height of each grid patch.
        patch_width (int): Width of each grid patch.
        num_windows (int): Number of windows to sample desired to be sampled on average.
        root_est (float, optional): Initial estimate of the grid probability. Defaults to 0.5.

    Returns:
        float: The grid probability, \gamma.
    """

    def grid_num_exp_window_samples(gamma):
        """"
        Calculates the expected number of windows sampled when using grid sampling with grid
        probability gamma for MCCLE training.
        """

        exp = 0

        for j in range(-patch_width + 1, 1, 1):
            for i in range(-patch_height + 1, 1, 1):
                if i != 0 and j != 0:
                    exp += ((img_height - i - patch_height) // patch_height) * ((img_width -
                                                                                 j - patch_width) // patch_width) * gamma * 1/patch_width * 1/patch_height
                elif i != 0 and j == 0:
                    exp += ((img_height - i - patch_height) // patch_height) * \
                        ((img_width - j) // patch_width) * \
                        gamma * 1/patch_width * 1/patch_height
                elif i == 0 and j != 0:
                    exp += ((img_height - i) // patch_height) * ((img_width - j -
                                                                  patch_width) // patch_width) * gamma * 1/patch_width * 1/patch_height
                else:
                    exp += ((img_height - i) // patch_height) * ((img_width - j) //
                                                                 patch_width) * gamma * 1/patch_width * 1/patch_height
        return exp

    root = fsolve(lambda x: grid_num_exp_window_samples(
        x) - num_windows, root_est)

    return root


def sample_plotter():
    """
    Plot inpainted image samples of the different EiNet models for both the MNIST and F-MNIST datasets.
    """
    # Load latex for plotting model names.
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    # Model names whose samples we are plotting.
    models = ['reference', 'em', 'sgd', 'bis_32', 'rand_4', 'rand_8',
              'grid_4_1024', 'grid_4_6272', 'grid_8_1451', 'grid_8_8889']

    fig, axs = plt.subplots(12, len(models), figsize=(
        13, 16), gridspec_kw={'hspace': 0.1, 'wspace': 0.1})

    for ax in axs.flatten():
        ax.axis('off')

    # Add the MNSIT samples to the figure.
    mnist_directory = 'data/output/samples/mnist/partial'

    for i, model in enumerate(models):
        model_directory = os.path.join(mnist_directory, model)
        image_files = sorted(os.listdir(model_directory))[:6]
        for j, image_file in enumerate(image_files):
            img = mpimg.imread(os.path.join(model_directory, image_file))
            axs[j, i].imshow(img, cmap='gray', aspect='auto')

    # Add the F-MNIST samples to the figure.
    f_mnist_base_directory = 'data/output/samples/f_mnist/partial'

    for i, model in enumerate(models):
        model_directory = os.path.join(f_mnist_base_directory, model)
        image_files = sorted(os.listdir(model_directory))[:6]
        for j, image_file in enumerate(image_files):
            img = mpimg.imread(os.path.join(model_directory, image_file))
            axs[j+6, i].imshow(img, cmap='gray', aspect='auto')

    # Model names formatted in latex to be included as the column headings.
    headings = [r'$\textbf{Ref.}$', r'$\textbf{EM}$', r'$\textbf{SGD}$', r'$\textbf{BIS.}_{\boldsymbol{32}}$', r'$\textbf{RAND}_{\boldsymbol{4}}$', r'$\textbf{RAND}_{\boldsymbol{8}}$',
                r'$\textbf{GR.}_{\boldsymbol{4, 0.1024}}$', r'$\textbf{GR.}_{\boldsymbol{4, 0.6272}}$', r'$\textbf{GR.}_{\boldsymbol{8, 0.1451}}$', r'$\textbf{GR.}_{\boldsymbol{8, 0.8889}}$']

    # Set column headings to our model names.
    for ax, heading in zip(axs[0], headings):
        ax.set_title(heading, fontsize=14.5, fontweight='bold')

    # Add horizontal line dividing the MNIST and F-MNIST samples.
    fig.lines.append(Line2D([0.115, 0.91], [
                     0.5-0.005, 0.5-0.005], transform=fig.transFigure, color='black', linewidth=2))

    # Make directory for saving if it doesn't already exist.
    if not os.path.exists('data/plots'):
        os.makedirs('data/plots')

    plt.savefig('data/plots/full_samples_mnist.pdf',
                dpi=300, bbox_inches='tight')
    plt.show()


def overfitting_plot():
    """
    Creates plot measuring the degree of overfitting vs generalisation performance as measured by
    test LL for baseline MLE models and the different MCCLE models.
    """

    # Set up for latex use in axis labels.
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    # Manually entered the training, validation and test LL for the models we have investigated.
    datasets = ['mnist', 'f_mnist']
    train_ll_values = {
        'mnist': np.array([1.238, 1.116, 1.315, 1.240, 1.184, 1.151, 1.148, 1.144, 1.212, 1.197, 1.151, 1.162, 1.130]),
        'f_mnist': np.array([3.295, 3.190, 3.488, 3.372, 3.370, 3.197, 3.204, 3.200, 3.308, 3.260, 3.216, 3.216, 3.202])
    }
    valid_ll_values = {
        'mnist': np.array([1.264, 1.214, 1.362, 1.314, 1.286, 1.227, 1.222, 1.219, 1.269, 1.271, 1.231, 1.238, 1.213]),
        'f_mnist': np.array([3.344, 3.324, 3.556, 3.496, 3.517, 3.335, 3.326, 3.333, 3.397, 3.371, 3.340, 3.349, 3.332])
    }
    test_ll_values = {
        'mnist': np.array([1.248, 1.195, 1.346, 1.298, 1.269, 1.211, 1.207, 1.203, 1.252, 1.256, 1.215, 1.222, 1.197]),
        'f_mnist': np.array([3.348, 3.329, 3.559, 3.500, 3.520, 3.339, 3.330, 3.337, 3.400, 3.375, 3.344, 3.352, 3.335])
    }

    # Manually have chosen 13 different color values, each one representing a different model.
    color_schm = ['#e6194B', '#3cb44b', '#000000', '#000080', '#f58231', '#FFD700',
                  '#008B8B', '#9400D3', '#696969', '#006400', '#8B4513', '#483D8B', '#BC8F8F']

    # Names of the different models that we investigated in our work.
    models = [r'$\text{EM}$', r'$\text{SGD}$', r'$\text{RAND}_{4}$', r'$\text{RAND}_{8}$', r'$\text{RAND}_{16}$'
              , r'$\text{BIS}_{n_{\text{bis}}=2}$', r'$\text{BIS}_{n_{\text{bis}}=8}$', r'$\text{BIS}_{n_{\text{bis}}=32}$',
              r'$\text{GRID}_{4, \gamma = 0.0256}$', r'$\text{GRID}_{4, \gamma = 0.1024}$', r'$\text{GRID}_{4, \gamma = 0.6272}$', 
              r'$\text{GRID}_{8, \gamma = 0.1451}$', r'$\text{GRID}_{8, \gamma = 0.8889}$']

    fig, axs = plt.subplots(1, 2, figsize=(15, 12), dpi=300)

    for ax, dataset in zip(axs, datasets):
        # Plot degree of overfitting on the horizontal axis.
        x_values = (
            valid_ll_values[dataset] - train_ll_values[dataset]) / valid_ll_values[dataset]

        # Plot degree of overfitting against test LL for each model
        for i, model in enumerate(models):
            ax.scatter(x_values[i], test_ll_values[dataset][i], label=model,
                       marker='x', s=100, c=color_schm[i], zorder=3, linewidth=2)

        # Set subplot title depending on the which dataset values we are plotting.
        if dataset == 'mnist':
            ax.set_title('MNIST', fontsize=20, fontweight='bold')
        else:
            ax.set_title('F-MNIST', fontsize=20, fontweight='bold')

        ax.grid(which='both')
        ax.minorticks_on()
        ax.set_facecolor('whitesmoke')
        ax.tick_params(axis='both', which='major', labelsize=18)

    axs[0].set_ylim([1.18, 1.5])
    axs[1].set_ylim([3.32, 3.6])

    # Add axis titles and figure title based on description form our work.
    fig.text(0.5, 0.06, 'Degree of Overfitting, ' +  # Adjust the second parameter
             r'$\mathcal{O}(\boldsymbol{\theta}_M)$', ha='center', fontsize=22, fontweight='bold')

    fig.text(0.068, 0.5, r'$-\ell_{LL} \left(\boldsymbol{\theta}_M ; \mathcal{D}_{\text{test}} \right)$',
             va='center', rotation='vertical', fontsize=22, fontweight='bold')

    fig.suptitle(
        r'$\textbf{Degree of Overfitting vs Generalisation Performance}$',
        fontsize=22, fontweight='bold', y=0.94)  # Adjust y value

    # Plot a legend showing which model corresponds to which color.
    fig.legend(models, loc='upper right', bbox_to_anchor=(0.49, 0.88), fontsize=18,
               title_fontsize='20', title='Model ' + r'$(M)$', handlelength=0.5, markerfirst=False, ncol=2)

    # Change the position of the x-axis label
    plt.subplots_adjust(wspace=0.15)

    # Make directory for saving if it doesn't already exist.
    if not os.path.exists('data/plots'):
        os.makedirs('data/plots')

    # Save figure.
    plt.savefig('data/plots/overfitting.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    overfitting_plot()
