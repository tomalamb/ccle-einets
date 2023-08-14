# Edinburgh MSc AI Dissertation - Alternative Methods of Unsupervised Learning of Tractable Generative Models.

Code associated with my dissertation on looking into alternative methods of training tractable generative models trainined. Specifically, we focus on Einsum Networks, which are a more general and efficient vectorised form of probablistic circuits (PC) and look into whether maximum conditional composite log-likelihood estimation (MCCLE) can act as a viable altertive for training EiNets over MLE training.

**Abstract**: *Einsum Networks (EiNets) are an efficient implementation of a general class of probabilistic models known as probabilistic circuits (PCs). These models have advantages over expressive generative models such as VAEs and GANs due to their ability to allow for exact and efficient probabilistic inference. However, as PCs grow in the number of parameters, they become more challenging to train and have been shown to be susceptible to ubiquitous problems in deep learning such as overfitting when trained via maximum likelihood estimation (MLE). Motivated by these problems when using MLE, we explore an alternative parameter learning method which is particularly applicable to EiNets known as maximum conditional composite log-likelihood estimation (MCCLE). We propose three methods of implementing MCCLE for EiNets: uniform random sampling, bisection sampling and grid sampling. In our experiments on MNIST and F-MNIST, we observe that MCCLE training shows promise as a valid alternative density estimation scheme over MLE, acting as a form of regularisation during training and for providing greater inpainting capabilities. However, we note that these findings are dependent on the MCCLE method used, the sizes of the patches chosen for conditional training and the information density of the datasets that they are trained on.*

Code builds upon and utlises the existing codebase by Peharz *et al.* (2020) in their paper:

R. Peharz, S. Lang, A. Vergari, K. Stelzner, A. Molina, M. Trapp, G. Van den Broeck, K. Kersting, Z. Ghahramani,
**Einsum Networks: Fast and Scalable Learning of Tractable Probabilistic Circuits**,
*ICML 2020*.

The following is a link to their original repository: [Einsum Networks](https://github.com/cambridge-mlg/EinsumNetworks)

### Training EiNets via CCLE

#### Preparing the Datasets

First, run the following command to download the MNIST and F-MNIST datasets needed for training:

```
python src/datasets.py 
```

#### Training an EiNet Model
To train an EiNet model via CCLE or MLE, execute the following command 
```
python src/test_ccll_evaluation.py --command_line_arguments
```
adding the the command line arguments you need for training. Below we give three examples of how to train EiNets using unfirom random smapling, bisection sampling and grid sampling respectively for CLLE training that we specifically investigated in this work:
```
python src/training.py -K 32 --max_num_epochs 64 --batch_size 100 --ccle --lr 0.01 --dataset f_mnist --patience 8 --pd_deltas 7,28 --patch_size 8  # RAND_4 model
python src/training.py -K 32 --max_num_epochs 64 --batch_size 100 --ccle --lr 0.01 --dataset f_mnist --patience 8 --pd_deltas 7,28 --patch_size 8  --bisection_sampling --num_bin_bisections 5  #BIS_{32} model
python src/training.py -K 32 --max_num_epochs 64 --batch_size 100 --ccle --lr 0.01 --dataset f_mnist --patience 8 --pd_deltas 7,28 --patch_size 8 --grid_sampling --grid_prob 0.8889  #GRID_{4, \gamma = 0.8889} model
```

