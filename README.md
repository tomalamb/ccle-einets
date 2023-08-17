# Edinburgh MSc AI Dissertation - Self-Supervised Learning of Tractable Generative Models.

Code associated with my dissertation on looking into alternative methods of training tractable generative models. Specifically, we focus on Einsum Networks, which are a more general and efficient vectorised form of probablistic circuit (PC) and look into whether conditional composite log-likelihood estimation (CCLE) can act as a viable alternative for training EiNets over MLE training, which often leads to overfitting.

**Abstract**: *Einsum Networks (EiNets) are an efficient implementation of a general class of probabilistic models known as probabilistic circuits (PCs). These models have advantages over expressive generative models such as VAEs and GANs because they allow for exact and efficient probabilistic inference of various types. However, as PCs grow in the number of parameters, they become more challenging to train. In particular, they have been shown to be susceptible to ubiquitous problems in deep learning, such as overfitting when trained via maximum likelihood estimation (MLE). Motivated by these problems, we explore an alternative parameter learning method particularly applicable to EiNets known as conditional composite log-likelihood estimation (CCLE). We propose three methods of implementing CCLE for EiNets: uniform random sampling, bisection sampling and grid sampling. In our experiments on MNIST and F-MNIST, we observe that CCLE training shows promise as a valid alternative density and generative training scheme for EiNets to MLE and for providing greater inpainting capabilities. However, a CCLE objective shows mixed results as a form of regularisation during training. Moreover, we note that these findings depend on the CCLE method used, the sizes of the patches chosen for conditional training and the information density of the images within a dataset.*

Our code builds upon and utlises the existing codebase by Peharz *et al.* (2020) in their paper:

R. Peharz, S. Lang, A. Vergari, K. Stelzner, A. Molina, M. Trapp, G. Van den Broeck, K. Kersting, Z. Ghahramani,
**Einsum Networks: Fast and Scalable Learning of Tractable Probabilistic Circuits**,
*ICML 2020*.

The following is a link to their original repository: [Einsum Networks](https://github.com/cambridge-mlg/EinsumNetworks)

### Training EiNets via CCLE

#### Getting Started

In order to get started, install the required dependencies using the command

```
bash setup.sh
```

This should create a conda environment named einet_ccle, install the required python dependencies and then download the MNIST and F-MNIST datasets. Note that our code uses WandB for logging.


#### Training an EiNet Model
To train an EiNet model via CCLE or MLE, execute the following command 
```
python src/training.py --command_line_arguments
```
adding the the command line arguments as you need for training. 

Below we give three examples of how to train EiNets using SGD for MLE, and then using uniform random smapling, bisection sampling and grid sampling for CLLE training. The examples we include are models that we specifically investigated in this work.

   a. MLE trained model using SGD, $\text{SGD}$ model:


    python src/training.py -K 32 --max_num_epochs 64 --batch_size 100 --lr 0.01 --dataset f_mnist --patience 8 --pd_deltas 7,28

   
   a. MLE trained model using EM, $\text{EM}$ model:


    python src/training.py -K 32 --max_num_epochs 64 --batch_size 100 --online_em_stepsize 0.05 --dataset f_mnist --patience 8 --pd_deltas 7,28 --em
         

   c. Uniform random sampling, $\text{RAND}_4$ model:

    
    python src/training.py -K 32 --max_num_epochs 64 --batch_size 100 --ccle --lr 0.01 --dataset f_mnist --patience 8 --pd_deltas 7,28 --patch_size 4
    

   d. Bisection sampling, $\text{BIS}_{32}$ model:

    
    python src/training.py -K 32 --max_num_epochs 64 --batch_size 100 --ccle --lr 0.01 --dataset f_mnist --patience 8 --pd_deltas 7,28 --patch_size 8  --bisection_sampling --num_bin_bisections 5
    

   e. Grid sampling, $\text{GRID}_{4, \gamma = 0.8889}$ model:

    
    python src/training.py -K 32 --max_num_epochs 64 --batch_size 100 --ccle --lr 0.01 --dataset f_mnist --patience 8 --pd_deltas 7,28 --patch_size 8 --grid_sampling --grid_prob 0.8889
    

#### Test Evaluation
We take as an example, the uniform random sample model, trained using the above command. We now list three commands so that you can evaluate the test CCLL, $\text{FID}$ and $\text{FID}_{inp}$ scores for this model.

   a. Test CCL scores:


    python src/test_ccll_evaluation.py -K 32 --pd_deltas 7,28 --patch_size 4  --dataset f_mnist  --ccll_test
         

   b. FID scores:

    
    python src/test_ccll_evaluation.py -K 32 --pd_deltas 7,28 --patch_size 4  --dataset f_mnist  --fid
    

   c. Inpainted FID scores:

    
    python src/test_ccll_evaluation.py -K 32 --pd_deltas 7,28 --patch_size 4  --dataset f_mnist  --fid_inpaint
    






