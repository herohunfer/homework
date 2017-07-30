# MXNET week3 - how to do a deep learning project 

## Step 1: Know your goal 

    Supervised or Unsupervised
    Classification, Regression or both 
    Computer Vision or NLP 
    ...

### Here we give the example of a supervised, computer vision related, classification problem.
    
    We want to build the state-of-art image classifier
    
## Step 2: Get to know the dataset, figure out the data IO and prepare the data
    
    In pratical, we just don't have the data given certain problem. 
    
    It usually takes >50% time to get/clean/label/annotate data in real work.
    
    Here we assume if we learn a not-easy dataset, named cifar10, well, then we have the state-of-art model.
   
    "Cifar10" is a dataset much harder than MNIST, small images with 10 possible labels, including frog, bird...
    
## Step 3 (not hard): Find a pre-trained model of this problem or similar tasks(transfer learing)
   
    There are a lot of well-trained model available online. 
   
    Feel free to download it and use it for your own projects.
    
    Some of the models are trained by hundreds of GPUs in severy months.
    
    The last thing you should do is to build the model from scratch by yourself.
   
## Step 3 (hard): Build a neural network from scratch based what you learn/read so far

    By given problem, we decide to use a CNN, RNN, GAN, AutoEncoder or a joint model for our tasks.
    E.g. cifar10 classification 
    1. Any data preprocession you need including: normalization or/and augmentation, RGB or grayscale
    2. Network Architecture including: depth, layer type (convolution, FC, BN, dropout, NIN), width
    3. Loss function design, cross-entropy or regression, weight decay
    4. Validate the network with a random inpiut (what is the expected output before our actual training)

## Step 4: Build/Use the training script 

    Training, Validation, Testing dataset split
    Optimization method: SGD, momentum, Adam, rmsprop ... 
    Learning epoch
    Learning rate, Learning Rate Scheduler
    Drop out rate
    Early stopping rules  

## Step 5: Overfit small dataset, 100% precision and recall over a very small dataset

    To validate the capability & correctness, it is necessary to do a sanity check before full-scale training. 
    If you can not overfit a verys small dataset, go back and check every step before. 

## Step 6: Full-scale training on the full training dataset

    1. Random search on hyperparameters:
        e.g. 10 random setting in reasonable range: lr ~ {0.1, 0.001}, dropout ~ {0.0 ~0.5} ...
        Run each setting on training dataset for 1 or less epoch. 
        
    2. Pick the best setting, train the network to be converged 
    
    3. Test on Testing dataset and wild testing dataset

### Deploy the model to your architecture/business 

Try to read im2rec.py for your own dataset.
   
