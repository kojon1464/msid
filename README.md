The program has been created for MSiD laboratories.

# Introduction

The application implements classification algorithms that use data sets provided in [FashionMnist repository](https://github.com/zalandoresearch/fashion-mnist).
The data sets contains training and testing set of clothing images and corresponding labels indicating piece of garment.

Each image is a 28 by 28 grayscale picture. Each label is a number that cen be described as in table:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

As a result of running each classifier we should get a accuracy of label prediction on test set.
Because the repository is very popular we can compare performance of classifiers with many other algorithms and
different set o parameters.

# Methods

#### Naive Bayes

This is the method that I have chosen to adapt for fashionMIST classification. I do not use any
advanced features extraction methods but I do check if pixel from image is above given threshold.
I had to modify `model_selection_nb` method in `byes` module so that it returns not only calculated errors
and values of best hyperparameters `a` and `b` but also probability distribution. A priori probability can be
always calculated so it doesn't have to be returned.

Because `model_selection_nb` has to select best parameters part of training set 
is used as validation data. In my case it doesn't matter due to fact that my Implementation of
Naive Bayes classifier does not support batches and my computer runs out of memory.

This classifier finds apriori probability and probability of pixel (feature) being assigned to label.
The apriori probability means likelihood of choosing label without any evidence.

#### Neural Network

I have created simple neural network with the help of the package PyTorch. I have mainly base my knowledge on [PyTorch Tutorials](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

The network consist of 4 layers.
Each node in my network is a linear node. That means that node applies linear transformation to the input (`y= ax + b`).
First layer consist of 784 nodes. They are corresponding to each pixel on the image. Two inner layers consist of 32 nodes.
Last layer consist of 10 nodes each node. I'm using `log_softmax` function on outputs of last layer so that i get likelihood 
of each label being the result. As a loss function I'm using Average Stochastic Gradient Descend.

# Results

Table below shows accuracy result of classifiers described above.

| Name | Accuracy |
| --- | --- |
| Naive Bayes | 0.657 | 
| Neural Network (4 layers, log_softmax)| 0.836 |
| Neural Network (5 layers, log_softmax)| 0.824 |
| Neural Network (4 layers, softmax)| 0.559 |

Let's compare some entries from [fashionMNIST benchamrks](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/#)

| Name | Accuracy |
| --- | --- |
| LogisticRegression | 0.839 | 
| SVC| 0.836 |
| LinearSVC | 0.832 |
| SGDClassifier | 0.831 |

My neural network is placed somewhere in the middle in the table. Considering that most almost all of the algorithms
presented in the benchmark table were trained for longer than hour my algorithm lacks 6 percent points to the best one.

# Usage

The program requires Python interpreter to run. The interpreter has to have following packages installed:

* NumPy
* PyTorch
* Torchvision

You don't have to download any data sets because Torchvision acquires needed files.

To run Naive Bayes classification algorithm run `main_byes` module.

To run classification algorithm based on neural network run `main_net` module.

Both modules shouldn't take more than a minute to complete.