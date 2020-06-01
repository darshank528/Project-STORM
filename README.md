# Project STORM-(STOchastic Recursive Momentum)

## Introduction 

Variance reduction has emerged in recent years as a strong competitor to stochastic gradient descent in non-convex problems, providing the first algorithms to improve upon the converge rate of stochastic gradient descent for finding first-order critical points. However, variance reduction techniques typically require carefully tuned learning rates and willingness to use excessively large mega-batches in order to achieve their improved results. This [paper](https://arxiv.org/abs/1905.10018) here presents a new algorithm, **Storm**, that does not require any batches and makes use of adaptive learning rates, enabling **simpler implementation and less hyperparameter tuning**. The technique for removing the batches uses a variant of momentum to achieve variance reduction in non-convex optimization. 

I am writing this summary to help my understanding,as well as anyone who might read this,along with my implementation too. The link to the implementation can be found [here](https://github.com/darshank528/Project-STORM).

## Objective

This paper addresses the **classic stochastic optimization algorithm** found in most of the problems in machine learning which require to achieve convergence of a non-convex function to its critical points. In past, many attempts have been carried out to optimize the convergence with variance reduction problem, and we have with us some amazing variance reduction algorithms like SVRG, SARAH, iSARAH, SAGA, ASGD and many more. With time and technology, sufficient efforts have been put on by the people to improve the convergence rate of critical points in case of non-convex SGD and they have achieved some success. However, despite this improvement, these algorithms have not seen as much success in practice in non-convex machine learning problems. Many reasons may contribute to this phenomenon, but two potential issues this paper addresses are **use of non-adaptive learning rates** and **reliance on giant batch sizes** to construct variance reduced gradients through the use of low-noise gradients calculated at a checkpoint. In particular, for non-convex losses these algorithms typically involve carefully selecting learning rates and the frequency of update of the checkpoint points. The optimal settings balance various unknown problem parameters exactly in order to obtain improved performance, making it especially important, and especially difficult, to tune them.

So this paper here, addresses both of these issues. It presents a new algorithm called **STOchastic Recursive Momentum (STORM)** that achieves variance reduction through the use of a variant of the momentum term, similar to the popular RMSProp or Adam momentum heuristics. Hence, this algorithm here does not require a gigantic batch to compute checkpoint gradients, in fact, it does not require any batches at all because it never needs to compute a checkpoint gradient. Storm achieves the **optimal convergence rate** and it uses an **adaptive learning rate schedule** that will automatically adjust to the variance values of the gradients. Overall, the algorithm is a significant qualitative departure from the usual paradigm for variance reduction, and this analysis may provide insight into the value of momentum in non-convex optimization. 

## Intuition behind the algorithm

Before describing the algorithm in details, we briefly explore the connection between SGD with momentum and variance reduction.
The stochastic gradient descent with momentum algorithm is typically implemented as

![](https://github.com/darshank528/Project-STORM/blob/master/Images/SGD%20with%20Momentum.png)

where a is small, i.e. a = 0.1. In words, instead of using the current gradient in the update of xt, we use an exponential average of the past observed gradients.

However, this paper takes a different route. Instead of showing that momentum in SGD works not so great in giving accelerated rates, it shows that a variant of momentum can provably reduce the variance of the gradients. In its simplest form, the variant proposed is:

![](https://github.com/darshank528/Project-STORM/blob/master/Images/SGD%20with%20updated%20Momentum.png)

The only difference is the addition of second term to the update part. However, this update does not require to use the gradients calculated at any checkpoint points. Note that if x(t)~x(t-1), then the update becomes approximately the momentum one. These two terms will be similar as long as the algorithm is actually converging to some point, and **so we can expect the algorithm to behave exactly like the classic momentum SGD towards the end of the optimization process**.

## Implementation with PyTorch on ResNet-32 model architecture

|![](https://github.com/darshank528/Project-STORM/blob/master/Images/Experiments.png)|
|:---:| 
|  **RESULTS ON CIFAR-10** |

In order to confirm that the update as per the paper do indeed yield an algorithm that performs well and requires little tuning, we implemented STORM in PyTorch and tested its performance on the **CIFAR-10 image recognition benchmark** using a **ResNet-32 model**, also implemented in Pytorch as implemented in this [paper](https://arxiv.org/abs/1512.03385).

We compared Storm to AdaGrad and Adam, which are both very popular and successful optimization algorithms. The learning rates for AdaGrad and Adam were swept over a logarithmically spaced grid. For Storm, we set w=k=0.1 as a default and swept c over a logarithmically spaced grid, so that all algorithms involved only one parameter to tune. No regularization was employed. We record train loss (cross-entropy), and accuracy on both the train and test sets.

#### Implementation Details:

- Implemented ResNet-32 network in Pytorch,both basic module and bottleneck module([paper](https://arxiv.org/abs/1512.03385)).
- Trained the model on Resnet32-bottleneck model as we found it to be more robust and fast as compared to basic model. 
- Implemented the STORM optimizer using the algorithm given in this [paper](https://arxiv.org/abs/1905.10018).
- Used learning rate schedulers for other two optimizers(Adam and Adagrad).
- No.of epochs(full passes through Cifar10 dataset):80
- Learning rate was swept over {0.01,0.1} logarithmically(for Adam and Adagrad) and we found lr=0.1 to give better results.
- c in STORM optimizer was swept over {10,100,1000}. However c=100 gave the best train and test accuracy among the 3 values.
- This table sums up the results of Implementation :

| Optimizer | Test Accuracy | Train Accuracy |
|:----------:|:-------------:|:--------------:|
|    Adam   |     71.44%    |       89.8%    |
| AdaGrad |    68.22%    |       92.3%      |
|    Storm   |     69.71%    |       94.8%     |

## Conclusion

These results show that, while Storm is only **marginally better** than AdaGrad on test accuracy and not as better as Adam, though on both training loss and training accuracy Storm appears to be somewhat faster in terms of number of iterations.

## Inference and scope of improvement:
- We get to learn about a new variance-reduction based algorithm, STORM, that finds critical points in stochastic, smooth, non-convex problems. This algorithm improves upon prior algorithms by virtue of removing the need for checkpoint gradients, and incorporating adaptive learning rates. These improvements mean that Storm is substantially easier to tune as compared to other optimizers.
- This algorithm enjoys the same robustness to learning rate tuning as popular algorithms like AdaGrad or Adam. Storm obtains the optimal convergence guarantee, adapting to the level of noise in the problem without knowledge of this parameter. 
- Storm indeed seems to be optimizing the objective in fewer iterations than baseline algorithms(verified on CIFAR-10 with a ResNet-32 architecture).
- Storm's update formula is strikingly similar to the standard SGD with momentum heuristic employed in practice.
- **SCOPE:** We note that the convergence proof given in the paper actually only applies to the training loss(since we are making multiple passes over the dataset). No **regularization** was employed in the above algorithm. It is possible that appropriate regularization can trade-off Storm's better training loss performance to obtain better test performance. Another thing we can say is that we have trained our model here only on 80 epochs as compared to 100k-300k epochs in the paper. So training longer could give us better training and test accuracy of our model.


