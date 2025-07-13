# VAE-from-scratch
VAEs implemented from scratch in Pytorch


The VAE is a type of generative model. That means it is capable of generating synthetic examples of a particular type of thing. 
The idea is that somehow the model will have learned the distribution of the data and now it's able to sample from that distribution to create artificial examples or even things like anomaly detection or inpainting.

Assuming we have the true distribution of the data $p^{*}(x)$, we want to build some probabilistic model $p_{\theta}(x)$ (probabilistic model P, parameterized by $\theta$ ) that will approximate  $p^{*}(x)$ as closely as possible. For e.g if we have a dataset of faces, $p_{\theta}(x)$ should register a high values for faces and low values for images of non-faces. 

In statistical terms,  $p_{\theta}(x)$ is the likelihood of data x given the model parameters $\theta$. So what we're trying to do is to adjust $\theta$ to achieve the maximal likelihood of observing the data points actually included in our data set D. In other words, it is a case of Maximum Likelihood Estimation. 

$p^{*}(x) = p_{\theta}(x)$  where $p_{\theta}(x)$ = likelihood. 

## Latent Variables
Let us take an example, we want our model to learn the distribution of circles. We first assume that there are some underlying factors of variation used to generate "circles". We call the underlying factors [position, radius]. 
So using just these two variables, we can account for all the variation within the domain of circles and we'll call these underlying factors latent variable Z. 
They're call latent variables because we never see them. Despite not seeing them the structure of the data is captured by the latent variables. Instead of specifying which pixels are filled in and which pixels are not, we can just specify the latent variable and it captures all the information we need to know. 

This intuition lies behind why the number of latent variables is a lot lower than the number of data variables (which in the case of distribution of circles is the number of pixels)

The latent representation is in a sense a maximally efficient compression of the data. So what we have here is a representation learning problem. How do we find an efficient latent representation that captures the factors of variation inherent in the data. 

So now that we have our latent variables Z in addition to our data variables X, and these coexist with one another, what we have is a situation where for every data point there is a data representation and a latent representation. 

Further, we should be able to infer the latent variables from the data variables and conversely to generate the data variables given the latent variables where these occur via some conditional distribution. 

The importance of theta here is that it ensures that for example we generate images of circles instead of squares or triangles and inversely that we extract the underlying features of circles during inference instead of the underlying features of anything else. 

 $p_{\theta}(z,x) = p_{\theta}(z) * p_{\theta}(x|z)$  (from the chain rule of probability)
 $p_{\theta}(z,x)$ = Joint Distribution 
 $p_{\theta}(z) = \mathcal{N}(0, I)$  
 $p_{\theta}(x|z)$ = Conditional Distribution
We don't really have any prior knowledge of how Z should be distributed so in practice we just assume some distribution, typically N(0,I), the Gaussian Distribution. 

The VAE is encouraged to map the full data distribution D to a latent distribution P of Z so if we choose a unit gaussian as our P of Z then sampling from this unit gaussian will produce realistic synthetic examples in data space. 

## Intractability of the Marginal Likelihood

To optimize $\theta$ we have to maximize our original target P of X and this can only be found by integrating the joint probability distribution over all possible values of Z, in other words by marginalizing over Z. 

$p_{\theta}(x) = \int_{z}p_{\theta}(x|z)dz = \int_{z}p_{\theta}(z)*p_{\theta}(x|z)dz$ 

$p_{\theta}(x)$ = Marginal Likelihood of X. 

We want to optimize theta such that P(X) is high for data points that belong in our data set so ideally if we could calculate P of X given $\theta$ we could compute the gradient of this with respect to $\theta$ over a batch of X and then optimize $\theta$ by gradient descent. 
However the problem we encounter is that the integral is intractable, it is infeasible to compute it, both because there is no analytical solution and because it cannot be estimated efficiently. If we use numerical integration techniques, we require a number of samples that increases exponentially with the number of latent variables. 

So how do we optimize $\theta$? We word around this by realizing that we don't need to compute P of X itself, we just need some way of increasing it as much as we can for the examples in our data set and it does this by a Bayesian framework. 

## Bayes Rule

P(H|E) = $\frac{P(H)* P(E|H)}{P(E)}$ 
