#### Gaussian Naive Bayes Classifier

Assumption: suppose data $\mathcal{D}_{n} = \{(x_{1}, y_{1}), \cdots, (x_{n}, y_{n}) \}$ and $x_{i} = (x_{i1}, \cdots, x_{id})' \in \mathbb{R}^{d}$, then

$$
\hat{P}(x_{i}|y) \sim \mathcal{N}(x_{i};\hat{\mu}_{y, i}, \sigma_{y, i}^{2})
$$

for $i \in \{1, 2, \cdots, d\}$. Namely, features are conditionally independent given class labels and following Gaussian distributions.

Consequently, the optimal prediction on $y$ is,

$$
y^{\ast} = argmax_{y'}P(y'|x_{1}, \cdots, x_{d}) = argmax_{y'}P(y')\prod_{i=1}^{d}P(x_{i}|y')
$$ since 

$$
P(x_{1}, \cdots, x_{d}|y') = \prod_{i=1}^{d}P(x_{i}|y')
$$

* Question: Is decision boundary of GNBC linear?

Yes, if and only if applying GNBC in binary classfication and variance-covariance matrix of features does not depend on class lables, i.e., features $X = (x_{1}, \cdots, x_{d})'$ has constant variance.

$$
X|Y=0 \sim \mathcal{N}(\mu_{X|Y=0}, \Sigma_{Y=0})
$$

$$
X|Y=1 \sim \mathcal{N}(\mu_{X|Y=1}, \Sigma_{Y=1})
$$

and $\Sigma_{Y=0} = \Sigma_{Y=1}$ is constant.

* Question: Gaussian NBC VS Logistic Regression?

In Gaussian NBC, predicted class label in binary classification,

$$
y = sign(\underbrace{log\frac{P(y=1|X)}{P(y=0|X)}}_\text{$f(x)$})
$$ 

with $f(x)$ called discriminant function. The Gaussian NBC is equivablent to logistic regression if 

i. Two classes share the same feature variance.  
ii. $f(x)$ is linear, i.e., $f(x) = w^{T}x$

But decision boundary might be different, as models have different assumptions.

* Question: What will happen if the assumption of conditional independent features does not hold?

Suppose all features are duplicated in a binary classification problem, i.e.,

$$
x_{1} = x_{2} = \cdots = x_{d}
$$

then discriminant function

\begin{equation}
\begin{split}
f(x) & = log\frac{P(y=1|x_{1}, \cdots, x_{d})}{P(y=0|x_{1}, \cdots, x_{d})}  \\
& = log\frac{P(y=1)\prod_{i=1}^{d}P(x_{i}|y=1)}{P(y=0)\prod_{i=1}^{d}P(x_{i}|y=0)} \\
& = log\frac{P(y=1)[P(x_{1}|y=1)]^{d}}{P(y=0)[P(x_{1}|y=0)]^{d}} \\
& = log\frac{P(y=1)P(x_{1}|y=1)[P(x_{1}|y=1)]^{d-1}}{P(y=0)P(x_{1}|y=0)[P(x_{1}|y=0)]^{d-1}} \\
& = log\frac{P(y=1|x_{1})[P(x_{1}|y=1)]^{d-1}}{P(y=0|x_{1})[P(x_{1}|y=0)]^{d-1}} \\
& = \underbrace{\underbrace{log\frac{P(y=1|x_{1})}{P(y=0|x_{1})}}_\text{$f_{1}(x)$} + (d-1)log\frac{P(x_{1}|y=1)}{P(x_{1}|y=0)}}_\text{$f_{d}(x)$}
\end{split}
\end{equation}

Since 

$$
P(y=1|x) = \frac{1}{1+e^{-f(x)}} = \sigma(f(x))
$$

\begin{equation}
\begin{split}
\sigma(f_{d}(x)) & = \frac{1}{1+e^{-f_{d}(x)}} \\ 
& = \frac{1}{1+e^{-f_{1}(x)}e^{-(d-1)log\mu}}
\end{split}
\end{equation}
with $\mu = \frac{P(x_{1}|y=1)}{P(x_{1}|y=0)}$. 

* $\mu > 1$, i.e., $log\mu > 0$ 

$P(y=1|x)$ will grow quickly towards 1 as $d$ increases. As a result, the model will generate a lot of false positives, as it predicts almost every case as positive.

* $\mu < 1$, i.e., $log\mu < 0$ 

$P(y=1|x)$ will grow quickly towards 0 as $d$ increases. As a result, the model will generate a lot of false negatives, as it predicts almost every case as negative.

* $\mu = 1$, i.e., $log\mu = 0$ 

$\sigma(f_{1}(x)) = \sigma(f_{d}(x))$

#### Gaussian Bayes Classifier

Assumption

$$
P(x|y) \sim \mathcal{N}(x;\mu_{y}, \Sigma_{y})
$$

with $\Sigma_{y}$ could be any positive semi-definite matrix. In GNBC, $\Sigma_{y}$ is diagnal because of conditional independence. When all classes share a constant feature variance, GBC is equivalent to Linear Discriminant Analysis.

#### Gaussian Mixtures

Idea

$$
p(x) = \sum_{y}p(x, y) = \sum_{y}p(x|y)p(y) = \sum_{y}p(y)\mathcal{N}(x;\mu_{y}, \Sigma_{y})
$$

In unsupervised learning, with $\mathcal{D} = \{x_{1}, \cdots, x_{n} \}$ in $K$ clusters, 

$$
P_{\theta}(\mathcal{D}) = \prod_{i=1}^{n}\sum_{j=1}^{K}w_{j}P(x_{i}|\theta_{j}) = \prod_{i=1}^{n}\sum_{j=1}^{K}w_{j}\mathcal{N}(x_{i};\mu_{j}, \Sigma_{j})
$$ where $w_{j}$ is likelihood of $j$-th cluster.

##### Hard EM

For iteration $t = 1, 2, \cdots$

i. Predict the most likely class for each observation $x_{i}$ with

$$
z_{i}^{(t)} = argmax_{z}P(z|x_{i};\theta^{(t-1)}) = \underbrace{argmax_{z}P(z)\mathcal{N}(x_{i}|\mu_{z_{i}}^{(t-1)}, \Sigma_{z_{i}}^{(t-1)})}_\text{Gaussian Bayes Classifier}
$$

ii. With complete data $\mathcal{D}^{(t)} = \{(x_{1},z_{1}^{(t)}), \cdots, (x_{n}, z_{n}^{(t)}) \}$, compute MLE for Gaussian Bayes Classifer with

$$
\theta^{(t)} = argmax_{\theta}P(\mathcal{D}^{(t)}|\theta^{(t-1)})
$$

* Question: K-Means VS Hard-EM?

K-Means is a special case of hard-EM when 

$w_{1} = w_{2} = \cdots = w_{K} = \frac{1}{K}$ and $\Sigma_{1} = \Sigma_{2} = \cdots = \Sigma_{K} = \mathcal{I}$ because

distance between any data $x_{i}$ and any cluster center $\mu_{j}$ in K-Means is defined by $L_{2}$-norm, i.e., 

$$
d(x_{i}, \mu_{j}) = \lVert x_{i}-\mu_{j} \rVert^{2}_{2} = (x_{i}-\mu_{j})^{T}(x_{i} - \mu_{j})
$$

The distance can be written as a Mahalanobis-distance, 

$$
d(x_{i}, \mu_{j}) = (x_{i}-\mu_{j})^{T}\Sigma^{-1}(x_{i} - \mu_{j})
$$
with $\Sigma = I$

Probability density of any data $x_{i}$ with $k$-dimensional Multivariate Gaussian distribution is determined by Mahalanobis-distance, 

$$
p(x_{i}|\mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^{k}| \Sigma|}}e^{(x_{i}-\mu)^{T}\Sigma^{-1}(x_{i} - \mu)}
$$

##### Soft EM

Membership likelihood of class $j$ given $x$,

$$
\gamma_{j}(x) = P(z=j|x, \underbrace{\mu, \Sigma, w}_\text{$\theta$})
$$

For each iteration,

i. E-step: compute $\gamma_{j}(x)$

$$
\mathcal{D}^{(t)} = \{(x_{1}, z_{1}^{(t)}), \cdots, (x_{n}, z_{n}^{(t)}) \}
$$

ii. M-step: compute MLE on $\theta$

* Question: Does EM converge to local optimum or global optimum for guassian mixtures?

Local optimum. Performance of EM for guassian mixtures is highly dependent on initializtion. As a rule of thumb, $\theta$ can be initialized with,

i. $w_{1} = \cdots = w_{K} = \frac{1}{K}$

ii. Initialize $\mu_{1}, \cdots, \mu_{K}$ by K-Means++.

iii. $\Sigma_{1} = \cdots = \Sigma_{K} = \frac{1}{n}\sum_{i}(X_{i}-\bar{X})(X_{i}-\bar{X})'$

##### Semi-supervised Learning

* Infer class membership for unlabelled data with 

$$
\gamma_{j}(x_{i}) = P(z_{i} = j|x_{i}, \theta) = \frac{w_{j}p(x_{i}|z_{i}=j, \theta)}{\sum_{k}w_{k}p(x_{i}|z_{i}=k,\theta)}
$$

* Update MLE on $\theta$ until algorithm converges.

##### Constrained Gaussian Mixtures

* Full variance-covariance matrix. No contraints on $\Sigma_{1}, \cdots, \Sigma_{K}$. 
* Spherical variance-covariance matrix. $\Sigma_{j} = \sigma_{j}^{2}\mathcal{I}$, i.e., $\sigma_{1}^{2} = \cdots = \sigma_{d}^{2}$. Number of parameters is $K$.

$$
\Sigma_{j} = \begin{pmatrix} \sigma_{j}^{2} \ 0 \ \cdots \ 0 \\ 0 \ \sigma_{j}^{2} \ \cdots \ 0 \\ \cdots \\ 0 \ 0 \ \cdots \ \sigma_{j}^{2} \end{pmatrix}_{d\times d}
$$

* Diagonal variance-covariance matrix (e.g., Gaussian Naive Bayes Classifier). Number of parameters is $Kd$.

$$
\Sigma_{j} = \begin{pmatrix} \sigma_{j, 1}^{2} \ 0 \ \cdots \ 0 \\ 0 \ \sigma_{j, 2}^{2} \ \cdots \ 0 \\ \cdots \\ 0 \ 0 \ \cdots \ \sigma_{j, d}^{2} \end{pmatrix}_{d\times d}
$$

* Tied variance-covariance matrix (e.g., Linear Discriminant Analysis). Number of parameters is $K\frac{d(d+1)}{2}$.
$$
\Sigma_{1} = \cdots = \Sigma_{K}
$$

* Question: How to select K?

By cross-validation. Picking K such that log-likelihood of validation set is maximized.

##### Degeneration of Gaussian Mixture Models

Suppose fit a GMM with only 1 data point. The model will overfit as $\sigma^{2} \rightarrow 0$.

Solution:

i. applying a wishart prior on $\sigma$. Wishart distribution is a multivariate generalization of Gamma distribution.

ii. adding a small quantity $\nu$ to $\Sigma_{j}^{(t)}$, i.e., 

$$
\Sigma_{j}^{(t)} \leftarrow \Sigma_{j}^{(t)\ast} + \nu^{2}\mathcal{I}
$$

iii. Replacing MLE with MAP on all model parameters, as MLE tends to overfit.