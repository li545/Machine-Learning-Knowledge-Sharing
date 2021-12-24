#### Bayesian

From Bayesian perspective, logistic regression can be viewed as a MLE with Bernoulli distribution<br>

$$
y|x, w \sim Bernoulli(\sigma(w^{T}x))
$$
with sigmoid function $\sigma(\cdot)$

$$
P(y|x, w) = (\sigma(w^{T}x))^{y}(1-\sigma(w^{T}x))^{(1-y)}
$$

$$
log[\frac{P(y=1|x, w)}{P(y=0|x, w)}] = w^{T}x
$$

With data $\mathcal{D}_{n} = \{(x_{1}, y_{1}), \cdots, (x_{n}, y_{n}) \}$ 
$$
\hat{w}^{MLE} = argmax_{w\in \Theta}\prod_{i=1}^{n}P(y_{i}|x_{i}, w) = argmin_{w\in \Theta} -\sum_{i}logP(y_{i}|x_{i}, w)
$$

#### Logistic Loss VS Cross Entropy

* Minimizing negative log likelihood with sigmoid likelihood (for binary classification) results in logistic loss.

$$
-logP(y_{i}|x_{i}, w) = -log\sigma(t_{i}) = -log\frac{1}{1+e^{-t_{i}}} = log(1+e^{-t_{i}})
$$ with $t_{i} = w^{T}x_{i}$

Hence, logistic loss is written as,

$$
l(z) = log(1+e^{-z})
$$

* Minimizing negative log likelihood with softmax likelihood (for multi-classfication) results in cross entropy.