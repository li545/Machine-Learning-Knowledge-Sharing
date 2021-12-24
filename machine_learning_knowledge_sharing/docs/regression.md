#### OLS
OLS is Maximum Likelihood Estimator and thus inclined to be overfitting.

$$
w^{\ast} = argmax_{w \in \Theta}p(y_{1}, \cdots, y_{n}|x_{1}, \cdots, x_{n}, w)
$$

#### Ridge 
To solve overfitting problem of MLE, replace MLE with MAP (maximum a posteriori) estimator.

$$
w^{\ast} = argmax_{w \in \Theta}(w|x, y)
$$

Ridge regression is MAP with Gaussian prior, namely, 

$$
y|x, w \sim \mathcal{N}(w^{T}x, \sigma^{2}) \ and \ w \sim \mathcal{N}(0, \beta^{2}\mathcal{I})
$$
with $\mathcal{I}$ as identity matrics.

The gaussian prior is equavelent to L-2 norm regularization.

#### Lasso 

Lasso regression is MAP with Laplace prior, 

$$
y|x, w \sim \mathcal{N}(w^{T}x, \sigma^{2}) \ and \ w \sim Laplace(0, b)
$$

The laplace prior is equavelent to L-1 norm regularization.

#### Kernel Regression

$$
h(x) = \sum_{i=1}^{N}\alpha_{i}k(x_{i}, x)
$$

#### Model Evaluation

* $R^{2}$ measures percent of variance explained by model in target variable. 
* Adjusted-$R^{2}$ considers degree of freedom as $R^{2}$ always grows as more features are included in models, no matter newly added features improve models. Suppose a duplicate feature is added to original model, 
$$
Var(\beta_{1}X + \beta_{2}X) = Var((\beta_{1}+\beta_{2})X) = (\beta_{1}+\beta_{2})^{2}Var(X) > \beta_{1}^{2}Var(X)
$$
as $\beta_{1}$ and $\beta_{2}$ have the same signs, since correlation between $X$ and $Y$ does not change.
* Mean Absolute Error measures proportion of error in target values.
$$
\epsilon = \frac{\|y-\hat{y}\|}{y}
$$
* MSE should be applied for models built after data normalization as it is not scale-free.