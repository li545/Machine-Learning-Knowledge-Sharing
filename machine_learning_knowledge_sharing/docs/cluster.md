#### K-Means

Define a loss on $\mathcal{D}_{n} = \{x_{1}, \cdots, x_{n} \}$ with a set of clusters centered at $\mu_{1}, \mu_{2}, \cdots, \mu_{k}$ as <br>

$$
R(\mu) = R(\mu_{1}, \mu_{2}, \cdots, \mu_{k}) = \sum_{i=1}^{n}min_{j\in \{1,\cdots, k\}} \lVert x_{i}-\mu_{j} \rVert^{2}_{2}
$$

* K-Means converges to local optimum.
* High time complexity
* Not able to model arbitrary-shaped clusters.
* Performance heavily depends on cluster initialization.
* Problems with uniform seeding
    * oversample large clusters
    * undersample small clusters

#### Seeding With K-Means++
To avoid sample all/most centers from one cluster, apply seedings with the K-Means++.<br>

$\mathbf{Step 1}$: Picking the first data point uniformly as the first cluster center $\mu_{1}$.<br>
$\mathbf{Step 2}$: For $j = 2, \cdots, k$, selecting data point $x$ as the next cluster center with probability 
$$
p \propto d^{2}(x; \mu_{1}, \cdots, \mu_{j-1}) = min_{l\in \{1, \cdots, j-1 \}}\lVert x - \mu_{l} \rVert_{2}^{2}
$$

#### Selecting $K$
* Elbow method based on a heuristic quality measure <br>
$$
Q = \sum_{i=1}^{n}\lVert x_{i} - \mu_{j} \rVert^{2}_{2}
$$
where $x_{i}$ belongs to the cluster $\mu_{j}$
* Regularization: adding penalty for complexity<br>
$$
\hat{R}(\mu) = R(\mu) + \lambda k
$$
* Robustness vs. Stability
    * Large $k$: more robust to outliers, but less stable
    * Small $k$: less robust to outliers, but more stable 

#### K-Means For Dimension Reduction
K-Means projects every data point to center of cluster it belongs to and hence can be used for data compression. Similar to PCA, find $w \in \mathbb{R}^{d\times k}$ such that<br>

$$
w^{\ast} = argmin_{w}\sum_{i=1}^{n}\lVert wz_{i} - x_{i} \rVert_{2}^{2}
$$

Consequently,<br>
$$
w^{\ast} = [\mu_{1}| \mu_{2}| \cdots| \mu_{k}]
$$

and

$$
z_{i} \in [e_{1}, e_{2}, \cdots, e_{k}]
$$
where $e_{j} \in \mathbb{R}^{k}$ with $j$-th element equals 1 and 0 elsewhere.