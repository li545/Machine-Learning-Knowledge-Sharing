#### Basic Idea
Suppose with data $\mathcal{D} = \{x_{1}, x_{2}, \cdots, x_{n} \}$, we want to find the optimal parameter $\theta$ with MLE. The objective function given by log-likelihood of joint density is,
$$
l(\theta) = \sum_{i}logp(x_{i}|\theta)
$$
With hidden class memebership $z_{i}$ for $x_{i}, i \in \{1, \cdots, n\}$,
$$
l(\theta) = \sum_{i}log\sum_{k}p(x_{i}, z_{i}=k|\theta)
$$
Since the above function is hard to optimize, we assume complete data $\mathcal{D}^{(c)} = \{(x_{1}, z_{1}), (x_{2}, z_{2}), \cdots, (x_{n}, z_{n}) \}$. 

Accordingly, the objective function is,
$$
l^{(c)}(\theta) = \sum_{i}logp(x_{i}, z_{i}|\theta)
$$
The objective function cannot be optimized because $z_{i}$ is unknown. To deal with it, we can define an expected log-likelihood on the complete data as,
$$
Q(\theta, \theta^{t-1}) = \mathbb{E}[l^{(c)}(\theta)|\mathcal{D}, \theta^{t-1}]
$$

* E-step: Compute $Q(\theta, \theta^{t-1})$
* M-step: Compute $\theta^{t} = argmax_{t}Q(\theta, \theta^{t-1})$

To replace MLE with MAP, simply add $logp(\theta)$ to Q-function, 
$$
\theta^{t} =  Q(\theta, \theta^{t-1}) + logp(\theta)
$$
with $p(\theta)$ is a prior of $\theta$.

#### Mixture of Experts
Given data $\mathcal{D} = \{(x_{1}, y_{1}), \cdots, (x_{n}, y_{n}) \}$ and a set of experts (classifiers on different range of $x$),
$$
y_{i}|x_{i}, z_{i}=k, \theta \sim \mathcal{N}(w_{k}^{T}x_{i}, \sigma_{k}^{2})
$$
$$
z_{i}=k|x_{i}, \theta \sim \mathcal{Cat}(\mathcal{S}(v^{T}x_{i}))
$$
where $\mathcal{Cat}$ is some categorical distribution.
\begin{equation}
\begin{split}
p(y_{i}|x_{i}, \theta) & = \sum_{k'}p(y_{i}, z_{i}=k'|x_{i}, \theta) \\
& = \sum_{k'}p(y_{i}|z_{i}=k',x_{i}, \theta)p(z_{i}=k'|x_{i}, \theta)
\end{split}
\end{equation}

\begin{equation}
\begin{split}
Q(\theta, \theta^{t-1}) &= \mathbb{E}[\sum_{i}logp(y_{i}|z_{i}=k, x_{i}, \theta^{t-1})p(z_{i}=k|x_{i}, \theta^{t-1})] \\
&= \mathbb{E}[\sum_{i}log\prod_{k'}[\underbrace{p(y_{i}|z_{i}=k', x_{i}, \theta^{t-1})}_\text{$\mathcal{N}(y_{i}|w_{k'}^{T}x_{i}, \sigma_{k'}^{2})$}\underbrace{p(z_{i}=k'|x_{i}, \theta^{t-1})}_\text{$\pi_{i,k'}=\mathcal{S}(v^{T}x_{i})_{k'}$}]^{\mathbb{I}_{[z_{i}=k']}}] \\
&= \mathbb{E}[\sum_{i}\sum_{k'}\mathbb{I}_{[z_{i}=k']}log\pi_{i,k'}\mathcal{N}(y_{i}|w_{k'}^{T}x_{i}, \sigma_{k'}^{2})] \\
&= \sum_{i}\sum_{k'}\mathbb{E}[\mathbb{I}_{[z_{i}=k']}]log\pi_{i,k'}\mathcal{N}(y_{i}|w_{k'}^{T}x_{i}, \sigma_{k'}^{2})
\end{split}
\end{equation}

Define the probability that $i$-th instance belongs to $k$-th class as $r_{ik}$, then

$$
\mathbb{E}[\mathbb{I}_{[z_{i}=k]}] = r_{ik} \times 1 + (1-r_{ik}) \times 0 = r_{ik}
$$ 

and

\begin{equation}
\begin{split}
r_{ik} &= p(z_{i}=k|y_{i}, x_{i}, \theta) \\
& = \frac{p(z_{i}=k, y_{i}|x_{i}, \theta)}{\sum_{k}p(z_{i}=k, y_{i}|x_{i}, \theta)} \\
& = \frac{p(y_{i}|z_{i}=k, x_{i}, \theta)p(z_{i}=k|x_{i}, \theta)}{\sum_{k}p(z_{i}=k, y_{i}|x_{i}, \theta)} \\
& \propto \pi_{i,k}\mathcal{N}(y_{i}|w_{k}^{T}x_{i}, \sigma_{k}^{2})
\end{split}
\end{equation}

with $\pi_{i,k} = \mathcal{S}(v^{T}x_{i})_{k}$

Consequently,

$$
Q(\theta, \theta^{t-1}) = \sum_{i}\sum_{k'}r_{ik'}log[\pi_{i,k'}\mathcal{N}(y_{i}|w_{k'}^{T}x_{i}, \sigma_{k'}^{2})]
$$

#### EM For Student Distribution
A t-distribution can be written as a Gaussian Scale Mixture, 

$$
\mathcal{T}(x_{i}|\mu, \Sigma, \nu) = \int\mathcal{N}(x_{i}|\mu, \frac{\Sigma}{z_{i}})\Gamma(z_{i}|\frac{\nu}{2}, \frac{\nu}{2})dz_{i}
$$

where $\Gamma(z_{i}|\frac{\nu}{2}, \frac{\nu}{2})$ is Gamma Distribution.
The t-distribution can be viewed as an infinite number of Gaussian mixtures and each component has a slightly different variance-covariance matrix.

With complete Data $\mathcal{D} = \{(x_{1}, z_{1}), \cdots, (x_{n}, z_{n})\}$ considering $z_{i}, i \in \{1, \cdots, n\}$ missing, 

\begin{equation}
\begin{split}
l^{(c)}(\theta) &= \sum_{i}logp(x_{i}, z_{i}|\theta) \\
&= \sum_{i}log[\underbrace{p(x_{i}|z_{i}, \mu, \Sigma)}_\text{$\mathcal{N}(x_{i}|\mu, \frac{\Sigma}{z_{i}})$}\underbrace{p(z_{i}|\nu)}_\text{$\Gamma(z_{i}|\frac{\nu}{2}, \frac{\nu}{2})$}] \\
&= \sum_{i}[log\mathcal{N}(x_{i}|\mu, \frac{\Sigma}{z_{i}}) + log\Gamma(z_{i}|\frac{\nu}{2}, \frac{\nu}{2})] \\
&= L_{\mathcal{N}}(\mu, \Sigma) + L_{\Gamma}(\nu)
\end{split}
\end{equation}

with 

$$
L_{\mathcal{N}}(\mu, \Sigma) = -\frac{n}{2}log|\Sigma| -\frac{1}{2}\sum_{i=1}^{n}z_{i}(x_{i}-\mu)^{T}\Sigma^{-1}(x_{i}-\mu)
$$

$$
L_{\Gamma}(\nu) = -nlog\Gamma(\frac{\nu}{2}) + \frac{1}{2}n\nu log(\frac{\nu}{2}) + \frac{1}{2}\nu\sum_{i}(logz_{i}-z_{i})
$$

* $\nu$ is known

We don't have to optimize $L_{\Gamma}(\nu)$. $\mathbb{E}[l^{(c)}(\theta)]$ depends on $\mathbb{E}[z_{i}]$. 

With a prior $z_{i}\sim \Gamma(\frac{\nu}{2}, \frac{\nu}{2})$ and data $\{x_{1}, \cdots, x_{n}\} \sim \mathcal{N}(x_{i}|\mu, \frac{\Sigma}{z_{i}})$, the posterior distribution of $z_{i}$ is also Gamma, i.e., 

$$
z_{i}|x_{i}, \theta \sim \Gamma(\frac{\nu+D}{2}, \frac{(x_{i}-\mu)^{T}\Sigma^{-1}(x_{i}-\mu)}{2})
$$

Consequently, $\mathbb{E}[z_{i}|x_{i}, \theta] = \frac{\nu+D}{(x_{i}-\mu)^{T}\Sigma^{-1}(x_{i}-\mu)}$ with $D$ as dimensions of $x_{i}$.

* $\nu$ is unknown

Expectation of $L_{\Gamma}(\nu)$ depends on $\mathbb{E}[logz_{i}|x_{i}, \theta]-\mathbb{E}[z_{i}|x_{i}, \theta]$.

$\textbf{Corollary}$

If $X\sim \Gamma(a,b)$, then 

$$
\mathbb{E}[log(X)|\theta] = \Psi(a) - logb
$$

where $\Psi(a)=\frac{d}{da}log\Gamma(a)$.

#### EM For Probit Regression
With data $\mathcal{D} = \{(x_{1}, y_{1}), \cdots, (x_{n}, y_{n})\}$ and a hidden variable $z_{i} \sim \mathcal{N}(w^{T}x_{i}, 1)$,

$$
p(y_{i}=1|z_{i}) = \mathbb{I}_{[z_{i}>0]}
$$

The log-likelihood of complete data is,

\begin{equation}
\begin{split}
l^{(c)}(\theta) &= \sum_{i}log[p(y_{i}, z_{i}|x_{i}, \theta)]\\
&= \sum_{i}log[p(y_{i}|z_{i}, x_{i}, \theta)p(z_{i}|x_{i}, \theta)] \\
&= \sum_{i}logp(y_{i}|z_{i}) - \frac{1}{2}(\mathbf{z}-\mathbf{w}^{T}\mathbf{x})^{T}(\mathbf{z}-\mathbf{w}^{T}\mathbf{x})
\end{split}
\end{equation}

$\mathbb{E}[l^{(c)}(\theta)|\mathcal{D}, \theta^{t-1}]$ depends on $\mathbb{E}[z_{i}]$. Posterior distribution of $z_{i}$ given $\mathcal{D}$, 

$$
p(z_{i}|x_{i}, y_{i}, \theta) = \frac{p(z_{i}, y_{i}|x_{i}, \theta)}{p(x_{i}, \theta)} \propto p(y_{i}|z_{i}, x_{i}, \theta)p(z_{i}|x_{i}, \theta)
$$

As a result, 

$$
p(z_{i}|x_{i}, y_{i}, \theta) =
\begin{cases}
\mathcal{N}(w^{T}x_{i}, 1)\mathbb{I}_{[z_{i}>0]} &\text{if }y_{i} = 1 \\ 
\mathcal{N}(w^{T}x_{i}, 1)\mathbb{I}_{[z_{i}<0]} &\text{if }y_{i} = 0  
\end{cases}
$$