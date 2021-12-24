#### Bayesian Risk

Suppose $(x_{i}, y_{i}) \overset{\text{i.i.d.}}{\sim} P(x, y)$, baysian risk function is defined as 

$$
R(h) = \int P(x, y)l_{h}(y, h(x))dxdy = E_{x, y}[l(y, h(x))]
$$
with loss function $l$ and model/hypothesis $h$. Since

$$
P(x, y) = P(y|x)*P(x)
$$ and squared-loss function,

\begin{equation}
\begin{split}
R(h) & = \int P(y|x)P(x)(y-h(x))^{2}dxdy \\
& = \int_{\mathcal{X}}P(x)\underbrace{\int_{\mathcal{Y}}(y-h(x))^{2}P(y|x)dy}_\text{$E_{y|x}(y-h(x))^{2}|x$}dx \\
& = \int_{\mathcal{X}}P(x)E_{y|x}[(y-h(x))^{2}|x]dx \\
& = E_{x}[\underbrace{E_{y|x}[(y-h(x))^{2}|x]}_\text{$h^{\ast}(x)$}]
\end{split}
\end{equation}

#### Bayesian Optimal Prediction

$$
h^{\ast}(x) = argmin_{h \in \Theta}R(h) = argmin_{h \in \Theta}E_{x}[E_{y|x}[(y-h(x))^{2}|x]]
$$
$$
min_{h}E_{x}[E_{y|x}[(y-h(x))^{2}|x]] = E_{x}min_{h}E_{y|x}[(y-h(x))^{2}|x]
$$
Since squared-loss is convex, 
$$
h^{\ast}(x) = E(Y|X=x) = \int_{\mathcal{Y}}yP_{Y|X}(y|x)dy
$$
is called Bayes Optimal Prediction. 

Moverover, 
$$
P(Y|X) = \int_{\omega}P(y|x, w)P(w|\mathcal{D})dw
$$
where $w$ represents model parameters and $\mathcal{D}$ represents data. Bayesian prediction can be viewed as a weighted average over all possible hypotheses (models) and $P(w|\mathcal{D})$ is served as weights. When data is supporting a particular set of model parameters, its corresponding hypothesis has high impacts on final prediction. 
