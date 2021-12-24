![Screenshot](img/bag_boost.png)

#### Bagging
Bagging is short for boostrap aggregating. Each tree is grown with a boostrap sample (sampling with replacements) drawn from traning dataset and a random subset of features.<br>
$$
MSE = Variance + Bias^{2}
$$

* To have small bias - Reduce depth of trees<br>

* To have small variance  
    * Increase the number of trees<br>
    * Reduce correlation among trees by decreasing the number of features of each tree

#### Boosting
Boosting improves prediction accuracy from prior trees. Trees are correlated.<br>

* Adaptive Boosting
![Screenshot](img/adaboost.png)
![Screenshot](img/adab_alg.png)
<br>
* Gradient Boosting
![Screenshot](img/grad_boost.png)

#### Out-of bag (OOB) error
![Screenshot](img/oob.png)

Bagging has low variance and Boosting has high accuracy.