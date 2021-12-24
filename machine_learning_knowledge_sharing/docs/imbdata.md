#### Oversampling

Apply KNN to fabricate data for minority class. 

Disadvantage

* Noise and even bias may be introduced to training data.

#### Downsampling

Random select a subset of data from majority class to match size of data in minority class.

Disadvantage

* Training data will be reduced and wasted.

#### Modify Objective Functions

Add a scalar to objective functions so that wrong predictions on minority class have similar loss as wrong predictions on majority class.