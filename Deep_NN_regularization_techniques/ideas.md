In this folder we want to try a few techniques that make training deep NN easier.

1) orthogonal weights layer-to-layer - earlier neurons need this more, since their goal is to increase expressiveness of the model, as opposed to follow gradient. Perhaps need variable weight on orthogonal penalty vs GD, similar to dual-variable.

2) smooth backpropagation - E[ \sum log(grad) ]=0. Activation function could also play a role in this: make activation function that isn't necessarily monotone. We want gradient descent to be approximated by a smoother method, leveraging LLN, CLT.

3) nested weights - instead of doing whole back-propagation, we ignore small entries. So, at each iteration we update only a few selected branches, and only once in a while do we update the whole network - for speed.



Let us first deal with orthogonal weights. An important aspect is tuning how strongly we enforce orthogonality. For that, we need to understand what tradeoff happens when enforcing orthogonality. 


how independent do we assume neurons to be? How stable is weights' orthogonality within GD - how much do orthogonal weights, after GD step, change?
Suppose independent neuron values. dL/dw for two weights attached to the same neuron from behind, (their difference) is only going to depend on value of the neuron in the earlier layer.
Basically, this yields, at each GD step, an addition of random vector. Now, the projection to other (previously orthogonal) vectors is going to be roughly evenly distributed. So, if random vector has length r, and there are n vectors/neurons in one layer, the sum of squares in dot product (Frobernious norm squared of matrix WW^T-D), is going to get an extra n*(r/n)^2=r/n term.
Summing over n vectors, we get r. So, deviation in Forbernious norm squared is same as r - the average GD stepsize.
In such GD, l2 distance traveled is roughly ||r*\all_ones_vector{1}||_2=sqrt(n)r.

If we instead measure l1 norm of WW^T-D (which roughly corresponds to extra_distance_traveled due to redundancy), we get n*r/(n-1)=r-1/(n-1) for each, and nr-1+1/n=nr-1 for all vectors combined.
l1 distance we travel in total with GD step is rn.

