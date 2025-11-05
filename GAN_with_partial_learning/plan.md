We want to train GAN with a twist.

Let us pick simple domain/fast to train - e.g. MNIST handwritten digits.

Architecture of NN: input and output layers are irrelevant. However, first and last hidden layers must have the same width.

The idea is this: a NN, for example, G (generator) is complicated, since it has hidden layers. If we removed the hidden layers, it would become simple.
Now, think of hidden layer as applying f(.) to first hidden layer to obtain last hidden layer (the neuron activations that is).
Imagine somehow applying f 0.5 times (denote it by f^0.5). f^1 would be same as f. f^0.5 is such that f^0.5(f^0.5(.))=f^1(.).
f^0.5 can be found by learning it from f^1: f^1(.)=f^0.5(f^0.5(.)). Note: that is why we need first and last hidden layers to have same width - so that we can apply f^0.5(f^0.5(.)).
Note that this can be generalized to arbitrary f^alpha. f^a_1(f^a_2(...))=f^b_1(f^b_2(...)), as long as \sum a_j = sum b_j.

We should have two separate folders - one with a list of G^alpha, the other with D^alpha, where .^alpha means applying . alpha times.

It is useful to store NNs G^alpha for alpha=2^-n, for n=0, 1, 2, 3, etc., since any other value (e.g. 0.75=1/2+1/4) can be obtained by combination of G^alpha.

Scheduling training: we have an estimate of how good each NN is for its task (note: f^0.5 needs to have lower loss that f for it to be as good - f^0.5(f^0.5) should have same loss as f to them to be equally good).

When training f^a_j against f^b_j (so, f^a_1(f^a_2(...))=f^b_1(f^b_2(...)), such that \sum a_j = \sum b_j or it is within small tolerance), we nudge weights of each f^k, depending on how "good" it is - if it is very good, we make smaller change to it.