The main idea in "partial learning" is the following: we have a (linear - in a sense of linear transforms) function F^1, one application and we want to learn F^epsilon for small epsilon.
F^epsilon satisfies, if we do F^x(F^x(...)) n times, we get F^(nx).
This is useful because if we want, for example, in RL, to learn something, it is difficult to directly maximize the score, F. It would be easier if we could do F^0.001, and slowly/gradually go to F^1. Note that as h goes to zero, F^h goes to identity.

Current project: Suppose we have a mystery NN, F, which takes x (a few dimensional vector) and has output of same datatype/dimension.
We'd like to learn partial Fs. Let us do it one at a time: learn G, such that F^4=G^5. Initialize G as F, and run gradient descent over random inputs until convergence. Repeat until we reach F^epsilon. Gradually picking G helps in smooth learning, as opposed to instantly trying G^100000=F, which would be basically impossible.