Suppose we have functions/NNs f^alpha, and we want to evaluate how accurate/good they are, given f - the function we try to approximate.
We need a scheme for updating the score (for goodness) for each NN. The main action that happens is, we train f^a_j against f^b_j.

We want goodness score to correspond to upper bound on error/loss. Let goodness score be g in (0, infinity), and f^a has score g if f^a(f^a(...)) applied 1/a times (so that we can compare with f), has error < 1/g (can be C/g for fixed constant C).

We can "prove" goodness of a NN by comparing it with good NN - e.g. if f^0.8 is good - close to the true function - and f^0.4(f^0.4) is close to f^0.8, by triangle inequality, f^0.4 has to be close to the true function also.
Proving isn't necessary - we can assume that error is towards random direction: functions with errors (C/g_1, C/g_2) e_1, e_2 from the optimal, have expected difference sqrt(e_1^2+e_2^2), approximately (e_1+e_2)/sqrt(2) (when e_1=e_2). This is tight in high dimensions.

For each comparison, let us nudge g - use adam (we need to think of dependence of stepsize on g).

Ignore this paragraph:
*
Also note, error is distributed in some way - ie when doing f^0.5(f^0.5(.)), there is error/deviation in f^0.5, and then we have f^0.5(f^0.5(.))=f^0.5(y+e), where e is the error, y is true value.
f^0.5(y+e) is approx, f^0.5(y)+e(f^0.5)'.
y, e are more skewed towards large eigenvalue eigenvector of f^0.5. Note: f^a has same spectrum^a as f (note that f isn't linear, but we can think of "eigencurves" which roughly have a spectrum - locally linearize).
Likewise, derivative of f^a is (f')^a (coordinate-wise).
*


Anyway, let us assume if f^a(.) has error e, its corresponding error in f^1 is e^(1/a).
So, if e is error of f^a(f^a(...))-f^1, then f^a, compared to true f^a, has error e^a.

As for dependence of learning rate on g, we need to roughly say, exp(d(log(e_1)/log(e_2))), where e_1, e_2 are errors (C/g), and d is "effective dimension".
Effective dimension is connected to number of steps one needs to decrease error - if we picked randomly, d would be dimension; but d is in reality much lower, since we don't do random search - rather it is guided by GD.
Let s(r) be number of steps/iterations to go from error r to r-dr. Then, the stepsize is (proportional to) s(r), r=C/g.
Let s(r) be (1/r)*sqrt of variance of the gradients when training (standard deviation/r).
Assume standard deviation is roughly sqrt(r).
Then, stepsize is roughly 1/sqrt(r)=(g/C)^2.

Okay, anyway basically, let goodness g be s.t. error is bounded by C/g^2.
Stepsize is roughly proportional to 1/g, and g is updated by. NEVERMIND.

Each NN, has error r. stepsize is lr*sqrt(r). Updated r is r+lr *sqrt(r) * sgn(E[ r_j ]-r) * sqrt(abs(E[ r_j ]-r)) (assume step was taken in average r_j direction, where r_j is errors of each function agaianst which we train). Note: r_j, for f^a is taken to be r_j^a - that's the  contribution of that function. E[] is also taken over the r_j^a.