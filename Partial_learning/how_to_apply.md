It isn't obvious how to use such a thing.

First, let us address the fact that input and output must be the same shape.
We can relax this condition by defining input and output to be the same dimension, and define some proxy/intermediate functions between actual input and NN input, actual output and NN output.
These functions would have to be highly non-symmetric in variables. So, we have conversion via these non-symmetric functions, the NN processing of same-sized input/output, and then conversion to actual output. These non-symmetric functions can be initialized by a random process (randomly multiplying/adding things for each input), and then stay fixed.
