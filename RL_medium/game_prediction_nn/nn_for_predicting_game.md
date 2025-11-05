We would like to train a NN that is going to predict game's next state (after some time, not necessarily immediately after), given its current state.

There are a few things to take into consideration: We'd like to be able to generate a game with its initial state to be the input of the NN. This is for training - we put random input in the NN, and check the answer with simulation (1 second in the future, for example).

Note: we also want to be able to compute score (displacement from the start of the game).

Another important aspect is the architecture: we want NN to go from inputs to layer of fixed width. Then, there are multiple hidden layers with the same width. At the end, there is output from the width to the output dimension. So, what is important is, first and last hidden layers have same dimension (inner hidden layers' widths don't really matter).