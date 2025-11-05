The purpose of this folder partial_learning is to implement something similar to folder Partial_learning (in the root directory).

Here's what we want to do: we have in NN_weights/game_prediction state_predictor.pt. That has architecture or widths of input = 15, ...., 15 = output.
There is also, first and last hidden layers which have same dimension.

The idea in partial learning is, we learn the NN, as if we had applied it p/q times, such that if our original function was F, and G is F applied p/q times, then G^q=F^p, where ^x refers to applying it x times (e.g. F^2(.)=F(F(.))).

Now, we have two folders: time-wise and complexity-wise.
With time-wise, we take input, output, (15 dimensions), and do the partial training on that. This means, similar to Partial_learning in the root directory, we specify how small we go, and with what factor. e.g. if factor is 4/5, and start is F, we do F_1_1=F. Then, F^(4/5), corresponding to F_4_5 (F_p_q is the naming convention of the .pt file) is learned, such that F_4_5^5=F_1_1^4. Then we repeat to go to smaller F^(p/q).

We want to save the weights in NN_weights/game_prediction/partial_state_prediction_time-wise, when we are doing input to output (15 dims).
We start from state_predictor_partial_time-wise_1_1 = state_predictor (in game_prediction folder). Then, we populate the game_prediction/partial_state_prediction_time-wise folder with state_predictor_partial_time-wise_p_q for different p, q, using the recursive method, similar to Partial_learning in the root folder.

As for complexity-wise, we only deal with NN without the input and output - only the hidden layers.
So, our state_predictor_partial_complexity-wise_p_q is, input \to HIDDEN \to output. The input \to HIDDEN and HIDDEN \to output don't change. However, we do partial learning on the HIDDEN part. So, it looks like, we take NN, take away the input, output layers, do the F^p=G^q learning. After we are done, we stitch the input, output back to how it was, and we save the G appropriately. Then we repeat if we want to go further down.