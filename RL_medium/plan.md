We want the following game (it happens in 2D):
Simple stick figure is running - he only has basic digital control for moving joints (such a game (or similar one) exists). He has to balance, and player can click buttons to make his joints move so that he doesn't fall, and continues to run very far. If he falls, game is over.


High level plan:
We are going to create NN for the agent for RL, and we will train it.
New/original: we also try an alternative method of RL - we are going to create neural networks for the game (each predicting dynamics of the game - whatever the character gets in RL, those are the inputs whose evolution we need to understand/approximate).
Then, we are going to calculate partial functions (just like in Partial_learning), and train the RL on the partial function, gradually going to the non-relaxed version.

There is state_predictor.pt, and using that, we generate state_predictor_p_q.pt in the appropriate folder (including p=q=1). We should have a python file which, if run, just clears the folder game_prediction, and generates these .pt using state_predictor.pt.

We use one second in the future to predict. If we do partial learning on the input, output (as opposed to hidden layers), we should get shorter time-steps.