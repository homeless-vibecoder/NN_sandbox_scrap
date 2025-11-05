Folder:
1) Game
2) RL NN for the game + visualization
3) NN for predicting game's next state + visualization
4) Partial NN-s for predicting game's next state - extracts hidden layers, does partial learning of it, and stitches the partially learned thing with the input, output layers, saves it as in Partial_learning (p_q, meaning F applied p/q times)
5) NN_weights - there are separate folders for RL, and game_prediction.
    -RL: we have two NN-s: conventional (which is going to be trained on the game) - so we'll name it qwop_policy_conventional.pt, and new_method (which is going to be trained using the process described in this project).
    -game_prediction. It is going to have a collection of state_prediction_p_q, similar naming convention to Partial_learning folder.