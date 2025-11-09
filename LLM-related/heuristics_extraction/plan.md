I am given a text prompt from a user.
I want to "project" that prompt onto the heuristics - so, the prompt will be in spirit of: "I want to listen to a relaxing classical music", and then we need to generate, what values of heuristics this prompt corresponds to.
The list of heuristics and their descriptions are in heuristics_description.txt

I want to be able to extract the heuristics from an input (that I will put in code as a string).

Create a file for each heuristic, where I can design a prompt (and then I can write some placeholder for the input of the user - the user's input will go there before feeding the prompt to LLM).
Make it such that I can design multiple prompts, so that I can get multiple different answers for more robustness.
I also need multiple different prompts, since I will want to compare the output score for the heuristic between the prompts.
I want a separate file for each heuristic (in prompts_for_heuristics folder).
In each file, I will input the collection of prompts.


I want a python file that is going to output the heuristics for the input string (by inserting them into the pre-designed prompts).
I want this file to print the results: what the heuristics are.
There might be multiple prompts for each heuristic. I want to print all of them.
So, there should be print, for each heuristic, different guesses for different prompts.
Then, after all heuristics are listed, we have a "final answer" which is a vector, containing all heuristics. Each heuristic is the average of the prompts of that heuristic (so just average the answers).


As for the heuristics/features, there is no need to include song_id. The rest, do include.

The folder in which I want this implemented is "/Users/irakli/Desktop/LLM sandbox/LLM-related/heuristics_extraction"

