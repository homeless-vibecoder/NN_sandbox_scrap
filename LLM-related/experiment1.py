import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch.nn.functional as F

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

text = "We must vaccinate people in third world countries so that we stop needless deaths of millions of people." #"Hello, teacher. I don't have my homework because my dog ate my homework, but then he had diarrhea" #"A man walks into a bar. Ouch!"
tokens = tokenizer.tokenize(text)
print(tokens)

hidden_states_over_time = []

for i in range(1, len(tokens)+1):
    prefix = tokenizer.convert_tokens_to_string(tokens[:i])
    inputs = tokenizer(prefix, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    # get last token vector at chosen layer
    layer_vectors = [h[0, -1, :] for h in outputs.hidden_states]
    hidden_states_over_time.append(layer_vectors)

# compare consecutive tokens at a few layers
layers_to_plot = [-1]  # index 6 (or -1) is last one
for layer in layers_to_plot:
    distances = []
    for i in range(len(hidden_states_over_time)-1):
        v1 = hidden_states_over_time[i][layer]
        v2 = hidden_states_over_time[i+1][layer]
        d = 1 - F.cosine_similarity(v1, v2, dim=0)
        distances.append(d.item())
    plt.plot(distances, label=f"Layer {layer}")

plt.legend()
plt.title("Semantic shift per token (cosine distance)")
plt.show()
