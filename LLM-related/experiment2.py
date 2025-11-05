import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("distilgpt2")
mod = AutoModelForCausalLM.from_pretrained("distilgpt2")
mod.eval()

text = "We must vaccinate people in third world countries so that we stop needless deaths of millions of mosquitoes."
enc = tok(text, return_tensors="pt")

with torch.no_grad():
    out = mod(**enc, output_hidden_states=True, return_dict=True)

# hidden states: (n_layers, 1, seq_len, hidden_dim)
hs = torch.stack(out.hidden_states)[:,0,:,:]
logits = out.logits[0]

# measure surprisal (negative log prob of actual token)
probs = torch.softmax(logits[:-1], dim=-1)
target = enc.input_ids[0,1:]
surprisal = -torch.log(probs.gather(-1, target.unsqueeze(-1)).squeeze(-1))

# cosine distance between consecutive context vectors
def cosine(a,b): return 1 - F.cosine_similarity(a,b,dim=-1)
dist = cosine(hs[-1,:-1,:], hs[-1,1:,:])

for tok_id, d, s in zip(enc.input_ids[0], dist, surprisal):
    print(f"{tok.decode([tok_id]):10s} Δ={d.item():.3f}  surprisal={s.item():.3f}")
