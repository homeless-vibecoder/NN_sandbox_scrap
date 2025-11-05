# distilgpt2_vectors.py
# Requires: pip install transformers torch
# Run: python distilgpt2_vectors.py

import os
os.environ["OMP_NUM_THREADS"] = "4"  # tune for your CPU
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

DEVICE = "cpu"  # we're targeting CPU per your request

# --------------------------
# Load model & tokenizer
# --------------------------
model_name = "distilgpt2"   # distilgpt2 = compact, faster than gpt2 and good for CPU experiments
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
model.eval()

# quick helpers
def ids_to_tokens(ids):
    return [tokenizer.decode([int(i)]) for i in ids]

def cosine(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# --------------------------
# Example prompt
# --------------------------
prompt = "The quick brown fox jumped over the"
print("PROMPT:", prompt)
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
input_ids = inputs["input_ids"]  # shape (1, seq_len)
seq_len = input_ids.shape[1]
tokens = ids_to_tokens(input_ids[0])
print("TOKENS:", tokens)

# --------------------------
# 1) Forward once to get hidden states for all layers (easy method)
# --------------------------
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True, return_dict=True)
hidden_states = out.hidden_states  # tuple: len = num_layers + 1
# hidden_states[0] = token embeddings after input embedding layer
# hidden_states[1] = output of transformer block 0, etc.
print("Number of hidden_state entries (incl. embeddings):", len(hidden_states))
print("Each hidden state shape (batch, seq_len, hidden_dim):")
for i, h in enumerate(hidden_states):
    print(f"  layer {i:2d}:", tuple(h.shape))

# extract vectors for position p in sequence across layers
pos = seq_len - 1  # last token in the prompt
print(f"\nInspecting token at position {pos} (token: '{tokens[pos]}') across layers:")
vecs_across_layers = [hidden_states[i][0, pos].cpu() for i in range(len(hidden_states))]

# show cosine similarity of later layers to initial embedding
base = vecs_across_layers[0]
sims = [float(F.cosine_similarity(base.unsqueeze(0), v.unsqueeze(0))) for v in vecs_across_layers]
for i, s in enumerate(sims):
    print(f" layer {i:2d} similarity to input embedding: {s:.4f}")

# --------------------------
# 2) Generate a few tokens while collecting hidden states at each generation step
# (iterative sampling loop which gives hidden states at each step)
# --------------------------
def sample_next_token(logits, temperature=1.0, top_k=50, top_p=0.95):
    # logits: (vocab,)
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / max(temperature, 1e-8)
    # top-k
    if top_k is not None and top_k > 0:
        topk_vals, topk_idx = torch.topk(logits, top_k)
        min_topk = topk_vals[-1]
        logits = torch.where(logits < min_topk, torch.tensor(-1e10, device=logits.device), logits)
    # top-p (nucleus)
    if top_p is not None and 0 < top_p < 1:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative > top_p
        if cutoff.any():
            # mask out tokens after first True
            first_index = int(torch.nonzero(cutoff, as_tuple=False)[0].item())
            cutoff_idx = sorted_idx[first_index+1:]
            logits[cutoff_idx] = -1e10
    probs = F.softmax(logits, dim=-1)
    next_id = int(torch.multinomial(probs, num_samples=1).item())
    return next_id

# iterative generation collecting hidden states step-by-step
def generate_and_collect(prompt_ids, gen_len=20, temperature=0.8, top_k=40, top_p=0.95):
    cur_ids = prompt_ids.clone()
    all_generated = []
    per_step_hidden = []  # list of tuples (hidden_states_for_layers) at each step
    with torch.no_grad():
        for step in range(gen_len):
            outputs = model(cur_ids, output_hidden_states=True, return_dict=True)
            # hidden_states: tuple (num_layers+1) of tensors (batch, seq_len, hidden_dim)
            hs = [h.detach().cpu().clone() for h in outputs.hidden_states]
            per_step_hidden.append(hs)
            # logits for last token:
            logits = outputs.logits[0, -1]  # (vocab,)
            next_id = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            all_generated.append(next_id)
            # append next token to cur_ids
            cur_ids = torch.cat([cur_ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)
    return all_generated, per_step_hidden

gen_len = 12
generated_ids, per_step_hidden = generate_and_collect(input_ids, gen_len=gen_len, temperature=0.8, top_k=40, top_p=0.9)
generated_tokens = ids_to_tokens(generated_ids)
print("\nGENERATED TOKENS:", generated_tokens)
print("Full generated text:\n", tokenizer.decode(torch.cat([input_ids[0], torch.tensor(generated_ids)]).tolist()))

# per_step_hidden[n] gives hidden states tuple after step n (0-based).
# each element in the tuple is (1, seq_len+n+1, hidden_dim) so you can inspect any token position.

# --------------------------
# 3) Visualize how vectors move across layers for the newly generated last token
# --------------------------
# pick the last generated token's position (after generation)
final_step = len(per_step_hidden)-1
# final_step is 0-based; at step s, sequence length is input_len + s, so last index is input_len + s - 1
last_pos = input_ids.shape[1] + final_step - 1  # index of last token in current sequence
print(f"\nInspecting generated last token at overall position {last_pos} -> token '{ids_to_tokens([generated_ids[-1]])[0]}'")
vecs_last_token = [per_step_hidden[final_step][layer][0, last_pos].cpu() for layer in range(len(per_step_hidden[final_step]))]
sims2 = [float(F.cosine_similarity(vecs_last_token[0].unsqueeze(0), v.unsqueeze(0))) for v in vecs_last_token]
for i, s in enumerate(sims2):
    print(f" generated token - layer {i:2d} similarity to its embedding: {s:.4f}")

# --------------------------
# 4) Intervention example: replace the hidden vector for a token at a chosen layer
#    We'll run the transformer "manually" layer-by-layer so we can intervene.
# --------------------------
def forward_with_intervention(input_ids, intervene_pos, layer_to_replace, replacement_vector):
    """
    input_ids: tensor (1, seq_len)
    intervene_pos: int, 0-based position in sequence to modify
    layer_to_replace: int, which hidden_state index to replace (0 = embeddings, 1 = after block 0, ...)
    replacement_vector: torch tensor of shape (hidden_dim,) or (1, hidden_dim)
    Returns logits for final tokens after finishing forward pass.
    """
    # prepare embeddings + position ids
    bsz, seqlen = input_ids.shape
    device = input_ids.device
    # input embeddings (wte) and positional embeddings (wpe)
    wte = model.transformer.wte
    wpe = model.transformer.wpe
    inputs_embeds = wte(input_ids) + wpe(torch.arange(seqlen, device=device).unsqueeze(0))
    hidden = inputs_embeds  # this corresponds to hidden_states[0]
    # if we need to replace at layer 0 (embeddings), do it now:
    if layer_to_replace == 0:
        hidden = hidden.clone()
        # replacement_vector shape fix
        rv = replacement_vector.clone().view(1, 1, -1)
        hidden[0, intervene_pos] = rv[0, 0]

    # iterate transformer blocks
    for i, block in enumerate(model.transformer.h):
        # call block: block returns (hidden_states, present, attn) depending on HF internals
        # Use kwargs that are safe for CPU
        out = block(hidden, use_cache=False)
        hidden = out[0]
        # if this is the layer to replace (layer index i+1 corresponds to hidden_states index)
        if layer_to_replace == (i + 1):
            hidden = hidden.clone()
            rv = replacement_vector.clone().view(1, 1, -1)
            hidden[0, intervene_pos] = rv[0, 0]

    # final layer norm
    hidden = model.transformer.ln_f(hidden)
    # logits
    logits = model.lm_head(hidden)  # (1, seq_len, vocab)
    return logits

# choose an intervention:
# - intervene at the last token of the original prompt (pos)
# - replace its vector at layer 3 with the vector of token "Paris" embedding
intervene_pos = pos
layer_to_replace = 3  # 0 = embeddings, 1..N = after block i-1
print(f"\nIntervention: replace token at position {intervene_pos} at layer {layer_to_replace}")

# build replacement vector: use the input embedding vector for the word "Paris"
rep_token = " Paris"
rep_id = tokenizer.encode(rep_token, add_special_tokens=False)
print("rep_id tokens for ' Paris':", rep_id, "-> decoded:", ids_to_tokens(rep_id))
# take the first subtoken's input embedding as replacement (shape hidden_dim)
rep_emb = model.get_input_embeddings()(torch.tensor([rep_id[0]], device=DEVICE)).detach().cpu().squeeze(0)

# run forward_with_intervention using the *original prompt* as input and then sample continuations
logits_after = forward_with_intervention(input_ids, intervene_pos, layer_to_replace, rep_emb.to(DEVICE))
# sample next token from logits for last position
next_logits = logits_after[0, -1]
next_tok = sample_next_token(next_logits, temperature=0.8, top_k=40, top_p=0.9)
print("Next token AFTER intervention (id):", next_tok, " token:", ids_to_tokens([next_tok])[0])

# For a more thorough check: generate N tokens after intervention by iterative loop
def generate_after_intervention(input_ids, intervene_pos, layer_to_replace, replacement_vector, gen_len=12):
    cur_ids = input_ids.clone()
    with torch.no_grad():
        # We'll run the intervened forward once, sample next token, append to cur_ids, and then use standard iterative generation
        for step in range(gen_len):
            if step == 0:
                logits = forward_with_intervention(cur_ids, intervene_pos, layer_to_replace, replacement_vector)[0, -1]
            else:
                o = model(cur_ids, output_hidden_states=False, return_dict=True)
                logits = o.logits[0, -1]
            next_id = sample_next_token(logits, temperature=0.8, top_k=40, top_p=0.9)
            cur_ids = torch.cat([cur_ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)
    return cur_ids

cur_after = generate_after_intervention(input_ids, intervene_pos, layer_to_replace, rep_emb.to(DEVICE), gen_len=12)
print("Text after intervention:\n", tokenizer.decode(cur_after[0].tolist()))

# --------------------------
# End of script
# --------------------------
