import os
import json
import torch

from nerual_network_class import (
    load_or_generate_F,
    learn_fractional_power,
    learn_fractional_chain,
)


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r") as f:
    CFG = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_dir = os.path.join(os.path.dirname(__file__), CFG["paths"]["weights_dir"])

dim = CFG["model"]["dim"]
hidden_dim = CFG["model"]["hidden_dim"]
num_layers = CFG["model"]["num_layers"]

steps = CFG["train"]["steps"]
batch_size = CFG["train"]["batch_size"]
lr = CFG["train"]["lr"]
input_scale = CFG["train"]["input_scale"]

p = CFG["train"]["p"]
q = CFG["train"]["q"]
num_iters = CFG["train"]["num_iters"]
min_alpha = CFG["train"]["min_alpha"]


if __name__ == "__main__":
    # Load or create the base F
    F_model = load_or_generate_F(
        dim=dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
        weights_dir=weights_dir,
        filename=CFG["paths"]["F_filename"],
        seed=0,
    )

    # Iteratively learn chain: G_k where F_{k-1}^p ≈ G_k^q and F_k := G_k
    results = learn_fractional_chain(
        F_model=F_model,
        p=p,
        q=q,
        dim=dim,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        input_scale=input_scale,
        device=device,
        weights_dir=weights_dir,
        save=True,
        num_iters=num_iters,
        min_alpha=min_alpha,
    )

    print({
        "device": str(device),
        "p_q": (p, q),
        "num_iters": num_iters,
        "min_alpha": min_alpha,
        "results": results,
        "weights_dir": weights_dir,
        "F_weights": os.path.join(weights_dir, CFG["paths"]["F_filename"]),
    })


