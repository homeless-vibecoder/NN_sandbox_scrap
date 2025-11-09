#!/usr/bin/env python3
"""
Compute embedding vectors for the term "energy" under each definition in
`prompt_engineering.md`, then report pairwise cosine similarities so you can see
how close the vectors are. You can pick between OpenAI's small embedding model
or a local Hugging Face GPT-2 checkpoint.

Usage (OpenAI backend, default)
-------------------------------
1. Export your OpenAI API key (once per shell):

   export OPENAI_API_KEY="sk-..."

2. Install dependencies (OpenAI backend needs `openai`):

   pip install --upgrade openai

3. Run the script (defaults point at the local `prompt_engineering.md` file):

   python get_energy_embeddings.py

Usage (Hugging Face GPT-2 backend)
----------------------------------
1. Install dependencies:

   pip install --upgrade transformers torch

2. Run with the `--backend hf` flag:

   python get_energy_embeddings.py --backend hf

Common options (usable with either backend):

   python get_energy_embeddings.py --file /absolute/path/to/prompt_engineering.md \\
       --term energy --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

try:  # Optional import; only required when using OpenAI backend.
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore[assignment]

try:  # Optional import; only required when using Hugging Face backend.
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover - handled at runtime
    torch = None  # type: ignore[assignment]
    AutoModel = AutoTokenizer = None  # type: ignore[assignment]


DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_HF_MODEL = "gpt2"


@dataclass(frozen=True)
class DefinitionChunk:
    title: str
    body: str

    @property
    def context(self) -> str:
        body = self.body.strip()
        return f"{self.title.strip()}\n\n{body}" if body else self.title.strip()


def parse_definitions(text: str) -> List[DefinitionChunk]:
    """Extract definition blocks that start with 'Energy — Definition ...'."""
    sections: List[DefinitionChunk] = []
    current_title: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_title, current_lines
        if current_title is None:
            return
        sections.append(
            DefinitionChunk(
                title=current_title.strip(),
                body="\n".join(current_lines).strip(),
            )
        )
        current_title = None
        current_lines = []

    for line in text.splitlines():
        if line.strip().startswith("Energy — Definition"):
            flush()
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    flush()
    return sections


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def _find_term_ranges(
    sequence: Sequence[int],
    pattern: Sequence[int],
) -> Iterator[Tuple[int, int]]:
    """Yield (inclusive, exclusive) index pairs where pattern appears in sequence."""
    if not pattern or len(pattern) > len(sequence):
        return
    pat_len = len(pattern)
    for idx in range(len(sequence) - pat_len + 1):
        if list(sequence[idx : idx + pat_len]) == list(pattern):
            yield idx, idx + pat_len


def create_embedding_inputs(
    sections: Iterable[DefinitionChunk],
    focus_term: str,
) -> list[str]:
    payloads = []
    for section in sections:
        payloads.append(
            f"{section.context}\n\nFocus term: {focus_term.strip()}"
        )
    return payloads


def fetch_embeddings_openai(
    model: str,
    inputs: list[str],
) -> list[list[float]]:
    if OpenAI is None:  # pragma: no cover - runtime guard
        raise SystemExit(
            "OpenAI backend requested, but the 'openai' package is missing.\n"
            "Install it via 'pip install openai', or run with '--backend hf'."
        )
    client = OpenAI()
    response = client.embeddings.create(
        model=model,
        input=inputs,
    )
    return [item.embedding for item in response.data]


def fetch_embeddings_hf(
    model_name: str,
    inputs: list[str],
    focus_term: str,
) -> list[list[float]]:
    if AutoModel is None or AutoTokenizer is None or torch is None:  # pragma: no cover
        raise SystemExit(
            "Hugging Face backend requested, but 'transformers' (and its dependencies) "
            "are missing.\nInstall them via 'pip install transformers torch', "
            "or run with the OpenAI backend."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    term_tokens = tokenizer(
        focus_term,
        add_special_tokens=False,
    ).input_ids

    vectors: list[list[float]] = []
    with torch.no_grad():
        for text in inputs:
            encodings = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            outputs = model(
                **encodings,
                output_hidden_states=True,
            )
            hidden = outputs.last_hidden_state[0]  # (seq_len, hidden_size)
            token_ids = encodings.input_ids[0].tolist()
            ranges = list(_find_term_ranges(token_ids, term_tokens))
            if ranges:
                slices = [
                    hidden[start:end] for start, end in ranges
                ]
                stacked = torch.cat(slices, dim=0)
                vector = stacked.mean(dim=0)
            else:
                vector = hidden.mean(dim=0)
            vectors.append(vector.cpu().tolist())

    return vectors


def fetch_embeddings(
    backend: str,
    openai_model: str,
    hf_model: str,
    inputs: list[str],
    focus_term: str,
) -> list[list[float]]:
    if backend == "openai":
        return fetch_embeddings_openai(openai_model, inputs)
    if backend == "hf":
        return fetch_embeddings_hf(hf_model, inputs, focus_term)
    raise ValueError(f"Unknown backend: {backend}")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare embeddings for 'energy' across prompt definitions."
    )
    default_file = Path(__file__).resolve().parent / "prompt_engineering.md"
    parser.add_argument(
        "--file",
        type=Path,
        default=default_file,
        help=f"Markdown file to read (default: {default_file})",
    )
    parser.add_argument(
        "--backend",
        choices=("openai", "hf"),
        default="openai",
        help="Embedding backend to use: OpenAI API or local Hugging Face (default: openai).",
    )
    parser.add_argument(
        "--term",
        default="energy",
        help="Focus term to embed within each context (default: energy).",
    )
    parser.add_argument(
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        help=f"OpenAI embedding model name (default: {DEFAULT_OPENAI_MODEL}).",
    )
    parser.add_argument(
        "--hf-model",
        default=DEFAULT_HF_MODEL,
        help=f"Hugging Face model for '--backend hf' (default: {DEFAULT_HF_MODEL}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the embeddings as JSON instead of a human-readable table.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_cli()
    args = parser.parse_args(argv)

    if not args.file.exists():
        parser.error(f"File does not exist: {args.file}")

    text = args.file.read_text(encoding="utf-8")
    sections = parse_definitions(text)
    if not sections:
        parser.error("No definition sections found that match 'Energy — Definition ...'")

    inputs = create_embedding_inputs(sections, args.term)
    embeddings = fetch_embeddings(
        backend=args.backend,
        openai_model=args.openai_model,
        hf_model=args.hf_model,
        inputs=inputs,
        focus_term=args.term,
    )

    if args.json:
        payload = {
            "backend": args.backend,
            "openai_model": args.openai_model if args.backend == "openai" else None,
            "hf_model": args.hf_model if args.backend == "hf" else None,
            "term": args.term,
            "file": str(args.file),
            "definitions": [
                {
                    "title": section.title,
                    "context": section.context,
                    "embedding": vector,
                }
                for section, vector in zip(sections, embeddings)
            ],
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.backend == "openai":
        print(f"Backend: OpenAI (model={args.openai_model})")
    else:
        print(f"Backend: Hugging Face (model={args.hf_model})")
    print(f"Term: {args.term!r}")
    print(f"Source file: {args.file}")
    print()

    for section, vector in zip(sections, embeddings):
        preview = ", ".join(f"{value:.3f}" for value in vector[:6])
        print(section.title)
        print(f"  Context length: {len(section.context.split())} words")
        print(f"  Embedding dims: {len(vector)}")
        print(f"  Head: [{preview}, ...]")
        print()

    if len(embeddings) > 1:
        print("Pairwise cosine similarity")
        for i, section_i in enumerate(sections):
            for j in range(i + 1, len(sections)):
                section_j = sections[j]
                score = cosine_similarity(embeddings[i], embeddings[j])
                print(f"  {section_i.title} ↔ {section_j.title}: {score:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

