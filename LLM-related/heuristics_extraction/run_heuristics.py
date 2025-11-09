#!/usr/bin/env python3
"""
Run heuristic estimation prompts with a local GPT-2 model.

The script reads prompt templates from `prompts_for_heuristics/*.txt`, injects the
user's request into each prompt, queries GPT-2 locally via `transformers`, parses
the numeric outputs, and finally prints both per-prompt details and the averaged
heuristic vector.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

try:  # Optional import; fail gracefully if dependencies are missing.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - runtime guard
    torch = None  # type: ignore[assignment]
    AutoModelForCausalLM = AutoTokenizer = None  # type: ignore[assignment]


ROOT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = ROOT_DIR / "prompts_for_heuristics"
DEFAULT_PROMPT_FILE = ROOT_DIR / "default_listener_prompt.txt"
DEFAULT_FALLBACK_PROMPT = "I want relaxing classical music that feels calm and soothing."


@dataclass(frozen=True)
class HeuristicSpec:
    name: str
    value_type: str  # "int" or "float"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[Sequence[int]] = None

    def clamp(self, value: float) -> float:
        """Clamp to configured bounds and nearest allowed value when needed."""
        if self.allowed_values is not None:
            # Choose the allowed value with the smallest absolute difference.
            return float(
                min(
                    self.allowed_values,
                    key=lambda allowed: abs(allowed - value),
                )
            )

        adjusted = value
        if self.min_value is not None:
            adjusted = max(self.min_value, adjusted)
        if self.max_value is not None:
            adjusted = min(self.max_value, adjusted)
        return adjusted

    def finalize(self, value: float) -> float:
        """Convert to the proper numeric type after clamping."""
        clamped = self.clamp(value)
        if self.value_type == "int":
            return float(int(round(clamped)))
        return clamped


NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# Order mirrors the acoustic feature list in `heuristics_description.txt`
HEURISTIC_SPECS: List[HeuristicSpec] = [
    HeuristicSpec("duration_ms", "int", min_value=0),
    HeuristicSpec("key", "int", allowed_values=list(range(-1, 12))),
    HeuristicSpec("mode", "int", allowed_values=[0, 1]),
    HeuristicSpec("time_signature", "int", min_value=1),
    HeuristicSpec("acousticness", "float", min_value=0.0, max_value=1.0),
    HeuristicSpec("danceability", "float", min_value=0.0, max_value=1.0),
    HeuristicSpec("energy", "float", min_value=0.0, max_value=1.0),
    HeuristicSpec("instrumentalness", "float", min_value=0.0, max_value=1.0),
    HeuristicSpec("liveness", "float", min_value=0.0, max_value=1.0),
    HeuristicSpec("loudness", "float", min_value=-60.0, max_value=0.0),
    HeuristicSpec("speechiness", "float", min_value=0.0, max_value=1.0),
    HeuristicSpec("valence", "float", min_value=0.0, max_value=1.0),
    HeuristicSpec("tempo", "float", min_value=0.0),
]


def parse_prompt_file(path: Path) -> tuple[List[str], List[str]]:
    """Return metadata comments and prompt bodies from a template file."""
    raw_text = path.read_text(encoding="utf-8")
    sections = [section.strip() for section in raw_text.split("---")]
    if not sections:
        raise ValueError(f"No sections found in prompt template: {path}")
    header = sections[0]
    prompts = [section for section in sections[1:] if section]
    metadata = [
        line.lstrip("# ").strip()
        for line in header.splitlines()
        if line.strip().startswith("#")
    ]
    if not prompts:
        raise ValueError(f"No prompt blocks found in template: {path}")
    return metadata, prompts


def generate_completion(
    prompt: str,
    tokenizer,
    model,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Generate text from GPT-2 and return only the new completion."""
    encoded = tokenizer(prompt, return_tensors="pt")
    input_length = encoded["input_ids"].shape[1]
    with torch.no_grad():  # type: ignore[union-attr]
        output_ids = model.generate(  # type: ignore[union-attr]
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else 1.0,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, input_length:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return completion.strip()


def extract_numeric_value(text: str, spec: HeuristicSpec) -> float:
    """Pull the first numeric token from the text and coerce to the spec."""
    match = NUMBER_PATTERN.search(text)
    if match is None:
        raise ValueError("No numeric value found in model output.")
    raw = match.group()
    try:
        value = float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Could not parse numeric value from '{raw}'.") from exc
    return spec.finalize(value)


def ensure_dependencies() -> None:
    if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
        raise SystemExit(
            "Missing dependencies for GPT-2 backend. Install them with:\n"
            "  pip install transformers torch"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project a listener prompt onto Spotify-style heuristics using GPT-2."
    )
    parser.add_argument(
        "--prompt",
        help="Listener request to analyze (overrides prompt file/fallback).",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help=(
            "Path to a file containing the default listener request "
            "(used when --prompt is not supplied)."
        ),
    )
    parser.add_argument(
        "--model",
        default="gpt2",
        help="Hugging Face causal language model checkpoint (default: gpt2).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=24,
        help="Maximum tokens to generate for each prompt completion (default: 24).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0.0 disables sampling (default: 0.0).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling value when temperature > 0 (default: 0.95).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_dependencies()

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not PROMPTS_DIR.exists():
        parser.error(f"Prompt directory does not exist: {PROMPTS_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    heuristic_results = {}

    if args.prompt is not None and args.prompt.strip():
        listener_prompt = args.prompt.strip()
    else:
        prompt_path = args.prompt_file
        listener_prompt = ""
        if prompt_path and prompt_path.exists():
            listener_prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not listener_prompt:
            listener_prompt = DEFAULT_FALLBACK_PROMPT

    if not listener_prompt.strip():
        parser.error(
            "No listener prompt provided. Supply --prompt or populate the prompt file."
        )

    print(f"Model: {args.model}")
    print(f"User prompt: {listener_prompt}")
    print()

    for spec in HEURISTIC_SPECS:
        template_path = PROMPTS_DIR / f"{spec.name}.txt"
        if not template_path.exists():
            parser.error(f"Missing prompt template for heuristic '{spec.name}': {template_path}")

        metadata, templates = parse_prompt_file(template_path)
        print(f"=== {spec.name} ===")
        for line in metadata:
            print(f"# {line}")
        prompt_values: List[float] = []

        for idx, template in enumerate(templates, start=1):
            filled_prompt = template.replace("{user_prompt}", listener_prompt)
            completion = generate_completion(
                filled_prompt,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(f"PROMPT INPUT ({idx}):\n{filled_prompt}")
            print(f"GPT RESPONSE ({idx}):\n{completion}\n")
            try:
                value = extract_numeric_value(completion, spec)
                prompt_values.append(value)
            except ValueError:
                print(f"- Parsing failed: no numeric value detected in GPT response.")

        if not prompt_values:
            heuristic_results[spec.name] = math.nan
        else:
            avg_value = statistics.fmean(prompt_values)
            finalized = spec.finalize(avg_value)
            display_value = (
                int(round(finalized)) if spec.value_type == "int" else round(finalized, 4)
            )
            print(f"Average: {round(avg_value, 4)} -> finalized value: {display_value}")
            heuristic_results[spec.name] = finalized
        print()

    ordered_values = [
        heuristic_results.get(spec.name, math.nan)
        for spec in HEURISTIC_SPECS
    ]
    formatted_values = [
        int(round(val)) if (not math.isnan(val) and spec.value_type == "int")
        else (round(val, 4) if not math.isnan(val) else float("nan"))
        for val, spec in zip(ordered_values, HEURISTIC_SPECS)
    ]
    print("Final heuristic vector:")
    print(formatted_values)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

