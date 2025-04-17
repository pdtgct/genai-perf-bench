#!/usr/bin/env python
import os
import glob
import json
import argparse
from typing import List, Dict, Iterator
from transformers import AutoTokenizer
from tqdm import tqdm
import unicodedata
import re

SYSTEM_PROMPT = (
    "You are an infinite-narrative engine. "
    "When given an excerpt, continue the text. Write without summarizing, "
    "concluding, or stopping, so the narrative can flow indefinitely."
)

def clean_text(s: str) -> str:
    """
    Remove ambiguous unicode (non-breaking space, zero-width, control chars except \\n).
    Replace curly quotes, dashes, ellipsis, and similar with ASCII equivalents. No normalization.
    """
    # Replace curly quotes, dashes, ellipsis, etc. with ASCII equivalents
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"',
        "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
        "\u2026": "...", "\u2010": "-", "\u2011": "-", "\u2012": "-",
        "\u2017": "_", "\u2032": "'", "\u2033": '"', "\u00B4": "'",
        "\u02BC": "'", "\u02BB": "'", "\u201A": "'", "\u201E": '"',
        "\u2039": "<", "\u203A": ">", "\u00AB": "<<", "\u00BB": ">>",
        "\u2022": "*", "\u2043": "-", "\u2219": "*", "\u25E6": "*",
        "\u00A0": " ", "\u2000": " ", "\u2001": " ", "\u2002": " ", "\u2003": " ",
        "\u2004": " ", "\u2005": " ", "\u2006": " ", "\u2007": " ", "\u2008": " ",
        "\u2009": " ", "\u200A": " ", "\u202F": " ", "\u205F": " ", "\u3000": " ",
        "\u200B": "", "\u200C": "", "\u200D": "", "\uFEFF": ""
    }
    for uni, asc in replacements.items():
        s = s.replace(uni, asc)
    s = re.sub(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000\u200C\u200D\uFEFF]", " ", s)
    s = "".join(c for c in s if c == "\n" or (c.isprintable() and unicodedata.category(c)[0] != "C"))
    return s

def load_book_texts(book_dir: str) -> str:
    """Load and clean all .txt files in the directory, concatenated into a single string."""
    text = []
    for file in sorted(glob.glob(os.path.join(book_dir, "*.txt"))):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("***") or line.lower().startswith("project gutenberg"):
                    continue
                cleaned = clean_text(line)
                if cleaned:
                    text.append(cleaned)
    return " ".join(text)

def split_text_by_tokens(
    tokenizer,
    text: str,
    prompt_size: int,
    system_prompt: str,
    num_samples: int
) -> Iterator[Dict[str, int]]:
    """
    Yield num_samples prompts of prompt_size tokens, wrapping text as needed,
    and only splitting at word boundaries. Ensures prompt is exactly prompt_size tokens,
    only exceeding if the final token is part of a multi-token word.
    """
    system_token_count = len(tokenizer.encode(system_prompt, add_special_tokens=False))
    if system_token_count >= prompt_size:
        raise ValueError("Prompt size too small for system prompt.")

    # Tokenize once, get input_ids and offsets
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    text_len = len(input_ids)

    # Precompute valid split points (token indices where the offset's start is at a word boundary)
    valid_starts = [0]
    for i in range(1, text_len):
        prev_end = offsets[i-1][1]
        curr_start = offsets[i][0]
        if prev_end == curr_start or text[prev_end:curr_start].isspace():
            valid_starts.append(i)

    for i in range(num_samples):
        # Find the initial start index
        start_idx = valid_starts[(i * (prompt_size - system_token_count)) % len(valid_starts)]
        # If start_idx is not a word boundary, backtrack to previous valid word boundary
        if start_idx not in valid_starts:
            # Find the closest previous valid start
            prev_starts = [s for s in valid_starts if s < start_idx]
            if prev_starts:
                start_idx = prev_starts[-1]
            else:
                start_idx = 0

        # Binary search for the largest chunk of user tokens that fits prompt_size
        left = 1
        right = min(prompt_size - system_token_count, text_len - start_idx)
        best_ids = None
        best_prompt_length = 0
        best_user_text = ""
        while left <= right:
            mid = (left + right) // 2
            end_idx = start_idx + mid
            if end_idx <= text_len:
                ids = input_ids[start_idx:end_idx]
            else:
                ids = input_ids[start_idx:] + input_ids[:end_idx - text_len]
            user_text = tokenizer.decode(ids)
            # Format prompt for model's tokenizer if possible
            if hasattr(tokenizer, "apply_chat_template"):
                # Use a single user message for "full" mode
                messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_text}"}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                prompt_length = len(tokenizer(prompt).input_ids)
            else:
                prompt = f"{system_prompt}\n\n{user_text}"
                prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
            if prompt_length <= prompt_size:
                best_ids = ids
                best_user_text = user_text
                best_prompt_length = prompt_length
                left = mid + 1
            else:
                right = mid - 1
        # If the next word would cause the prompt to exceed prompt_size and the word is multi-token, allow the overrun
        if best_ids is not None:
            end_idx = start_idx + len(best_ids)
            if end_idx < text_len:
                next_start = end_idx
                for j in range(next_start + 1, text_len + 1):
                    if j == text_len or (offsets[j-1][1] == offsets[j][0] or text[offsets[j-1][1]:offsets[j][0]].isspace()):
                        ids = input_ids[start_idx:j]
                        user_text = tokenizer.decode(ids)
                        if hasattr(tokenizer, "apply_chat_template"):
                            messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_text}"}]
                            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                            prompt_length = len(tokenizer(prompt).input_ids)
                        else:
                            prompt = f"{system_prompt}\n\n{user_text}"
                            prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
                        if prompt_length > prompt_size:
                            if len(ids) - len(best_ids) > 1:
                                best_ids = ids
                                best_user_text = user_text
                                best_prompt_length = prompt_length
                        break
        # Final prompt formatting
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{best_user_text}"}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            prompt = f"{system_prompt}\n\n{best_user_text}"
        yield {"prompt": prompt, "prompt_length": best_prompt_length}

def split_text_by_tokens_chat(
    tokenizer,
    text: str,
    prompt_size: int,
    system_prompt: str,
    num_samples: int
) -> Iterator[Dict[str, int]]:
    """
    Yield num_samples chat prompts of prompt_size tokens, wrapping text as needed,
    and only splitting at word boundaries. Ensures prompt is exactly prompt_size tokens,
    only exceeding if the final token is part of a multi-token word.
    """
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    text_len = len(input_ids)

    # Precompute valid split points (token indices where the offset's start is at a word boundary)
    valid_starts = [0]
    for i in range(1, text_len):
        prev_end = offsets[i-1][1]
        curr_start = offsets[i][0]
        if prev_end == curr_start or text[prev_end:curr_start].isspace():
            valid_starts.append(i)

    for i in range(num_samples):
        start_idx = valid_starts[(i * (prompt_size // 2)) % len(valid_starts)]
        # Find the chunk of user tokens that, when combined with system prompt, is as close as possible to prompt_size
        left = 1
        right = min(prompt_size, text_len - start_idx)
        best_ids = None
        best_prompt_length = 0
        best_user_text = ""
        while left <= right:
            mid = (left + right) // 2
            end_idx = start_idx + mid
            if end_idx <= text_len:
                ids = input_ids[start_idx:end_idx]
            else:
                ids = input_ids[start_idx:] + input_ids[:end_idx - text_len]
            user_text = tokenizer.decode(ids)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
            if hasattr(tokenizer, "apply_chat_template"):
                chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                chat_prompt = f"{system_prompt}\n\n{user_text}"
            prompt_length = len(tokenizer(chat_prompt).input_ids)
            if prompt_length <= prompt_size:
                best_ids = ids
                best_user_text = user_text
                best_prompt_length = prompt_length
                left = mid + 1
            else:
                right = mid - 1
        # If the next word would cause the prompt to exceed prompt_size and the word is multi-token, allow the overrun
        # Only if the last token is part of a multi-token word
        if best_ids is not None:
            # Check if the next word is multi-token and would cause an overrun
            end_idx = start_idx + len(best_ids)
            if end_idx < text_len:
                # Try to add the next word if it is multi-token
                next_start = end_idx
                # Find next word boundary
                for j in range(next_start + 1, text_len + 1):
                    if j == text_len or (offsets[j-1][1] == offsets[j][0] or text[offsets[j-1][1]:offsets[j][0]].isspace()):
                        ids = input_ids[start_idx:j]
                        user_text = tokenizer.decode(ids)
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_text}
                        ]
                        if hasattr(tokenizer, "apply_chat_template"):
                            chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                        else:
                            chat_prompt = f"{system_prompt}\n\n{user_text}"
                        prompt_length = len(tokenizer(chat_prompt).input_ids)
                        if prompt_length > prompt_size:
                            # Only allow if the last word is multi-token
                            if len(ids) - len(best_ids) > 1:
                                best_ids = ids
                                best_user_text = user_text
                                best_prompt_length = prompt_length
                        break
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": best_user_text}
        ]
        yield {"messages": messages, "prompt_length": best_prompt_length}

def generate_prompts(
    tokenizer_dir: str,
    book_dir: str,
    prompt_sizes: List[int],
    num_samples: int,
    mode: str,
    output_dir: str
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    text = load_book_texts(book_dir)
    os.makedirs(output_dir, exist_ok=True)
    for prompt_size in prompt_sizes:
        output_path = os.path.join(output_dir, f"prompts_{mode}_{prompt_size}_{num_samples}.jsonl")
        if mode == "full":
            generator = split_text_by_tokens(tokenizer, text, prompt_size, SYSTEM_PROMPT, num_samples)
        elif mode == "chat":
            generator = split_text_by_tokens_chat(tokenizer, text, prompt_size, SYSTEM_PROMPT, num_samples)
        else:
            raise ValueError("Invalid mode. Use 'full' or 'chat'.")
        with open(output_path, "w", encoding="utf-8") as out_f:
            for item in tqdm(generator, total=num_samples, desc=f"Generating {prompt_size}-token prompts"):
                item["prompt_size"] = prompt_size
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate sample prompts from public book texts.")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Path to tokenizer directory (e.g., tokenizers/llama3)")
    parser.add_argument("--book_dir", type=str, default="datasets/public_book_texts", help="Path to directory with public book .txt files")
    parser.add_argument("--prompt_sizes", type=int, nargs="+", required=True, help="List of prompt sizes in tokens (e.g., 500 1000 3000)")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples per prompt size to output")
    parser.add_argument("--mode", type=str, choices=["full", "chat"], default="full", help="Output mode: 'full' for single prompt, 'chat' for system/user messages")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to write output files")
    args = parser.parse_args()
    generate_prompts(
        tokenizer_dir=args.tokenizer_dir,
        book_dir=args.book_dir,
        prompt_sizes=args.prompt_sizes,
        num_samples=args.num_samples,
        mode=args.mode,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()