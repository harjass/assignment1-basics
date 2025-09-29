import os
import multiprocessing as mp
import regex as re
from typing import BinaryIO
from collections import Counter
from pprint import pprint

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
special_token = "<|endoftext|>"

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

def iter_tokens(text: str):
    for match in PAT.finditer(text):
        yield match.group()

def process_chunk(chunk: str, special_token: str):
    pair_counts = Counter()
    # 1. pre-tokenise each story
    stories = chunk.split(special_token)
    for story in stories:
        tokens = list(iter_tokens(story))   # materialise generator
        # 2. count adjacent TOKEN pairs inside this story
        for left, right in zip(tokens, tokens[1:]):
            pair_counts[(left, right)] += 1
    return pair_counts

def multi_chunk(args):
    filepath, start, end, special_token = args
    with open(filepath, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return process_chunk(chunk, special_token)

# Usage
if __name__ == "__main__":
    filepath = "/Users/maple/Documents/github-harjass/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    num_processes = 4

    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    tasks = [(filepath, start, end, special_token) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with mp.Pool(processes=num_processes) as pool:
        chunk_counter = pool.map(multi_chunk, tasks)

    total = Counter()
    for c in chunk_counter:
        total.update(c)
        
    pprint(total.most_common(10))
    pprint(total.most_common(1)[0])