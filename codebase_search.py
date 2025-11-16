#!/usr/bin/env python3
"""
LATENT TRAJECTORY CODEBASE SEARCH
==================================

Semantic code search using the latent trajectory transformer.
Encodes code snippets into time-dependent latent representations
for similarity-based search across your codebase.

Key Innovation:
- Code → latent trajectory (constant-length representation)
- Semantic similarity via latent space distance
- Works on CPU (single core optimized)
- Handles long code sequences without quadratic growth

Usage:
    python codebase_search.py index <directory>     # Index codebase
    python codebase_search.py search "<query>"      # Search by description
    python codebase_search.py similar <file>:<line> # Find similar code
"""

import os
import sys
import math
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
#  CODE TOKENIZATION (Extended character set for Python/code)
# ──────────────────────────────────────────────────────────────────────────────

# Extended vocabulary for code: standard ASCII printable + special tokens
CODE_SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<START>", "<END>", "<NEWLINE>"]
# Printable ASCII: space (32) to ~ (126)
CODE_CHARS = CODE_SPECIAL_TOKENS + [chr(i) for i in range(32, 127)]
CODE_CHAR2IDX = {ch: i for i, ch in enumerate(CODE_CHARS)}
CODE_IDX2CHAR = {i: ch for ch, i in CODE_CHAR2IDX.items()}
CODE_VOCAB_SIZE = len(CODE_CHARS)

# Token IDs
PAD_ID = CODE_CHAR2IDX["<PAD>"]
UNK_ID = CODE_CHAR2IDX["<UNK>"]
START_ID = CODE_CHAR2IDX["<START>"]
END_ID = CODE_CHAR2IDX["<END>"]
NEWLINE_ID = CODE_CHAR2IDX["<NEWLINE>"]


def encode_code(code: str, max_len: int = 256) -> Tensor:
    """Encode code string to token IDs."""
    # Replace actual newlines with special token
    code = code.replace('\n', '<NEWLINE>')

    # Encode characters
    tokens = []
    for ch in code[:max_len-2]:  # Reserve space for START/END
        tokens.append(CODE_CHAR2IDX.get(ch, UNK_ID))

    # Add START/END tokens
    tokens = [START_ID] + tokens + [END_ID]

    # Pad to max_len
    while len(tokens) < max_len:
        tokens.append(PAD_ID)

    return torch.tensor(tokens[:max_len], dtype=torch.long)


def decode_code(tokens: Tensor) -> str:
    """Decode token IDs back to code string."""
    chars = []
    for idx in tokens:
        idx_val = int(idx)
        if idx_val == PAD_ID or idx_val == END_ID:
            break
        if idx_val == START_ID:
            continue
        ch = CODE_IDX2CHAR.get(idx_val, '?')
        chars.append(ch)

    # Replace special newline token back
    code = ''.join(chars).replace('<NEWLINE>', '\n')
    return code


# ──────────────────────────────────────────────────────────────────────────────
#  LIGHTWEIGHT TRANSFORMER ENCODER (CPU-optimized)
# ──────────────────────────────────────────────────────────────────────────────

class AddPositionalEncoding(nn.Module):
    def __init__(self, len_max: float = 1e4):
        super().__init__()
        self.len_max = len_max

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, C)
        u = torch.arange(x.size(1), device=x.device)[:, None]
        j = torch.arange(x.size(2), device=x.device)[None, :]
        k = j % 2
        t = u / (self.len_max ** ((j - k) / x.size(2))) + math.pi / 2 * k
        return x + torch.sin(t)


class QKVAttention(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, nb_heads=1, causal=False, dropout=0.0):
        super().__init__()

        def randw(*d):
            return nn.Parameter(torch.randn(*d) / math.sqrt(d[-1]))

        self.causal = causal
        self.dropout = dropout

        self.w_q = randw(nb_heads, dim_qk, dim_in)
        self.w_k = randw(nb_heads, dim_qk, dim_in)
        self.w_v = randw(nb_heads, dim_v, dim_in)
        self.w_o = randw(dim_v * nb_heads, dim_in)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, T, C)
        q = torch.einsum("ntc,hdc->nhtd", x, self.w_q)
        k = torch.einsum("ntc,hdc->nhtd", x, self.w_k)
        v = torch.einsum("ntc,hdc->nhtd", x, self.w_v)

        a = torch.einsum("nhtd,nhsd->nhts", q, k) / math.sqrt(self.w_q.size(1))

        if self.causal:
            t = torch.arange(x.size(1), device=x.device)
            attzero = t[None, None, :, None] < t[None, None, None, :]
            a = a.masked_fill(attzero, float("-inf"))

        a = a.softmax(dim=3)
        a = F.dropout(a, self.dropout, self.training)
        y = torch.einsum("nhts,nhsd->nthd", a, v).flatten(2)

        y = y @ self.w_o
        return y


class TransformerBlock(nn.Module):
    def __init__(self, dim_model, dim_keys, dim_hidden, nb_heads, causal, dropout):
        super().__init__()
        self.att_ln = nn.LayerNorm((dim_model,))
        self.att_mh = QKVAttention(
            dim_in=dim_model,
            dim_qk=dim_keys,
            dim_v=dim_model // nb_heads,
            nb_heads=nb_heads,
            causal=causal,
            dropout=dropout,
        )
        self.ffn_ln = nn.LayerNorm((dim_model,))
        self.ffn_fc1 = nn.Linear(dim_model, dim_hidden)
        self.ffn_fc2 = nn.Linear(dim_hidden, dim_model)

    def forward(self, x: Tensor) -> Tensor:
        r = x
        x = self.att_ln(r)
        x = self.att_mh(x)
        r = r + x

        x = self.ffn_ln(r)
        x = self.ffn_fc1(x)
        x = F.relu(x)
        x = self.ffn_fc2(x)
        r = r + x

        return r


class CodeEncoder(nn.Module):
    """
    Lightweight transformer encoder for code.
    Optimized for CPU with reduced parameters.
    """
    def __init__(
        self,
        vocab_size: int = CODE_VOCAB_SIZE,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)

        if embed_dim != hidden_dim:
            self.in_proj = nn.Linear(embed_dim, hidden_dim)
        else:
            self.in_proj = nn.Identity()

        self.pos_enc = AddPositionalEncoding(len_max=1e4)

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim_model=hidden_dim,
                dim_keys=hidden_dim // num_heads,
                dim_hidden=hidden_dim,
                nb_heads=num_heads,
                causal=False,  # Bidirectional for encoding
                dropout=0.0,
            )
            for _ in range(num_blocks)
        ])

        # Project to latent space
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Encode tokens to latent trajectory.

        Args:
            tokens: (B, L) token IDs
        Returns:
            latent: (B, latent_dim) - mean-pooled latent representation
        """
        x = self.emb(tokens)          # (B, L, E)
        x = self.in_proj(x)           # (B, L, H)
        x = self.pos_enc(x)           # (B, L, H)
        x = self.blocks(x)            # (B, L, H)

        # Create mask for padding
        mask = (tokens != PAD_ID).float().unsqueeze(-1)  # (B, L, 1)

        # Mean pooling (ignoring padding)
        x = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # (B, H)

        # Project to latent
        z = self.latent_proj(x)  # (B, latent_dim)

        return z


# ──────────────────────────────────────────────────────────────────────────────
#  CODE SNIPPET DATACLASS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CodeSnippet:
    """Represents a code snippet with metadata."""
    file_path: str
    start_line: int
    end_line: int
    code: str
    language: str
    embedding: Optional[List[float]] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'CodeSnippet':
        return cls(**d)

    def __repr__(self):
        return f"<CodeSnippet {self.file_path}:{self.start_line}-{self.end_line}>"


# ──────────────────────────────────────────────────────────────────────────────
#  CODE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_code_snippets(
    directory: Path,
    extensions: List[str] = ['.py', '.js', '.java', '.cpp', '.c', '.h'],
    snippet_lines: int = 20,
    overlap: int = 5,
) -> List[CodeSnippet]:
    """
    Extract overlapping code snippets from directory.

    Args:
        directory: Root directory to search
        extensions: File extensions to process
        snippet_lines: Lines per snippet
        overlap: Overlap between consecutive snippets

    Returns:
        List of CodeSnippet objects
    """
    snippets = []

    # Find all code files
    code_files = []
    for ext in extensions:
        code_files.extend(directory.rglob(f'*{ext}'))

    print(f"📁 Found {len(code_files)} code files")

    for file_path in tqdm(code_files, desc="Extracting snippets"):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            # Extract language from extension
            lang = file_path.suffix[1:]  # Remove leading dot

            # Create overlapping snippets
            stride = snippet_lines - overlap
            for i in range(0, len(lines), stride):
                chunk = lines[i:i + snippet_lines]
                if len(chunk) < snippet_lines // 2:  # Skip very short chunks
                    continue

                code = ''.join(chunk)

                # Skip empty or whitespace-only chunks
                if not code.strip():
                    continue

                snippet = CodeSnippet(
                    file_path=str(file_path.relative_to(directory)),
                    start_line=i + 1,
                    end_line=i + len(chunk),
                    code=code,
                    language=lang,
                )
                snippets.append(snippet)

        except Exception as e:
            print(f"⚠️  Error processing {file_path}: {e}")
            continue

    print(f"✅ Extracted {len(snippets)} code snippets")
    return snippets


# ──────────────────────────────────────────────────────────────────────────────
#  SEARCH INDEX
# ──────────────────────────────────────────────────────────────────────────────

class CodeSearchIndex:
    """
    Search index using latent trajectory embeddings.
    Optimized for CPU with efficient nearest neighbor search.
    """
    def __init__(
        self,
        model: CodeEncoder,
        snippets: List[CodeSnippet],
        embeddings: Optional[Tensor] = None,
    ):
        self.model = model
        self.snippets = snippets
        self.embeddings = embeddings  # (N, latent_dim) tensor

    def build_index(self, batch_size: int = 32, device: str = 'cpu'):
        """Compute embeddings for all snippets."""
        print(f"\n🔨 Building search index...")

        self.model.eval()
        self.model.to(device)

        all_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(self.snippets), batch_size), desc="Encoding"):
                batch = self.snippets[i:i+batch_size]

                # Tokenize batch
                tokens = torch.stack([
                    encode_code(s.code) for s in batch
                ]).to(device)

                # Encode
                z = self.model(tokens)  # (B, latent_dim)
                all_embeddings.append(z.cpu())

        self.embeddings = torch.cat(all_embeddings, dim=0)  # (N, latent_dim)
        print(f"✅ Index built: {self.embeddings.shape[0]} embeddings")

    def search(
        self,
        query: str,
        top_k: int = 10,
        device: str = 'cpu',
    ) -> List[Tuple[CodeSnippet, float]]:
        """
        Search for code similar to query.

        Args:
            query: Code or natural language query
            top_k: Number of results to return
            device: Device to run inference on

        Returns:
            List of (snippet, similarity_score) tuples
        """
        if self.embeddings is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        self.model.eval()
        self.model.to(device)

        # Encode query
        with torch.no_grad():
            query_tokens = encode_code(query).unsqueeze(0).to(device)  # (1, L)
            query_emb = self.model(query_tokens)  # (1, latent_dim)

        # Compute cosine similarity
        query_emb = query_emb.cpu()
        query_norm = F.normalize(query_emb, p=2, dim=-1)
        embeddings_norm = F.normalize(self.embeddings, p=2, dim=-1)

        similarities = (query_norm @ embeddings_norm.T).squeeze(0)  # (N,)

        # Get top-k
        top_indices = similarities.argsort(descending=True)[:top_k]

        results = []
        for idx in top_indices:
            idx_val = int(idx)
            snippet = self.snippets[idx_val]
            score = float(similarities[idx])
            results.append((snippet, score))

        return results

    def save(self, path: Path):
        """Save index to disk."""
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(self.model.state_dict(), path / "model.pt")

        # Save embeddings
        torch.save(self.embeddings, path / "embeddings.pt")

        # Save snippets (without embeddings to save space)
        snippets_dict = [s.to_dict() for s in self.snippets]
        with open(path / "snippets.json", 'w') as f:
            json.dump(snippets_dict, f, indent=2)

        print(f"💾 Index saved to {path}")

    @classmethod
    def load(cls, path: Path, model: CodeEncoder) -> 'CodeSearchIndex':
        """Load index from disk."""
        # Load model weights
        model.load_state_dict(torch.load(path / "model.pt", map_location='cpu'))

        # Load embeddings
        embeddings = torch.load(path / "embeddings.pt", map_location='cpu')

        # Load snippets
        with open(path / "snippets.json", 'r') as f:
            snippets_dict = json.load(f)
        snippets = [CodeSnippet.from_dict(d) for d in snippets_dict]

        print(f"📂 Index loaded from {path}")
        return cls(model, snippets, embeddings)


# ──────────────────────────────────────────────────────────────────────────────
#  CLI INTERFACE
# ──────────────────────────────────────────────────────────────────────────────

def cmd_index(args):
    """Index a codebase directory."""
    directory = Path(args.directory)
    index_path = Path(args.output) if args.output else Path(".code_index")

    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return

    print(f"🔍 Indexing codebase: {directory}")

    # Extract snippets
    snippets = extract_code_snippets(
        directory,
        extensions=args.extensions.split(','),
        snippet_lines=args.snippet_lines,
        overlap=args.overlap,
    )

    if not snippets:
        print("❌ No code snippets found")
        return

    # Initialize model
    print("\n🤖 Initializing encoder model...")
    model = CodeEncoder(
        vocab_size=CODE_VOCAB_SIZE,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {param_count:,}")

    # Build index
    index = CodeSearchIndex(model, snippets)
    index.build_index(batch_size=args.batch_size, device='cpu')

    # Save
    index.save(index_path)

    print(f"\n🎉 Indexing complete!")
    print(f"📦 Snippets: {len(snippets)}")
    print(f"💾 Index saved to: {index_path}")


def cmd_search(args):
    """Search the indexed codebase."""
    index_path = Path(args.index) if args.index else Path(".code_index")

    if not index_path.exists():
        print(f"❌ Index not found: {index_path}")
        print("   Run 'index' command first to create an index")
        return

    # Load index
    print(f"📂 Loading index from {index_path}...")
    model = CodeEncoder(
        vocab_size=CODE_VOCAB_SIZE,
        embed_dim=64,
        hidden_dim=128,
        latent_dim=64,
        num_blocks=2,
        num_heads=2,
    )
    index = CodeSearchIndex.load(index_path, model)

    # Search
    query = args.query
    print(f"\n🔎 Query: {query}")
    print(f"=" * 80)

    results = index.search(query, top_k=args.top_k, device='cpu')

    # Display results
    for i, (snippet, score) in enumerate(results, 1):
        print(f"\n[{i}] {snippet.file_path}:{snippet.start_line}-{snippet.end_line}")
        print(f"    Similarity: {score:.3f}")
        print(f"    Language: {snippet.language}")

        if args.show_code:
            print(f"\n{snippet.code}")
            print("-" * 80)


def cmd_similar(args):
    """Find code similar to a specific location."""
    index_path = Path(args.index) if args.index else Path(".code_index")

    if not index_path.exists():
        print(f"❌ Index not found: {index_path}")
        return

    # Parse location (file:line)
    try:
        file_path, line = args.location.rsplit(':', 1)
        line = int(line)
    except:
        print(f"❌ Invalid location format. Use: <file>:<line>")
        return

    # Load index
    model = CodeEncoder()
    index = CodeSearchIndex.load(index_path, model)

    # Find snippet at location
    query_snippet = None
    for snippet in index.snippets:
        if snippet.file_path == file_path and snippet.start_line <= line <= snippet.end_line:
            query_snippet = snippet
            break

    if not query_snippet:
        print(f"❌ No snippet found at {file_path}:{line}")
        return

    print(f"\n📍 Finding code similar to:")
    print(f"   {query_snippet.file_path}:{query_snippet.start_line}-{query_snippet.end_line}")
    print(f"=" * 80)

    # Search using snippet code
    results = index.search(query_snippet.code, top_k=args.top_k + 1, device='cpu')

    # Skip the first result (itself)
    results = results[1:args.top_k + 1]

    for i, (snippet, score) in enumerate(results, 1):
        print(f"\n[{i}] {snippet.file_path}:{snippet.start_line}-{snippet.end_line}")
        print(f"    Similarity: {score:.3f}")

        if args.show_code:
            print(f"\n{snippet.code}")
            print("-" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Latent Trajectory Codebase Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Index command
    parser_index = subparsers.add_parser('index', help='Index a codebase')
    parser_index.add_argument('directory', help='Directory to index')
    parser_index.add_argument('--output', '-o', help='Output index directory (default: .code_index)')
    parser_index.add_argument('--extensions', default='.py,.js,.java,.cpp,.c,.h',
                             help='File extensions to index (comma-separated)')
    parser_index.add_argument('--snippet-lines', type=int, default=20,
                             help='Lines per snippet')
    parser_index.add_argument('--overlap', type=int, default=5,
                             help='Overlap between snippets')
    parser_index.add_argument('--embed-dim', type=int, default=64,
                             help='Embedding dimension')
    parser_index.add_argument('--hidden-dim', type=int, default=128,
                             help='Hidden dimension')
    parser_index.add_argument('--latent-dim', type=int, default=64,
                             help='Latent dimension')
    parser_index.add_argument('--num-blocks', type=int, default=2,
                             help='Number of transformer blocks')
    parser_index.add_argument('--num-heads', type=int, default=2,
                             help='Number of attention heads')
    parser_index.add_argument('--batch-size', type=int, default=32,
                             help='Batch size for encoding')

    # Search command
    parser_search = subparsers.add_parser('search', help='Search the codebase')
    parser_search.add_argument('query', help='Search query (code or description)')
    parser_search.add_argument('--index', '-i', help='Index directory (default: .code_index)')
    parser_search.add_argument('--top-k', '-k', type=int, default=10,
                              help='Number of results')
    parser_search.add_argument('--show-code', '-c', action='store_true',
                              help='Show full code snippets')

    # Similar command
    parser_similar = subparsers.add_parser('similar', help='Find similar code')
    parser_similar.add_argument('location', help='File location (file:line)')
    parser_similar.add_argument('--index', '-i', help='Index directory (default: .code_index)')
    parser_similar.add_argument('--top-k', '-k', type=int, default=10,
                               help='Number of results')
    parser_similar.add_argument('--show-code', '-c', action='store_true',
                               help='Show full code snippets')

    args = parser.parse_args()

    if args.command == 'index':
        cmd_index(args)
    elif args.command == 'search':
        cmd_search(args)
    elif args.command == 'similar':
        cmd_similar(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
