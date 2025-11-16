"""
Codebase Semantic Search using Latent Trajectory Transformer
=============================================================

Search your entire codebase using natural language queries powered by
the Raccoon continual learning system. Goes beyond keyword matching
(grep/ripgrep) to understand semantic meaning.

Example queries:
- "where is attention implemented"
- "how does the ODE solver work"
- "memory buffer for experience replay"
- "normalizing flow coupling layers"

Architecture:
1. Code Tokenizer: 94-token vocabulary (letters, numbers, symbols)
2. File Chunker: Split files into overlapping 256-token windows
3. Latent Encoder: Raccoon model encodes chunks to 32-dim latent vectors
4. Faiss Index: Fast k-NN similarity search (cosine distance)
5. CLI Interface: Natural language query → top-k results with snippets

Author: Claude (Latent Trajectory Transformer Project)
Date: 2025-11-16
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle
import argparse
from dataclasses import dataclass

# Import Raccoon model
from latent_drift_trajectory import RaccoonLogClassifier, log_vocab_size


# ══════════════════════════════════════════════════════════════════════════════
# CODE TOKENIZER (94 tokens)
# ══════════════════════════════════════════════════════════════════════════════

# Extended vocabulary for code (vs log_vocab which is 39 tokens)
# Build vocabulary with sequential indices 0-93
CODE_VOCAB = {}

# Letters A-Z (indices 0-25)
for i, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    CODE_VOCAB[char] = i

# Letters a-z (indices 26-51)
for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
    CODE_VOCAB[char] = i + 26

# Numbers 0-9 (indices 52-61)
for i, char in enumerate('0123456789'):
    CODE_VOCAB[char] = i + 52

# Common programming symbols (indices 62-91)
symbols = '_./-(){ }[]:;,=+*#@!?<>&| \n\t"\'\\'
for i, char in enumerate(symbols):
    CODE_VOCAB[char] = i + 62

# Special tokens (indices 92-93)
CODE_VOCAB['<PAD>'] = 92
CODE_VOCAB['<UNK>'] = 93

# Reverse mapping (index → char)
REVERSE_CODE_VOCAB = {v: k for k, v in CODE_VOCAB.items()}

CODE_VOCAB_SIZE = len(CODE_VOCAB)  # 94 tokens


def tokenize_code(text: str, max_length: int = 256) -> torch.Tensor:
    """
    Tokenize code string to tensor of indices.

    Args:
        text: Source code string
        max_length: Maximum sequence length (truncate or pad)

    Returns:
        torch.Tensor: (max_length,) tensor of token indices
    """
    # Convert each character to index
    indices = []
    for char in text[:max_length]:
        indices.append(CODE_VOCAB.get(char, CODE_VOCAB['<UNK>']))

    # Pad if needed
    while len(indices) < max_length:
        indices.append(CODE_VOCAB['<PAD>'])

    return torch.tensor(indices, dtype=torch.long)


def decode_code(tokens: torch.Tensor) -> str:
    """
    Decode tensor of indices back to string.

    Args:
        tokens: (seq_len,) tensor of token indices

    Returns:
        Decoded string
    """
    chars = []
    for idx in tokens.tolist():
        if idx == CODE_VOCAB['<PAD>']:
            break  # Stop at padding
        chars.append(REVERSE_CODE_VOCAB.get(idx, '?'))

    return ''.join(chars)


# ══════════════════════════════════════════════════════════════════════════════
# FILE CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CodeChunk:
    """
    Represents a chunk of code from a file.
    """
    file_path: str
    start_line: int
    end_line: int
    text: str
    tokens: torch.Tensor  # (chunk_size,)


def chunk_file(
    file_path: str,
    chunk_size: int = 256,
    overlap: int = 64
) -> List[CodeChunk]:
    """
    Split file into overlapping chunks for indexing.

    Args:
        file_path: Path to source file
        chunk_size: Max tokens per chunk
        overlap: Overlap between chunks (provides context)

    Returns:
        List of CodeChunk objects
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return []

    chunks = []
    lines = content.split('\n')

    i = 0
    while i < len(lines):
        # Take chunk_size lines
        chunk_lines = lines[i:i + chunk_size]
        chunk_text = '\n'.join(chunk_lines)

        # Tokenize
        tokens = tokenize_code(chunk_text, max_length=chunk_size)

        chunks.append(CodeChunk(
            file_path=file_path,
            start_line=i + 1,
            end_line=i + len(chunk_lines),
            text=chunk_text,
            tokens=tokens
        ))

        # Slide window (with overlap)
        i += (chunk_size - overlap)

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# CODEBASE INDEX (Faiss-based similarity search)
# ══════════════════════════════════════════════════════════════════════════════

class CodebaseIndex:
    """
    Fast semantic search index for codebase using Faiss.
    """

    def __init__(self, latent_dim: int = 32):
        """
        Initialize empty index.

        Args:
            latent_dim: Dimension of latent vectors from model
        """
        self.latent_dim = latent_dim
        self.chunks: List[CodeChunk] = []
        self.vectors: Optional[np.ndarray] = None
        self.index = None

    def add_chunks(self, chunks: List[CodeChunk], vectors: np.ndarray):
        """
        Add encoded chunks to index.

        Args:
            chunks: List of CodeChunk objects
            vectors: (n_chunks, latent_dim) encoded latent vectors
        """
        assert len(chunks) == vectors.shape[0]
        assert vectors.shape[1] == self.latent_dim

        self.chunks.extend(chunks)

        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

        self._rebuild_index()

    def _rebuild_index(self):
        """
        Rebuild Faiss index for fast k-NN search.
        """
        try:
            import faiss
        except ImportError:
            print("Warning: Faiss not installed. Using slow numpy search.")
            self.index = None
            return

        # Normalize vectors for cosine similarity
        vectors_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8)

        # Use IndexFlatIP (inner product) for cosine similarity
        self.index = faiss.IndexFlatIP(self.latent_dim)
        self.index.add(vectors_norm.astype('float32'))

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[CodeChunk, float]]:
        """
        Find top-k most similar chunks to query.

        Args:
            query_vector: (latent_dim,) query latent vector
            top_k: Number of results to return

        Returns:
            List of (chunk, similarity_score) tuples
        """
        if len(self.chunks) == 0:
            return []

        # Normalize query
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        query_norm = query_norm.astype('float32').reshape(1, -1)

        if self.index is not None:
            # Fast Faiss search
            similarities, indices = self.index.search(query_norm, min(top_k, len(self.chunks)))
            similarities = similarities[0]
            indices = indices[0]
        else:
            # Slow numpy fallback
            vectors_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(vectors_norm, query_norm.T).squeeze()
            indices = np.argsort(similarities)[::-1][:top_k]
            similarities = similarities[indices]

        results = []
        for idx, sim in zip(indices, similarities):
            results.append((self.chunks[idx], float(sim)))

        return results

    def save(self, path: str):
        """
        Save index to disk.

        Args:
            path: File path to save (.pkl)
        """
        data = {
            'latent_dim': self.latent_dim,
            'chunks': self.chunks,
            'vectors': self.vectors,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'CodebaseIndex':
        """
        Load index from disk.

        Args:
            path: File path to load (.pkl)

        Returns:
            CodebaseIndex instance
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        index = cls(latent_dim=data['latent_dim'])
        index.chunks = data['chunks']
        index.vectors = data['vectors']
        index._rebuild_index()

        return index


# ══════════════════════════════════════════════════════════════════════════════
# CODEBASE SEARCH ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class CodebaseSearchEngine:
    """
    Complete semantic search engine for codebases.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        index_path: Optional[str] = None,
        latent_dim: int = 32,
        device: str = 'cpu'
    ):
        """
        Initialize search engine.

        Args:
            model_path: Path to trained Raccoon model (if None, create new)
            index_path: Path to saved index (if None, create new)
            latent_dim: Latent dimension for model
            device: 'cpu' or 'cuda'
        """
        self.latent_dim = latent_dim
        self.device = torch.device(device)

        # Load or create model
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        else:
            print("Creating new Raccoon model for codebase embedding...")
            # Use same architecture as log classifier but with code vocab
            self.model = RaccoonLogClassifier(
                vocab_size=CODE_VOCAB_SIZE,  # 94 tokens for code
                num_classes=10,  # Dummy (not used for embedding)
                latent_dim=latent_dim,
                hidden_dim=64,
                embed_dim=32,
                memory_size=1000,
            ).to(self.device)

        self.model.eval()

        # Load or create index
        if index_path and os.path.exists(index_path):
            print(f"Loading index from {index_path}...")
            self.index = CodebaseIndex.load(index_path)
        else:
            print("Creating new empty index...")
            self.index = CodebaseIndex(latent_dim=latent_dim)

    def encode_chunk(self, chunk: CodeChunk) -> np.ndarray:
        """
        Encode code chunk to latent vector.

        Args:
            chunk: CodeChunk to encode

        Returns:
            (latent_dim,) numpy array
        """
        with torch.no_grad():
            tokens = chunk.tokens.unsqueeze(0).to(self.device)  # (1, seq_len)

            # Encode to latent
            mean, logvar = self.model.encode(tokens)
            # Use mean (deterministic) for indexing
            latent = mean.squeeze(0).cpu().numpy()  # (latent_dim,)

        return latent

    def index_file(self, file_path: str, chunk_size: int = 256, overlap: int = 64):
        """
        Index a single file.

        Args:
            file_path: Path to file to index
            chunk_size: Tokens per chunk
            overlap: Overlap between chunks
        """
        # Chunk file
        chunks = chunk_file(file_path, chunk_size, overlap)

        if len(chunks) == 0:
            return

        # Encode all chunks
        vectors = []
        for chunk in chunks:
            vector = self.encode_chunk(chunk)
            vectors.append(vector)

        vectors = np.array(vectors)  # (n_chunks, latent_dim)

        # Add to index
        self.index.add_chunks(chunks, vectors)

    def index_directory(
        self,
        directory: str,
        extensions: List[str] = ['.py', '.md', '.txt'],
        exclude_dirs: List[str] = ['.git', '__pycache__', 'node_modules', '.venv'],
        chunk_size: int = 256,
        overlap: int = 64
    ):
        """
        Index all files in directory recursively.

        Args:
            directory: Root directory to index
            extensions: File extensions to include
            exclude_dirs: Directory names to skip
            chunk_size: Tokens per chunk
            overlap: Overlap between chunks
        """
        print(f"\n🔍 Indexing directory: {directory}")
        print(f"   Extensions: {extensions}")
        print(f"   Excluding: {exclude_dirs}")
        print()

        files_indexed = 0
        chunks_total = 0

        for root, dirs, files in os.walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                # Check extension
                if not any(file.endswith(ext) for ext in extensions):
                    continue

                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, directory)

                print(f"   Indexing: {rel_path}...", end=' ')

                chunks_before = len(self.index.chunks)
                self.index_file(file_path, chunk_size, overlap)
                chunks_added = len(self.index.chunks) - chunks_before

                print(f"({chunks_added} chunks)")

                files_indexed += 1
                chunks_total += chunks_added

        print()
        print(f"✅ Indexed {files_indexed} files → {chunks_total} chunks")
        print()

    def search(
        self,
        query: str,
        top_k: int = 10,
        chunk_size: int = 256
    ) -> List[Tuple[CodeChunk, float]]:
        """
        Search codebase with natural language query.

        Args:
            query: Natural language query string
            top_k: Number of results to return
            chunk_size: Size for query tokenization

        Returns:
            List of (chunk, similarity_score) tuples
        """
        # Tokenize query
        query_tokens = tokenize_code(query, max_length=chunk_size)

        # Encode query to latent
        with torch.no_grad():
            query_tokens_batch = query_tokens.unsqueeze(0).to(self.device)  # (1, chunk_size)
            query_mean, query_logvar = self.model.encode(query_tokens_batch)
            query_vector = query_mean.squeeze(0).cpu().numpy()  # (latent_dim,)

        # Search index
        results = self.index.search(query_vector, top_k)

        return results

    def save(self, model_path: str, index_path: str):
        """
        Save model and index to disk.

        Args:
            model_path: Path to save model (.pt)
            index_path: Path to save index (.pkl)
        """
        print(f"Saving model to {model_path}...")
        torch.save(self.model, model_path)

        print(f"Saving index to {index_path}...")
        self.index.save(index_path)

        print("✅ Saved successfully")


# ══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def display_results(results: List[Tuple[CodeChunk, float]], max_lines: int = 10):
    """
    Display search results in a nice format.

    Args:
        results: List of (chunk, similarity) tuples
        max_lines: Max lines to show per result
    """
    if len(results) == 0:
        print("No results found.")
        return

    for i, (chunk, similarity) in enumerate(results, 1):
        print(f"\n{i}. {chunk.file_path}:{chunk.start_line}-{chunk.end_line} "
              f"(similarity: {similarity:.3f})")
        print("─" * 80)

        # Show snippet
        lines = chunk.text.split('\n')[:max_lines]
        for j, line in enumerate(lines, chunk.start_line):
            print(f"  {j:4d} | {line}")

        chunk_lines = chunk.text.split('\n')
        if len(chunk_lines) > max_lines:
            remaining = len(chunk_lines) - max_lines
            print(f"  ... ({remaining} more lines)")


def main_cli():
    """
    Command-line interface for codebase search.
    """
    parser = argparse.ArgumentParser(
        description='Semantic codebase search using Latent Trajectory Transformer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index current directory
  python codebase_search.py --index .

  # Search for attention implementation
  python codebase_search.py "multi-head attention with causal masking"

  # Search with more results
  python codebase_search.py "ODE solver" --top-k 20

  # Index and then search
  python codebase_search.py --index . "experience replay memory buffer"
        """
    )

    parser.add_argument('query', nargs='?', help='Search query (natural language)')
    parser.add_argument('--index', type=str, metavar='DIR',
                        help='Index directory before searching')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of results to return (default: 10)')
    parser.add_argument('--model', type=str, default='codebase_model.pt',
                        help='Model file path (default: codebase_model.pt)')
    parser.add_argument('--index-file', type=str, default='codebase.index',
                        help='Index file path (default: codebase.index)')
    parser.add_argument('--extensions', type=str, default='.py,.md,.txt',
                        help='File extensions to index (comma-separated, default: .py,.md,.txt)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda, default: cpu)')

    args = parser.parse_args()

    # Initialize engine
    engine = CodebaseSearchEngine(
        model_path=args.model if os.path.exists(args.model) else None,
        index_path=args.index_file if os.path.exists(args.index_file) else None,
        device=args.device
    )

    # Index directory if requested
    if args.index:
        extensions = [ext.strip() for ext in args.extensions.split(',')]
        engine.index_directory(args.index, extensions=extensions)

        # Save index
        engine.save(args.model, args.index_file)

    # Search if query provided
    if args.query:
        if len(engine.index.chunks) == 0:
            print("Error: Index is empty. Use --index DIR to index a directory first.")
            return 1

        print(f"\n🔍 Searching for: \"{args.query}\"")
        print(f"   Indexed chunks: {len(engine.index.chunks)}")
        print("=" * 80)

        results = engine.search(args.query, top_k=args.top_k)
        display_results(results)

    elif not args.index:
        # No index and no query - show help
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    exit(main_cli())
