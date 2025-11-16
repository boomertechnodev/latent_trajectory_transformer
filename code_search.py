#!/usr/bin/env python3
"""
Latent Code Search - Search your codebase using trajectory planning in latent space.

This is a practical application of the latent trajectory transformer that:
1. Tokenizes code at the character level
2. Learns latent trajectory representations incrementally (continual learning)
3. Finds relevant code chunks using trajectory similarity
4. Runs efficiently on CPU (single-core)

Usage:
    # Index a repository
    python code_search.py index /path/to/repo --output repo.index

    # Search
    python code_search.py query "where is the SDE dynamics?" --index repo.index
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Iterator, Dict, Optional
import pickle
import argparse
from tqdm import tqdm
import re

from latent_drift_trajectory import (
    RaccoonDynamics,
    RaccoonFlow,
    RaccoonMemory,
    solve_sde,
    SlicingUnivariateTest,
    FastEppsPulley,
)


# ============================================================================
# Component 1: Code Tokenizer & Preprocessor
# ============================================================================

class CodeTokenizer:
    """Simple character-level tokenizer for code.

    Vocabulary: 256 ASCII characters (0-255)
    """

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.pad_token = 0

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        return [min(ord(c), self.vocab_size - 1) for c in text]

    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs back to text."""
        return ''.join(chr(t) for t in tokens if t < self.vocab_size)

    def chunk(self, tokens: List[int], window: int = 512, stride: int = 256) -> Iterator[List[int]]:
        """Chunk tokens into overlapping windows."""
        for i in range(0, len(tokens), stride):
            chunk = tokens[i:i + window]
            # Pad if needed
            if len(chunk) < window:
                chunk = chunk + [self.pad_token] * (window - len(chunk))
            yield chunk

    def pad_or_truncate(self, tokens: List[int], length: int) -> List[int]:
        """Ensure tokens are exactly the specified length."""
        if len(tokens) < length:
            return tokens + [self.pad_token] * (length - len(tokens))
        else:
            return tokens[:length]


class CodeChunk:
    """Represents a chunk of code with metadata."""

    def __init__(self, tokens: List[int], file_path: Path, start_line: int, end_line: int, metadata: Dict):
        self.tokens = tokens
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.metadata = metadata  # Function names, class names, imports, etc.


class CodebaseCrawler:
    """Crawls a repository and extracts code chunks."""

    def __init__(self, tokenizer: CodeTokenizer, window: int = 512, stride: int = 256):
        self.tokenizer = tokenizer
        self.window = window
        self.stride = stride

    def extract_metadata(self, code: str) -> Dict:
        """Extract metadata from code using simple regex patterns.

        This is a simple implementation - could be enhanced with AST parsing.
        """
        metadata = {
            'functions': [],
            'classes': [],
            'imports': [],
        }

        # Find function definitions
        func_pattern = r'def\s+(\w+)\s*\('
        metadata['functions'] = re.findall(func_pattern, code)

        # Find class definitions
        class_pattern = r'class\s+(\w+)\s*[\(:]'
        metadata['classes'] = re.findall(class_pattern, code)

        # Find imports
        import_pattern = r'(?:from\s+[\w.]+\s+)?import\s+([\w,\s.]+)'
        metadata['imports'] = re.findall(import_pattern, code)

        return metadata

    def crawl(self, repo_path: Path, pattern: str = "*.py") -> Iterator[CodeChunk]:
        """Crawl repository and yield code chunks."""
        repo_path = Path(repo_path)

        for file_path in repo_path.rglob(pattern):
            try:
                code = file_path.read_text(encoding='utf-8')
            except (UnicodeDecodeError, PermissionError):
                continue

            # Extract metadata
            metadata = self.extract_metadata(code)

            # Tokenize
            tokens = self.tokenizer.encode(code)

            # Calculate lines per character (for line number mapping)
            lines = code.split('\n')
            chars_per_line = [len(line) + 1 for line in lines]  # +1 for newline
            cumulative_chars = [sum(chars_per_line[:i+1]) for i in range(len(chars_per_line))]

            # Chunk
            chunk_idx = 0
            for chunk_tokens in self.tokenizer.chunk(tokens, self.window, self.stride):
                # Calculate line numbers for this chunk
                start_char = chunk_idx * self.stride
                end_char = start_char + len(chunk_tokens)

                # Find corresponding lines
                start_line = next((i for i, c in enumerate(cumulative_chars) if c > start_char), 0)
                end_line = next((i for i, c in enumerate(cumulative_chars) if c >= end_char), len(lines))

                yield CodeChunk(
                    tokens=chunk_tokens,
                    file_path=file_path,
                    start_line=start_line + 1,  # 1-indexed
                    end_line=end_line + 1,
                    metadata=metadata,
                )

                chunk_idx += 1


# ============================================================================
# Component 2: Code Search Model (Adapted from RaccoonLogClassifier)
# ============================================================================

class CodeSearchModel(nn.Module):
    """Latent trajectory model for code search.

    Key differences from RaccoonLogClassifier:
    - No classification head (we want embeddings, not predictions)
    - Loss is contrastive + trajectory smoothness (not cross-entropy)
    - Designed for similarity search, not classification
    """

    def __init__(
        self,
        vocab_size: int = 256,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        seq_len: int = 512,
        memory_size: int = 2000,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # Encoder: tokens -> latent distribution
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(seq_len * embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # VAE latent projection
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # SDE dynamics
        self.dynamics = RaccoonDynamics(latent_dim, hidden_dim)

        # Normalizing flow
        self.flow = RaccoonFlow(latent_dim, hidden_dim, num_layers=4)

        # Experience replay memory
        self.memory = RaccoonMemory(max_size=memory_size)

        # Statistical regularization (EP test for latent smoothness)
        univariate_test = FastEppsPulley(t_max=5.0, n_points=17)
        self.latent_test = SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=256
        )

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def encode(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode tokens to latent distribution.

        Args:
            tokens: (batch, seq_len)

        Returns:
            mean: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        # Embed
        x = self.embedding(tokens)  # (batch, seq_len, embed_dim)
        x = x.flatten(1)  # (batch, seq_len * embed_dim)

        # Encode
        h = self.encoder(x)  # (batch, hidden_dim)

        # Project to latent
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)

        return mean, logvar

    def sample_latent(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def get_trajectory(self, z0: torch.Tensor, t_span: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate latent trajectory using SDE dynamics.

        Args:
            z0: (batch, latent_dim) - initial latent state
            t_span: (num_steps,) - time points (default: [0.0, 0.1] with 3 steps)

        Returns:
            trajectory: (batch, num_steps, latent_dim)
        """
        if t_span is None:
            t_span = torch.linspace(0.0, 0.1, 3, device=z0.device)

        return solve_sde(self.dynamics, z0, t_span)

    def get_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """Get final embedding for a code chunk (for similarity search).

        This is the main interface for search - returns a single vector per chunk.

        Args:
            tokens: (batch, seq_len)

        Returns:
            embedding: (batch, latent_dim) - final trajectory state after flow
        """
        # Encode to latent
        mean, logvar = self.encode(tokens)
        z0 = self.sample_latent(mean, logvar)

        # Get trajectory
        t_span = torch.linspace(0.0, 0.1, 3, device=tokens.device)
        z_traj = self.get_trajectory(z0, t_span)

        # Apply flow to final state
        z_final = z_traj[:, -1]
        t_final = t_span[-1:].expand(z_final.size(0)).unsqueeze(1)  # (batch, 1)
        z_flow, _ = self.flow(z_final, t_final)

        return z_flow

    def compute_loss(
        self,
        tokens: torch.Tensor,
        loss_weights: Tuple[float, float] = (1.0, 0.1),
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute training loss (for indexing phase).

        Loss components:
        1. KL divergence (regularize latent to be Gaussian)
        2. EP test (regularize trajectory to be smooth/normal)

        Args:
            tokens: (batch, seq_len)
            loss_weights: (kl_weight, ep_weight)

        Returns:
            loss: scalar
            metrics: dict with loss components
        """
        kl_weight, ep_weight = loss_weights

        # Encode
        mean, logvar = self.encode(tokens)
        z0 = self.sample_latent(mean, logvar)

        # KL divergence: KL(q(z|x) || N(0,I))
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()

        # Get trajectory
        t_span = torch.linspace(0.0, 0.1, 3, device=tokens.device)
        z_traj = self.get_trajectory(z0, t_span)

        # EP test on trajectory (regularize to be smooth/normal)
        z_traj_flat = z_traj.reshape(-1, self.latent_dim)  # (batch * steps, latent_dim)
        if z_traj_flat.size(0) > 1:  # Need at least 2 samples
            ep_loss = self.latent_test(z_traj_flat)
        else:
            ep_loss = torch.tensor(0.0, device=tokens.device)

        # Total loss
        loss = kl_weight * kl_loss + ep_weight * ep_loss

        metrics = {
            'loss': loss.item(),
            'kl_loss': kl_loss.item(),
            'ep_loss': ep_loss.item() if isinstance(ep_loss, torch.Tensor) else ep_loss,
        }

        return loss, metrics


# ============================================================================
# Component 3: Indexing (Build search index from codebase)
# ============================================================================

class CodeIndex:
    """Index for code search using latent trajectories."""

    def __init__(self, model: CodeSearchModel, tokenizer: CodeTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.chunks = []  # List of CodeChunk objects
        self.embeddings = None  # (num_chunks, latent_dim) tensor

    def add_chunks(self, chunks: List[CodeChunk], batch_size: int = 16, device: str = 'cpu'):
        """Add code chunks to index and compute embeddings.

        Uses continual learning - model adapts to each batch.
        """
        self.model.to(device)
        self.model.train()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        all_embeddings = []

        with tqdm(total=len(chunks), desc="Indexing") as pbar:
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]

                # Prepare batch
                tokens_list = [chunk.tokens for chunk in batch_chunks]
                tokens = torch.tensor(tokens_list, dtype=torch.long, device=device)

                # Train step (continual learning)
                optimizer.zero_grad()
                loss, metrics = self.model.compute_loss(tokens, loss_weights=(1.0, 0.1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                # Get embeddings (no gradient)
                self.model.eval()
                with torch.no_grad():
                    batch_embeddings = self.model.get_embedding(tokens)
                    all_embeddings.append(batch_embeddings.cpu())

                    # Store in memory for experience replay
                    for j, chunk in enumerate(batch_chunks):
                        score = metrics['kl_loss']  # Use KL as priority score
                        self.model.memory.add(
                            {'tokens': tokens[j:j+1].cpu()},
                            score=score
                        )
                self.model.train()

                # Update progress
                pbar.update(len(batch_chunks))
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.3f}",
                    'kl': f"{metrics['kl_loss']:.3f}",
                    'mem': len(self.model.memory),
                })

        # Store chunks and embeddings
        self.chunks.extend(chunks)
        self.embeddings = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, self.model.latent_dim)

    def save(self, path: Path):
        """Save index to disk."""
        path = Path(path)
        index_data = {
            'model_state': self.model.state_dict(),
            'memory_state': self.model.memory.state_dict(),
            'chunks': [(c.tokens, str(c.file_path), c.start_line, c.end_line, c.metadata) for c in self.chunks],
            'embeddings': self.embeddings,
            'config': {
                'vocab_size': self.model.vocab_size,
                'latent_dim': self.model.latent_dim,
                'hidden_dim': self.model.hidden_dim,
                'embed_dim': self.model.embed_dim,
                'seq_len': self.model.seq_len,
                'memory_size': len(self.model.memory),
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"✓ Index saved to {path}")

    @classmethod
    def load(cls, path: Path, tokenizer: CodeTokenizer, device: str = 'cpu'):
        """Load index from disk."""
        path = Path(path)
        with open(path, 'rb') as f:
            index_data = pickle.load(f)

        # Recreate model
        config = index_data['config']
        model = CodeSearchModel(
            vocab_size=config['vocab_size'],
            latent_dim=config['latent_dim'],
            hidden_dim=config['hidden_dim'],
            embed_dim=config['embed_dim'],
            seq_len=config['seq_len'],
            memory_size=config['memory_size'],
        )
        model.load_state_dict(index_data['model_state'])
        model.memory.load_state_dict(index_data['memory_state'])
        model.to(device)
        model.eval()

        # Recreate index
        index = cls(model, tokenizer)
        index.embeddings = index_data['embeddings'].to(device)

        # Recreate chunks (simplified - no actual CodeChunk objects)
        for tokens, file_path, start_line, end_line, metadata in index_data['chunks']:
            chunk = CodeChunk(tokens, Path(file_path), start_line, end_line, metadata)
            index.chunks.append(chunk)

        print(f"✓ Index loaded from {path} ({len(index.chunks)} chunks)")
        return index


# ============================================================================
# Component 4: Query Search (Find relevant code using trajectory similarity)
# ============================================================================

def search(
    query: str,
    index: CodeIndex,
    tokenizer: CodeTokenizer,
    top_k: int = 5,
    device: str = 'cpu',
) -> List[Tuple[CodeChunk, float]]:
    """Search for relevant code chunks using trajectory similarity.

    Args:
        query: Natural language query
        index: CodeIndex with embeddings
        tokenizer: CodeTokenizer
        top_k: Number of results to return
        device: Device to run on

    Returns:
        List of (CodeChunk, similarity_score) tuples, sorted by relevance
    """
    # Encode query
    query_tokens = tokenizer.encode(query)
    query_tokens = tokenizer.pad_or_truncate(query_tokens, index.model.seq_len)
    query_tensor = torch.tensor([query_tokens], dtype=torch.long, device=device)

    # Get query embedding
    index.model.eval()
    with torch.no_grad():
        query_embedding = index.model.get_embedding(query_tensor)  # (1, latent_dim)

    # Compute similarity with all chunks
    # Use cosine similarity on final latent states
    embeddings = index.embeddings.to(device)  # (num_chunks, latent_dim)
    similarities = F.cosine_similarity(
        query_embedding.unsqueeze(1),  # (1, 1, latent_dim)
        embeddings.unsqueeze(0),  # (1, num_chunks, latent_dim)
        dim=2
    ).squeeze(0)  # (num_chunks,)

    # Get top-k
    top_k = min(top_k, len(index.chunks))
    top_scores, top_indices = similarities.topk(top_k)

    results = []
    for idx, score in zip(top_indices.cpu().tolist(), top_scores.cpu().tolist()):
        results.append((index.chunks[idx], score))

    return results


def display_results(results: List[Tuple[CodeChunk, float]], tokenizer: CodeTokenizer):
    """Display search results in a nice format."""
    print("\n" + "="*70)
    print(f"Found {len(results)} results:")
    print("="*70)

    for i, (chunk, score) in enumerate(results, 1):
        # Decode chunk to show preview
        preview = tokenizer.decode(chunk.tokens)
        preview_lines = preview.split('\n')[:10]  # First 10 lines
        preview_text = '\n    '.join(preview_lines)

        # Show metadata if available
        metadata_str = ""
        if chunk.metadata['functions']:
            metadata_str += f"  Functions: {', '.join(chunk.metadata['functions'][:3])}\n"
        if chunk.metadata['classes']:
            metadata_str += f"  Classes: {', '.join(chunk.metadata['classes'][:3])}\n"

        print(f"\n[{i}] {chunk.file_path}:{chunk.start_line}-{chunk.end_line} (score: {score:.3f})")
        if metadata_str:
            print(metadata_str)
        print(f"    {preview_text}")
        print("─" * 70)


# ============================================================================
# CLI Interface
# ============================================================================

def cmd_index(args):
    """Index a repository."""
    print("🦝 Latent Code Search - Indexing")
    print("="*70)

    # Setup
    device = torch.device('cpu')  # Force CPU for compatibility
    tokenizer = CodeTokenizer(vocab_size=256)

    # Create model
    model = CodeSearchModel(
        vocab_size=256,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        seq_len=args.seq_len,
        memory_size=args.memory_size,
    )
    print(f"✓ Model created ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # Crawl repository
    crawler = CodebaseCrawler(tokenizer, window=args.seq_len, stride=args.seq_len // 2)
    print(f"✓ Crawling {args.repo_path}...")
    chunks = list(crawler.crawl(Path(args.repo_path), pattern=args.pattern))
    print(f"✓ Found {len(chunks)} code chunks")

    if len(chunks) == 0:
        print("❌ No code chunks found. Check the repository path and pattern.")
        return

    # Build index
    index = CodeIndex(model, tokenizer)
    index.add_chunks(chunks, batch_size=args.batch_size, device=device)

    # Save
    index.save(Path(args.output))
    print(f"\n✅ Indexing complete!")
    print(f"   Indexed {len(chunks)} chunks from {args.repo_path}")
    print(f"   Index saved to {args.output}")


def cmd_query(args):
    """Search the index."""
    print("🦝 Latent Code Search - Query")
    print("="*70)

    # Setup
    device = torch.device('cpu')
    tokenizer = CodeTokenizer(vocab_size=256)

    # Load index
    print(f"Loading index from {args.index}...")
    index = CodeIndex.load(Path(args.index), tokenizer, device=device)

    # Search
    print(f"\nQuery: \"{args.query}\"")
    results = search(args.query, index, tokenizer, top_k=args.top_k, device=device)

    # Display
    display_results(results, tokenizer)


def main():
    parser = argparse.ArgumentParser(description="Latent Code Search - Search codebases using trajectory planning")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Index command
    index_parser = subparsers.add_parser('index', help='Index a repository')
    index_parser.add_argument('repo_path', type=str, help='Path to repository to index')
    index_parser.add_argument('--output', '-o', type=str, default='code.index', help='Output index file')
    index_parser.add_argument('--pattern', type=str, default='*.py', help='File pattern to match')
    index_parser.add_argument('--seq-len', type=int, default=512, help='Sequence length for chunks')
    index_parser.add_argument('--latent-dim', type=int, default=32, help='Latent dimension')
    index_parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    index_parser.add_argument('--embed-dim', type=int, default=32, help='Embedding dimension')
    index_parser.add_argument('--memory-size', type=int, default=2000, help='Memory buffer size')
    index_parser.add_argument('--batch-size', type=int, default=16, help='Batch size for indexing')

    # Query command
    query_parser = subparsers.add_parser('query', help='Search the index')
    query_parser.add_argument('query', type=str, help='Query string')
    query_parser.add_argument('--index', '-i', type=str, default='code.index', help='Index file to search')
    query_parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results to return')

    args = parser.parse_args()

    if args.command == 'index':
        cmd_index(args)
    elif args.command == 'query':
        cmd_query(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
