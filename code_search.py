#!/usr/bin/env python3
"""
Latent Code Search - Search your codebase using trajectory planning in latent space.

This is a practical application of the latent trajectory transformer that:
1. Indexes ALL file types: .py, .md, .txt, .yaml, .json, .rs, .cpp, .js, etc.
2. Extracts intelligent metadata: functions/classes (code), headers (markdown), keys (config)
3. Uses paragraph-aware chunking for markdown, respecting document structure
4. Learns latent trajectory representations incrementally (continual learning)
5. Finds relevant content using trajectory similarity
6. Returns results with intelligent EXPLANATIONS of why they match
7. Runs efficiently on CPU (single-core)

Usage:
    # Index entire repository (all supported file types)
    python code_search.py index . --output repo.index

    # Index specific file types
    python code_search.py index . --extensions ".py,.md,.txt" --output repo.index

    # Search with intelligent explanations
    python code_search.py query "where is the SDE dynamics?" --index repo.index
    python code_search.py query "explain continual learning" --index repo.index
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
    """Crawls a repository and extracts code chunks.

    Supports multiple file types with intelligent metadata extraction and chunking:
    - Code files (.py, .js, .rs, .cpp, .java, etc.): Extract functions/classes, fixed-window chunking
    - Markdown files (.md): Extract headers, paragraph-aware chunking
    - Config files (.yaml, .json, .toml, .txt): Extract keys, structure-aware chunking
    """

    # Supported file extensions by category
    CODE_EXTENSIONS = {'.py', '.js', '.ts', '.rs', '.cpp', '.c', '.h', '.hpp', '.java', '.go', '.rb', '.php', '.sh'}
    MARKDOWN_EXTENSIONS = {'.md', '.markdown'}
    CONFIG_EXTENSIONS = {'.yaml', '.yml', '.json', '.toml', '.txt', '.ini', '.cfg', '.conf'}

    def __init__(self, tokenizer: CodeTokenizer, window: int = 512, stride: int = 256):
        self.tokenizer = tokenizer
        self.window = window
        self.stride = stride

    def get_file_type(self, file_path: Path) -> str:
        """Determine file type category from extension."""
        ext = file_path.suffix.lower()
        if ext in self.CODE_EXTENSIONS:
            return 'code'
        elif ext in self.MARKDOWN_EXTENSIONS:
            return 'markdown'
        elif ext in self.CONFIG_EXTENSIONS:
            return 'config'
        else:
            return 'text'  # Fallback

    def extract_metadata_python(self, code: str) -> Dict:
        """Extract metadata from Python code."""
        metadata = {
            'functions': [],
            'classes': [],
            'imports': [],
            'type': 'python',
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

    def extract_metadata_markdown(self, text: str) -> Dict:
        """Extract metadata from Markdown files."""
        metadata = {
            'headers': [],
            'code_blocks': [],
            'lists': [],
            'type': 'markdown',
        }

        # Extract headers (# Header, ## Subheader, etc.)
        header_pattern = r'^(#{1,6})\s+(.+)$'
        for match in re.finditer(header_pattern, text, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            metadata['headers'].append((level, title))

        # Extract code block languages
        code_block_pattern = r'```(\w+)?'
        metadata['code_blocks'] = re.findall(code_block_pattern, text)

        # Count list items (rough estimate of structure)
        list_pattern = r'^\s*[-*+]\s+'
        metadata['lists'] = len(re.findall(list_pattern, text, re.MULTILINE))

        return metadata

    def extract_metadata_config(self, text: str, file_ext: str) -> Dict:
        """Extract metadata from config files."""
        metadata = {
            'keys': [],
            'sections': [],
            'type': 'config',
            'format': file_ext.lstrip('.'),
        }

        if file_ext in {'.yaml', '.yml'}:
            # YAML top-level keys
            key_pattern = r'^(\w+):\s*'
            metadata['keys'] = re.findall(key_pattern, text, re.MULTILINE)
        elif file_ext == '.json':
            # JSON top-level keys (rough approximation)
            key_pattern = r'^\s*"(\w+)"\s*:'
            metadata['keys'] = re.findall(key_pattern, text, re.MULTILINE)
        elif file_ext == '.toml':
            # TOML sections and keys
            section_pattern = r'^\[(.+)\]$'
            metadata['sections'] = re.findall(section_pattern, text, re.MULTILINE)
            key_pattern = r'^(\w+)\s*='
            metadata['keys'] = re.findall(key_pattern, text, re.MULTILINE)
        else:
            # Generic text - extract common key-value patterns
            key_pattern = r'^(\w+)\s*[=:]'
            metadata['keys'] = re.findall(key_pattern, text, re.MULTILINE)

        return metadata

    def extract_metadata_code(self, code: str, file_ext: str) -> Dict:
        """Extract metadata from generic code files (JS, Rust, C++, etc.)."""
        metadata = {
            'functions': [],
            'classes': [],
            'imports': [],
            'type': 'code',
            'language': file_ext.lstrip('.'),
        }

        # Generic function pattern (works for JS, C++, Rust, etc.)
        # Matches: function foo(), fn foo(), void foo(), etc.
        func_pattern = r'(?:function|fn|def|void|int|bool|async|pub\s+fn)\s+(\w+)\s*\('
        metadata['functions'] = re.findall(func_pattern, code)

        # Generic class/struct pattern
        class_pattern = r'(?:class|struct|interface|trait|impl)\s+(\w+)'
        metadata['classes'] = re.findall(class_pattern, code)

        # Generic import pattern
        import_pattern = r'(?:import|require|use|#include)\s+[\'"]?([^\'";\n]+)'
        metadata['imports'] = re.findall(import_pattern, code)

        return metadata

    def extract_metadata(self, text: str, file_type: str, file_ext: str) -> Dict:
        """Extract metadata based on file type."""
        if file_type == 'code':
            if file_ext == '.py':
                return self.extract_metadata_python(text)
            else:
                return self.extract_metadata_code(text, file_ext)
        elif file_type == 'markdown':
            return self.extract_metadata_markdown(text)
        elif file_type == 'config':
            return self.extract_metadata_config(text, file_ext)
        else:
            return {'type': 'text'}

    def chunk_markdown(self, text: str, tokens: List[int]) -> Iterator[Tuple[List[int], Dict]]:
        """Chunk markdown by paragraphs and sections, respecting document structure.

        Returns: Iterator of (chunk_tokens, chunk_info) where chunk_info contains
                line offsets for proper line number calculation.
        """
        lines = text.split('\n')
        paragraphs = []
        current_para = []
        current_line = 0

        for line in lines:
            if line.strip() == '':
                if current_para:
                    paragraphs.append({
                        'lines': current_para,
                        'start_line': current_line - len(current_para),
                        'text': '\n'.join(current_para)
                    })
                    current_para = []
            else:
                current_para.append(line)
            current_line += 1

        # Add last paragraph
        if current_para:
            paragraphs.append({
                'lines': current_para,
                'start_line': current_line - len(current_para),
                'text': '\n'.join(current_para)
            })

        # Chunk paragraphs, combining small ones
        for para in paragraphs:
            para_tokens = self.tokenizer.encode(para['text'])

            if len(para_tokens) <= self.window:
                # Small paragraph - pad and yield
                chunk_tokens = self.tokenizer.pad_or_truncate(para_tokens, self.window)
                yield chunk_tokens, {
                    'start_line': para['start_line'],
                    'end_line': para['start_line'] + len(para['lines'])
                }
            else:
                # Large paragraph - use sliding window
                for i in range(0, len(para_tokens), self.stride):
                    chunk = para_tokens[i:i + self.window]
                    chunk = self.tokenizer.pad_or_truncate(chunk, self.window)

                    # Estimate line numbers for this chunk
                    char_ratio = i / len(para_tokens)
                    start_offset = int(char_ratio * len(para['lines']))
                    yield chunk, {
                        'start_line': para['start_line'] + start_offset,
                        'end_line': para['start_line'] + min(start_offset + 10, len(para['lines']))
                    }

    def crawl(self, repo_path: Path, extensions: Optional[List[str]] = None) -> Iterator[CodeChunk]:
        """Crawl repository and yield code chunks.

        Args:
            repo_path: Path to repository root
            extensions: List of file extensions to include (e.g., ['.py', '.md', '.txt'])
                       If None, includes all supported extensions
        """
        repo_path = Path(repo_path)

        # Default to all supported extensions
        if extensions is None:
            extensions = list(self.CODE_EXTENSIONS | self.MARKDOWN_EXTENSIONS | self.CONFIG_EXTENSIONS)

        # Convert to set for fast lookup
        extensions_set = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions}

        # Iterate over all files
        for file_path in repo_path.rglob('*'):
            if not file_path.is_file():
                continue

            # Check extension
            if file_path.suffix.lower() not in extensions_set:
                continue

            # Read file
            try:
                text = file_path.read_text(encoding='utf-8')
            except (UnicodeDecodeError, PermissionError):
                continue

            # Skip empty files
            if not text.strip():
                continue

            # Determine file type
            file_type = self.get_file_type(file_path)
            file_ext = file_path.suffix.lower()

            # Extract metadata
            metadata = self.extract_metadata(text, file_type, file_ext)

            # Tokenize
            tokens = self.tokenizer.encode(text)

            # Chunk based on file type
            if file_type == 'markdown':
                # Use paragraph-aware chunking for markdown
                for chunk_tokens, chunk_info in self.chunk_markdown(text, tokens):
                    yield CodeChunk(
                        tokens=chunk_tokens,
                        file_path=file_path,
                        start_line=chunk_info['start_line'] + 1,  # 1-indexed
                        end_line=chunk_info['end_line'] + 1,
                        metadata=metadata,
                    )
            else:
                # Use fixed-window chunking for code and config files
                lines = text.split('\n')
                chars_per_line = [len(line) + 1 for line in lines]  # +1 for newline
                cumulative_chars = [sum(chars_per_line[:i+1]) for i in range(len(chars_per_line))]

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


def generate_explanation(chunk: CodeChunk, query: str, score: float) -> str:
    """Generate intelligent explanation for why this chunk matches the query.

    Uses template-based approach with metadata to create human-readable explanations.
    """
    file_name = chunk.file_path.name
    file_type = chunk.metadata.get('type', 'unknown')
    explanations = []

    # Analyze query for keywords
    query_lower = query.lower()
    query_words = set(re.findall(r'\w+', query_lower))

    # Explanation based on file type
    if file_type == 'python' or file_type == 'code':
        language = chunk.metadata.get('language', 'Python')

        # Check for function/class matches
        functions = chunk.metadata.get('functions', [])
        classes = chunk.metadata.get('classes', [])
        imports = chunk.metadata.get('imports', [])

        if functions:
            func_matches = [f for f in functions if any(word in f.lower() for word in query_words)]
            if func_matches:
                explanations.append(f"contains function(s) `{', '.join(func_matches[:2])}` matching query terms")
            elif functions:
                explanations.append(f"defines {len(functions)} function(s) including `{functions[0]}`")

        if classes:
            class_matches = [c for c in classes if any(word in c.lower() for word in query_words)]
            if class_matches:
                explanations.append(f"defines class `{class_matches[0]}` matching query")
            elif classes:
                explanations.append(f"implements class `{classes[0]}`")

        if imports:
            explanations.append(f"imports {len(imports)} module(s)")

        if not explanations:
            explanations.append(f"{language} code implementation")

    elif file_type == 'markdown':
        headers = chunk.metadata.get('headers', [])
        code_blocks = chunk.metadata.get('code_blocks', [])

        if headers:
            # Find headers matching query
            header_matches = [(lvl, title) for lvl, title in headers if any(word in title.lower() for word in query_words)]
            if header_matches:
                _, title = header_matches[0]
                explanations.append(f"section titled '{title}' matching query")
            else:
                _, title = headers[0]
                explanations.append(f"documentation section '{title}'")

        if code_blocks:
            explanations.append(f"includes {len(code_blocks)} code example(s)")

        if not explanations:
            explanations.append("documentation content")

    elif file_type == 'config':
        config_format = chunk.metadata.get('format', 'config')
        keys = chunk.metadata.get('keys', [])
        sections = chunk.metadata.get('sections', [])

        if keys:
            key_matches = [k for k in keys if any(word in k.lower() for word in query_words)]
            if key_matches:
                explanations.append(f"{config_format} with key `{key_matches[0]}` matching query")
            else:
                explanations.append(f"{config_format} configuration with {len(keys)} keys")

        if sections:
            explanations.append(f"defines section(s): {', '.join(sections[:2])}")

        if not explanations:
            explanations.append(f"{config_format} configuration file")

    else:
        explanations.append("text content")

    # Build final explanation
    explanation_text = " and ".join(explanations)

    # Add relevance context
    if score > 0.8:
        relevance = "highly relevant"
    elif score > 0.6:
        relevance = "relevant"
    else:
        relevance = "potentially relevant"

    return f"Found in **{file_name}** lines {chunk.start_line}-{chunk.end_line}: {explanation_text}. This is {relevance} to your query (similarity: {score:.2f})."


def display_results(results: List[Tuple[CodeChunk, float]], tokenizer: CodeTokenizer, query: str = ""):
    """Display search results with intelligent explanations."""
    print("\n" + "="*70)
    print(f"Found {len(results)} results:")
    print("="*70)

    for i, (chunk, score) in enumerate(results, 1):
        # Decode chunk to show preview
        preview = tokenizer.decode(chunk.tokens)
        preview_lines = preview.split('\n')[:10]  # First 10 lines
        preview_text = '\n    '.join(preview_lines)

        # Generate intelligent explanation
        explanation = generate_explanation(chunk, query, score)

        # Show metadata summary
        metadata_items = []
        if 'functions' in chunk.metadata and chunk.metadata['functions']:
            metadata_items.append(f"Functions: {', '.join(chunk.metadata['functions'][:3])}")
        if 'classes' in chunk.metadata and chunk.metadata['classes']:
            metadata_items.append(f"Classes: {', '.join(chunk.metadata['classes'][:3])}")
        if 'headers' in chunk.metadata and chunk.metadata['headers']:
            headers_str = ', '.join(f"{'#'*(lvl)} {title}" for lvl, title in chunk.metadata['headers'][:2])
            metadata_items.append(f"Headers: {headers_str}")
        if 'keys' in chunk.metadata and chunk.metadata['keys']:
            metadata_items.append(f"Keys: {', '.join(chunk.metadata['keys'][:5])}")

        print(f"\n[{i}] {chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
        print(f"    📖 {explanation}")
        if metadata_items:
            print(f"    🔍 {' | '.join(metadata_items)}")
        print(f"\n    Preview:\n    {preview_text}")
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

    # Parse extensions
    if args.extensions:
        extensions = [ext.strip() for ext in args.extensions.split(',')]
        print(f"✓ Indexing file types: {', '.join(extensions)}")
    else:
        extensions = None  # Use all supported extensions
        print(f"✓ Indexing all supported file types (.py, .md, .txt, .yaml, .json, etc.)")

    # Crawl repository
    crawler = CodebaseCrawler(tokenizer, window=args.seq_len, stride=args.seq_len // 2)
    print(f"✓ Crawling {args.repo_path}...")
    chunks = list(crawler.crawl(Path(args.repo_path), extensions=extensions))
    print(f"✓ Found {len(chunks)} code chunks")

    if len(chunks) == 0:
        print("❌ No code chunks found. Check the repository path and extensions.")
        return

    # Show file type breakdown
    file_types = {}
    for chunk in chunks:
        ftype = chunk.metadata.get('type', 'unknown')
        file_types[ftype] = file_types.get(ftype, 0) + 1
    print(f"   File type breakdown: {', '.join(f'{k}={v}' for k, v in file_types.items())}")

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

    # Display with intelligent explanations
    display_results(results, tokenizer, query=args.query)


def main():
    parser = argparse.ArgumentParser(description="Latent Code Search - Search codebases using trajectory planning")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Index command
    index_parser = subparsers.add_parser('index', help='Index a repository')
    index_parser.add_argument('repo_path', type=str, help='Path to repository to index')
    index_parser.add_argument('--output', '-o', type=str, default='code.index', help='Output index file')
    index_parser.add_argument('--extensions', '-e', type=str, default=None,
                              help='Comma-separated file extensions (e.g., ".py,.md,.txt"). Default: all supported types')
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
