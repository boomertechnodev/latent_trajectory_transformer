#!/usr/bin/env python3
"""
NEURAL CODE SEARCH - Intelligent Code Search with Explanations
================================================================

Multi-scale semantic code search using latent trajectory transformers.
NOT just cosine similarity - learns to generate natural language explanations
of what code does and why it matches your query.

Architecture:
1. Universal Tokenizer - preserves structure (docstrings, comments, code)
2. Encoder → SDE Trajectory → Normalizing Flow → Semantic Latent Space
3. GRU Explanation Decoder - generates natural language from latent
4. Incremental Learning - trains while indexing your codebase
5. Intelligent Search - returns explanations, not just code chunks

Key Innovation:
- Learns WHILE indexing on code-documentation pairs
- Understands YOUR codebase (e.g., "Raccoon" = continual learning)
- Generates explanations specific to actual code semantics
- Multi-level understanding: t=0.0 (syntax) → t=0.1 (semantics)

Usage:
    python neural_code_search.py index <directory> --output index.pt
    python neural_code_search.py search "SDE dynamics" --index index.pt
    python neural_code_search.py explain <file>:<line> --index index.pt
"""

import os
import re
import sys
import ast
import json
import math
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange

# Import Raccoon components from latent_drift_trajectory.py
from latent_drift_trajectory import (
    RaccoonDynamics,
    RaccoonFlow,
    RaccoonMemory,
    solve_sde,
    FastEppsPulley,
    SlicingUnivariateTest,
    TimeAwareTransform,
)


# ──────────────────────────────────────────────────────────────────────────────
#  UNIVERSAL TOKENIZER - Structure-Preserving Multi-Format
# ──────────────────────────────────────────────────────────────────────────────

# Extended vocabulary: printable ASCII + special structure tokens
SPECIAL_STRUCTURE_TOKENS = {
    '<PAD>': 0,
    '<UNK>': 1,
    '<DOCSTRING>': 2,
    '</DOCSTRING>': 3,
    '<COMMENT>': 4,
    '</COMMENT>': 5,
    '<CODE>': 6,
    '</CODE>': 7,
    '<HEADER>': 8,
    '</HEADER>': 9,
}

# Printable ASCII characters (32-126)
BASE_CHARS = [chr(i) for i in range(32, 127)]
VOCAB_CHARS = list(SPECIAL_STRUCTURE_TOKENS.keys()) + BASE_CHARS
CHAR_TO_IDX = {ch: i for i, ch in enumerate(VOCAB_CHARS)}
IDX_TO_CHAR = {i: ch for ch, i in CHAR_TO_IDX.items()}
VOCAB_SIZE = len(VOCAB_CHARS)

# Token IDs for quick access
PAD_ID = SPECIAL_STRUCTURE_TOKENS['<PAD>']
UNK_ID = SPECIAL_STRUCTURE_TOKENS['<UNK>']
DOCSTRING_START = SPECIAL_STRUCTURE_TOKENS['<DOCSTRING>']
DOCSTRING_END = SPECIAL_STRUCTURE_TOKENS['</DOCSTRING>']
COMMENT_START = SPECIAL_STRUCTURE_TOKENS['<COMMENT>']
COMMENT_END = SPECIAL_STRUCTURE_TOKENS['</COMMENT>']
CODE_START = SPECIAL_STRUCTURE_TOKENS['<CODE>']
CODE_END = SPECIAL_STRUCTURE_TOKENS['</CODE>']
HEADER_START = SPECIAL_STRUCTURE_TOKENS['<HEADER>']
HEADER_END = SPECIAL_STRUCTURE_TOKENS['</HEADER>']


@dataclass
class CodeChunk:
    """Structured code chunk with documentation and metadata."""
    filepath: str
    start_line: int
    end_line: int
    code_text: str
    explanation_target: Optional[str]  # Docstring/comment for training
    language: str
    chunk_type: str  # 'function', 'class', 'module', 'markdown', 'text'
    metadata: Dict[str, Any]  # function_name, class_name, imports, etc.

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'CodeChunk':
        return cls(**d)


class UniversalTokenizer:
    """
    Structure-preserving tokenizer for ALL file types.

    Extracts:
    - Python: docstrings, comments, function/class names, imports
    - Markdown: headers, code blocks, paragraphs
    - Text: natural language chunks
    - JSON/YAML: structure-aware parsing
    - Shell: comments, function names

    Returns chunks with:
    - Structured tokens (with special markers)
    - Explanation targets (docstrings/comments)
    - Rich metadata
    """

    # Python docstring patterns
    PYTHON_DOCSTRING_PATTERN = re.compile(
        r'(""".*?"""|\'\'\'.*?\'\'\')',
        re.DOTALL
    )

    # Comment patterns by language
    PYTHON_COMMENT = re.compile(r'#\s*(.*?)$', re.MULTILINE)
    SHELL_COMMENT = re.compile(r'#\s*(.*?)$', re.MULTILINE)
    YAML_COMMENT = re.compile(r'#\s*(.*?)$', re.MULTILINE)

    # Markdown patterns
    MD_HEADER = re.compile(r'^(#{1,6})\s+(.+?)$', re.MULTILINE)
    MD_CODE_BLOCK = re.compile(r'```(\w*)\n(.*?)\n```', re.DOTALL)

    # Shell function pattern
    SHELL_FUNCTION = re.compile(r'^\s*function\s+(\w+)|^(\w+)\s*\(\)\s*\{', re.MULTILINE)

    def __init__(self, chunk_size: int = 512, stride: int = 256):
        self.chunk_size = chunk_size
        self.stride = stride

    def tokenize_string(self, text: str) -> List[int]:
        """Convert string to token IDs."""
        tokens = []
        for char in text:
            token_id = CHAR_TO_IDX.get(char, UNK_ID)
            tokens.append(token_id)
        return tokens

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to string."""
        chars = [IDX_TO_CHAR.get(int(tid), '?') for tid in token_ids if tid != PAD_ID]
        return ''.join(chars)

    def extract_python_metadata(self, code: str, filepath: str) -> List[CodeChunk]:
        """
        Extract structured Python code with docstrings and comments.

        Returns chunks with:
        - Function/class definitions + docstrings
        - Module-level docstrings
        - Inline comments as explanation targets
        """
        chunks = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Fallback for unparseable code
            return self._chunk_raw_code(code, filepath, 'py')

        lines = code.split('\n')

        # Extract module-level docstring
        module_docstring = ast.get_docstring(tree)
        if module_docstring:
            chunk = CodeChunk(
                filepath=filepath,
                start_line=1,
                end_line=len(module_docstring.split('\n')),
                code_text=module_docstring,
                explanation_target=module_docstring,
                language='py',
                chunk_type='module_docstring',
                metadata={'type': 'module'}
            )
            chunks.append(chunk)

        # Extract functions and classes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Extract function with docstring
                docstring = ast.get_docstring(node)
                start_line = node.lineno
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10

                # Get function code
                func_code = '\n'.join(lines[start_line-1:end_line])

                chunk = CodeChunk(
                    filepath=filepath,
                    start_line=start_line,
                    end_line=end_line,
                    code_text=func_code,
                    explanation_target=docstring,
                    language='py',
                    chunk_type='function',
                    metadata={
                        'function_name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                    }
                )
                chunks.append(chunk)

            elif isinstance(node, ast.ClassDef):
                # Extract class with docstring
                docstring = ast.get_docstring(node)
                start_line = node.lineno
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 20

                # Get class code
                class_code = '\n'.join(lines[start_line-1:end_line])

                chunk = CodeChunk(
                    filepath=filepath,
                    start_line=start_line,
                    end_line=end_line,
                    code_text=class_code,
                    explanation_target=docstring,
                    language='py',
                    chunk_type='class',
                    metadata={
                        'class_name': node.name,
                        'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                    }
                )
                chunks.append(chunk)

        # If no structured chunks, fall back to raw chunking
        if not chunks:
            chunks = self._chunk_raw_code(code, filepath, 'py')

        return chunks

    def extract_markdown_metadata(self, content: str, filepath: str) -> List[CodeChunk]:
        """
        Extract structured Markdown with headers and code blocks.
        """
        chunks = []
        lines = content.split('\n')

        # Extract headers as explanation targets
        for match in self.MD_HEADER.finditer(content):
            level = len(match.group(1))
            header_text = match.group(2)
            line_num = content[:match.start()].count('\n') + 1

            # Get paragraph after header (up to next header or 10 lines)
            paragraph_lines = []
            for i in range(line_num, min(line_num + 10, len(lines))):
                if i < len(lines) and not lines[i].startswith('#'):
                    paragraph_lines.append(lines[i])
                else:
                    break

            paragraph = '\n'.join(paragraph_lines)

            chunk = CodeChunk(
                filepath=filepath,
                start_line=line_num,
                end_line=line_num + len(paragraph_lines),
                code_text=paragraph,
                explanation_target=header_text,
                language='md',
                chunk_type='section',
                metadata={'header': header_text, 'level': level}
            )
            chunks.append(chunk)

        # Extract code blocks
        for match in self.MD_CODE_BLOCK.finditer(content):
            lang = match.group(1) or 'text'
            code_text = match.group(2)
            line_num = content[:match.start()].count('\n') + 1

            chunk = CodeChunk(
                filepath=filepath,
                start_line=line_num,
                end_line=line_num + code_text.count('\n'),
                code_text=code_text,
                explanation_target=None,  # Code blocks don't have inherent explanations
                language=lang,
                chunk_type='code_block',
                metadata={'language': lang}
            )
            chunks.append(chunk)

        return chunks

    def extract_shell_metadata(self, content: str, filepath: str) -> List[CodeChunk]:
        """Extract shell functions and comments."""
        chunks = []
        lines = content.split('\n')

        # Extract function definitions
        for match in self.SHELL_FUNCTION.finditer(content):
            func_name = match.group(1) or match.group(2)
            start_line = content[:match.start()].count('\n') + 1

            # Get function body (up to closing brace or 20 lines)
            func_lines = []
            brace_count = 0
            for i in range(start_line - 1, min(start_line + 20, len(lines))):
                if i < len(lines):
                    func_lines.append(lines[i])
                    brace_count += lines[i].count('{') - lines[i].count('}')
                    if brace_count == 0 and i > start_line:
                        break

            func_code = '\n'.join(func_lines)

            # Extract comment before function as explanation
            explanation = None
            if start_line > 1:
                prev_line = lines[start_line - 2]
                comment_match = self.SHELL_COMMENT.search(prev_line)
                if comment_match:
                    explanation = comment_match.group(1).strip()

            chunk = CodeChunk(
                filepath=filepath,
                start_line=start_line,
                end_line=start_line + len(func_lines),
                code_text=func_code,
                explanation_target=explanation,
                language='sh',
                chunk_type='function',
                metadata={'function_name': func_name}
            )
            chunks.append(chunk)

        return chunks

    def _chunk_raw_code(self, content: str, filepath: str, language: str) -> List[CodeChunk]:
        """
        Fallback: chunk raw text with overlap.
        Extracts comments as explanation targets.
        """
        chunks = []
        lines = content.split('\n')

        # Extract comments based on language
        if language in ['py', 'sh', 'yaml']:
            comment_pattern = self.PYTHON_COMMENT
        else:
            comment_pattern = None

        # Chunk with overlap
        for i in range(0, len(lines), self.stride):
            chunk_lines = lines[i:i + self.chunk_size // 10]  # Approx chars to lines
            if not chunk_lines:
                continue

            code_text = '\n'.join(chunk_lines)

            # Extract explanation from comments in this chunk
            explanation = None
            if comment_pattern:
                comments = comment_pattern.findall(code_text)
                if comments:
                    explanation = ' '.join(c.strip() for c in comments)

            chunk = CodeChunk(
                filepath=filepath,
                start_line=i + 1,
                end_line=i + len(chunk_lines),
                code_text=code_text,
                explanation_target=explanation,
                language=language,
                chunk_type='raw_chunk',
                metadata={}
            )
            chunks.append(chunk)

        return chunks

    def chunk_file(self, filepath: Path) -> List[CodeChunk]:
        """
        Main entry point: chunk any file type with structure preservation.

        Args:
            filepath: Path to file
        Returns:
            List of CodeChunk objects with explanation targets
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"⚠️  Error reading {filepath}: {e}")
            return []

        suffix = filepath.suffix.lower()
        filepath_str = str(filepath)

        # Route to appropriate extractor
        if suffix == '.py':
            chunks = self.extract_python_metadata(content, filepath_str)
        elif suffix == '.md':
            chunks = self.extract_markdown_metadata(content, filepath_str)
        elif suffix in ['.sh', '.bash']:
            chunks = self.extract_shell_metadata(content, filepath_str)
        elif suffix in ['.txt', '.json', '.yaml', '.yml']:
            # For now, treat as raw text
            chunks = self._chunk_raw_code(content, filepath_str, suffix[1:])
        else:
            chunks = self._chunk_raw_code(content, filepath_str, 'text')

        return chunks

    def encode_chunk(self, chunk: CodeChunk, max_len: int = 512) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Encode chunk to tokens with structure markers.

        Returns:
            code_tokens: (max_len,) padded token IDs
            explanation_tokens: (max_len,) padded token IDs if explanation exists, else None
        """
        # Build structured representation
        tokens = []

        # Add structure markers based on chunk type
        if chunk.explanation_target:
            # Chunk with documentation
            tokens.append(DOCSTRING_START)
            tokens.extend(self.tokenize_string(chunk.explanation_target[:100]))  # Limit explanation
            tokens.append(DOCSTRING_END)

        tokens.append(CODE_START)
        tokens.extend(self.tokenize_string(chunk.code_text))
        tokens.append(CODE_END)

        # Truncate or pad to max_len
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens.extend([PAD_ID] * (max_len - len(tokens)))

        code_tokens = torch.tensor(tokens, dtype=torch.long)

        # Encode explanation target separately for training
        explanation_tokens = None
        if chunk.explanation_target:
            exp_tokens = self.tokenize_string(chunk.explanation_target)
            if len(exp_tokens) > max_len:
                exp_tokens = exp_tokens[:max_len]
            else:
                exp_tokens.extend([PAD_ID] * (max_len - len(exp_tokens)))
            explanation_tokens = torch.tensor(exp_tokens, dtype=torch.long)

        return code_tokens, explanation_tokens


# ──────────────────────────────────────────────────────────────────────────────
#  NEURAL CODE SEARCH MODEL - Encoder + SDE + Flow + Explanation Decoder
# ──────────────────────────────────────────────────────────────────────────────

class ExplanationDecoder(nn.Module):
    """
    GRU-based explanation decoder.

    Generates natural language explanations autoregressively from semantic latent.
    Architecture:
    - 2-layer GRU (64 hidden units)
    - Character-level generation
    - Teacher forcing during training
    - Sampling during inference
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        latent_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding for target tokens
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)

        # Project latent to initial GRU hidden state
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)

        # GRU decoder
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        z_semantic: Tensor,
        target_tokens: Optional[Tensor] = None,
        max_len: int = 128,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Generate explanation from semantic latent.

        Args:
            z_semantic: Semantic latent (batch, latent_dim)
            target_tokens: Ground truth for training (batch, seq_len) or None
            max_len: Maximum generation length

        Returns:
            logits: (batch, seq_len, vocab_size) if training
            generated: (batch, seq_len) if inference
            loss: Explanation loss if target provided
        """
        batch_size = z_semantic.shape[0]
        device = z_semantic.device

        # Initialize GRU hidden state from latent
        h = self.latent_to_hidden(z_semantic)  # (batch, hidden*layers)
        h = h.view(batch_size, self.num_layers, self.hidden_dim)  # (batch, layers, hidden)
        h = h.permute(1, 0, 2).contiguous()  # (layers, batch, hidden)

        if target_tokens is not None:
            # Training mode: teacher forcing
            target_embed = self.embedding(target_tokens)  # (batch, seq_len, embed)
            rnn_out, _ = self.gru(target_embed, h)  # (batch, seq_len, hidden)
            logits = self.output_proj(rnn_out)  # (batch, seq_len, vocab)

            # Compute loss
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                target_tokens.reshape(-1),
                ignore_index=PAD_ID,
            )

            return logits, loss

        else:
            # Inference mode: autoregressive generation
            generated = []
            current_token = torch.full((batch_size, 1), ord(' '), dtype=torch.long, device=device)

            for _ in range(max_len):
                token_embed = self.embedding(current_token)  # (batch, 1, embed)
                rnn_out, h = self.gru(token_embed, h)  # (batch, 1, hidden)
                logits = self.output_proj(rnn_out.squeeze(1))  # (batch, vocab)

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                current_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
                generated.append(current_token)

            generated_tokens = torch.cat(generated, dim=1)  # (batch, max_len)
            return generated_tokens, None


class NeuralCodeSearchModel(nn.Module):
    """
    Complete neural code search model with explanation generation.

    Architecture:
    1. Encoder: tokens → mean & logvar → z0 (latent distribution)
    2. SDE Dynamics: z0 → trajectory [t=0.0, 0.05, 0.1] → z_endpoint
    3. Normalizing Flow: z_endpoint → z_semantic (semantic space)
    4. Explanation Decoder: z_semantic → natural language

    Training:
    - Explanation loss (CrossEntropy on docstrings/comments)
    - KL divergence (regularize latent distribution)
    - Epps-Pulley test (latent smoothness)

    Inference:
    - Encode query → z_semantic
    - Cosine similarity search
    - Generate explanations for top-k results
    """
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_classes: int = 0,  # Optional classification head
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

        # Encoder: tokens → latent distribution
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.enc_mean = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # SDE dynamics (reuse RaccoonDynamics)
        self.dynamics = RaccoonDynamics(latent_dim, hidden_dim, sigma_min=1e-4, sigma_max=1.0)

        # Normalizing flow (reuse RaccoonFlow)
        self.flow = RaccoonFlow(latent_dim, hidden_dim, num_layers=4, time_dim=32)

        # Explanation decoder (NEW - key component!)
        self.decoder = ExplanationDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            hidden_dim=64,
            num_layers=2,
        )

        # Latent regularization (reuse SlicingUnivariateTest)
        univariate_test = FastEppsPulley(t_max=5.0, n_points=17)
        self.latent_test = SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=64,
            reduction="mean",
        )

    def encode(self, tokens: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode tokens to latent distribution.

        Args:
            tokens: (batch, seq_len)
        Returns:
            mean: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        # Embed and pool
        x = self.embedding(tokens)  # (batch, seq_len, embed)

        # Mask padding
        mask = (tokens != PAD_ID).float().unsqueeze(-1)  # (batch, seq_len, 1)
        x = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # (batch, embed)

        # Encode
        h = self.encoder(x)  # (batch, hidden)
        mean = self.enc_mean(h)
        logvar = self.enc_logvar(h)

        return mean, logvar

    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode_to_semantic(self, tokens: Tensor) -> Tensor:
        """
        Full encoding pipeline to semantic space.

        Args:
            tokens: (batch, seq_len)
        Returns:
            z_semantic: (batch, latent_dim) in semantic space
        """
        # Encode to distribution
        mean, logvar = self.encode(tokens)
        z0 = self.reparameterize(mean, logvar)

        # SDE trajectory (multi-level understanding)
        # t=0.0: syntax, t=0.05: structure, t=0.1: semantics
        device = tokens.device
        t_span = torch.linspace(0.0, 0.1, 3, device=device)
        z_traj = solve_sde(self.dynamics, z0, t_span)  # (batch, 3, latent)
        z_endpoint = z_traj[:, -1, :]  # Take final state (semantics)

        # Normalizing flow to semantic space
        t_flow = torch.ones(z0.shape[0], 1, device=device) * 0.5
        z_semantic, _ = self.flow(z_endpoint, t_flow, reverse=False)

        return z_semantic

    def generate_explanation(self, z_semantic: Tensor, max_len: int = 128) -> Tensor:
        """
        Generate explanation from semantic latent.

        Args:
            z_semantic: (batch, latent_dim)
            max_len: Maximum explanation length
        Returns:
            explanation_tokens: (batch, max_len)
        """
        generated, _ = self.decoder(z_semantic, target_tokens=None, max_len=max_len)
        return generated

    def forward(
        self,
        tokens: Tensor,
        explanation_targets: Optional[Tensor] = None,
        loss_weights: Tuple[float, float, float] = (1.0, 0.1, 0.01),
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Full forward pass with loss computation.

        Args:
            tokens: (batch, seq_len) input code tokens
            explanation_targets: (batch, seq_len) target explanations (if available)
            loss_weights: (explanation_weight, kl_weight, ep_weight)

        Returns:
            loss: Total loss
            stats: Dictionary of loss components
        """
        batch_size = tokens.shape[0]
        device = tokens.device

        # Encode
        mean, logvar = self.encode(tokens)
        z0 = self.reparameterize(mean, logvar)

        # SDE trajectory
        t_span = torch.linspace(0.0, 0.1, 3, device=device)
        z_traj = solve_sde(self.dynamics, z0, t_span)
        z_endpoint = z_traj[:, -1, :]

        # Normalizing flow
        t_flow = torch.ones(batch_size, 1, device=device) * 0.5
        z_semantic, log_det = self.flow(z_endpoint, t_flow, reverse=False)

        # Explanation loss (if targets available)
        explanation_loss = torch.tensor(0.0, device=device)
        if explanation_targets is not None:
            _, exp_loss = self.decoder(z_semantic, target_tokens=explanation_targets)
            explanation_loss = exp_loss

        # KL divergence to N(0,I)
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        kl_loss = torch.clamp(kl_loss, min=0.0)

        # Epps-Pulley regularization
        z_for_test = z_semantic.unsqueeze(0)  # (1, batch, latent)
        ep_loss = self.latent_test(z_for_test)

        # Total loss
        w_exp, w_kl, w_ep = loss_weights
        loss = w_exp * explanation_loss + w_kl * kl_loss + w_ep * ep_loss

        stats = {
            'explanation_loss': explanation_loss.detach(),
            'kl_loss': kl_loss.detach(),
            'ep_loss': ep_loss.detach(),
            'z_semantic': z_semantic.detach(),
        }

        return loss, stats


# ──────────────────────────────────────────────────────────────────────────────
#  INCREMENTAL LEARNING & INDEXING
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchIndex:
    """Search index with embeddings and metadata."""
    embeddings: Tensor  # (N, latent_dim)
    chunks: List[CodeChunk]
    model_state: dict
    config: dict

    def save(self, path: Path):
        """Save index to disk."""
        path = Path(path)
        data = {
            'embeddings': self.embeddings.cpu(),
            'chunks': [c.to_dict() for c in self.chunks],
            'model_state': self.model_state,
            'config': self.config,
        }
        torch.save(data, path)
        print(f"💾 Index saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'SearchIndex':
        """Load index from disk."""
        data = torch.load(path, map_location='cpu')
        return cls(
            embeddings=data['embeddings'],
            chunks=[CodeChunk.from_dict(c) for c in data['chunks']],
            model_state=data['model_state'],
            config=data['config'],
        )


def build_index_with_learning(
    directory: Path,
    output_path: Path,
    model: NeuralCodeSearchModel,
    tokenizer: UniversalTokenizer,
    file_extensions: List[str] = ['.py', '.md', '.txt', '.json', '.yaml', '.sh'],
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    num_epochs: int = 3,
    device: str = 'cpu',
):
    """
    Build search index with incremental learning.

    Trains model WHILE indexing on code-documentation pairs.
    This enables codebase-specific understanding.

    Args:
        directory: Root directory to index
        output_path: Where to save index
        model: NeuralCodeSearchModel instance
        tokenizer: UniversalTokenizer instance
        file_extensions: File types to index
        batch_size: Batch size for training
        learning_rate: Learning rate for incremental updates
        num_epochs: Passes over codebase during indexing
        device: Device to train on
    """
    device = torch.device(device)
    model = model.to(device)

    # Find all files
    print(f"🔍 Scanning {directory} for code files...")
    files = []
    for ext in file_extensions:
        files.extend(directory.rglob(f'*{ext}'))

    print(f"📁 Found {len(files)} files")

    # Extract all chunks
    print(f"\n📝 Extracting code chunks with structure preservation...")
    all_chunks = []
    for filepath in tqdm(files, desc="Extracting"):
        chunks = tokenizer.chunk_file(filepath)
        all_chunks.extend(chunks)

    print(f"✅ Extracted {len(all_chunks)} structured chunks")

    # Filter chunks with explanation targets for training
    training_chunks = [c for c in all_chunks if c.explanation_target]
    print(f"🎓 Found {len(training_chunks)} chunks with documentation for training")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Incremental learning: train on code-documentation pairs
    if training_chunks:
        print(f"\n🏋️  Phase 1: Incremental Learning on Code-Documentation Pairs")
        print(f"    Training for {num_epochs} epochs on {len(training_chunks)} examples")

        for epoch in range(num_epochs):
            # Shuffle training data
            import random
            random.shuffle(training_chunks)

            epoch_losses = []
            pbar = tqdm(range(0, len(training_chunks), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}")

            for i in pbar:
                batch_chunks = training_chunks[i:i+batch_size]

                # Encode chunks
                code_tokens_list = []
                explanation_tokens_list = []

                for chunk in batch_chunks:
                    code_tok, exp_tok = tokenizer.encode_chunk(chunk, max_len=512)
                    code_tokens_list.append(code_tok)
                    if exp_tok is not None:
                        explanation_tokens_list.append(exp_tok)

                if not code_tokens_list:
                    continue

                code_tokens = torch.stack(code_tokens_list).to(device)
                explanation_tokens = torch.stack(explanation_tokens_list).to(device) if explanation_tokens_list else None

                # Forward pass
                model.train()
                loss, stats = model(
                    code_tokens,
                    explanation_targets=explanation_tokens,
                    loss_weights=(1.0, 0.1, 0.01),
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(loss.item())

                # Update progress bar
                if len(epoch_losses) > 0:
                    pbar.set_postfix({
                        'loss': f"{epoch_losses[-1]:.4f}",
                        'exp': f"{stats['explanation_loss']:.3f}",
                        'kl': f"{stats['kl_loss']:.3f}",
                    })

            avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            print(f"  Epoch {epoch+1} average loss: {avg_loss:.4f}")

        print("✅ Incremental learning complete!")

    # Compute embeddings for ALL chunks
    print(f"\n🔨 Phase 2: Computing semantic embeddings for {len(all_chunks)} chunks")
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Encoding"):
            batch_chunks = all_chunks[i:i+batch_size]

            # Encode chunks
            code_tokens_list = []
            for chunk in batch_chunks:
                code_tok, _ = tokenizer.encode_chunk(chunk, max_len=512)
                code_tokens_list.append(code_tok)

            if not code_tokens_list:
                continue

            code_tokens = torch.stack(code_tokens_list).to(device)

            # Encode to semantic space
            z_semantic = model.encode_to_semantic(code_tokens)
            all_embeddings.append(z_semantic.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    print(f"✅ Computed {embeddings.shape[0]} semantic embeddings")

    # Create search index
    index = SearchIndex(
        embeddings=embeddings,
        chunks=all_chunks,
        model_state=model.state_dict(),
        config={
            'vocab_size': model.vocab_size,
            'latent_dim': model.latent_dim,
            'num_chunks': len(all_chunks),
            'num_trained': len(training_chunks),
        }
    )

    # Save index
    index.save(output_path)

    print(f"\n🎉 Indexing complete!")
    print(f"📊 Statistics:")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Trained on: {len(training_chunks)} chunks with documentation")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Index size: {output_path}")

    return index


def search_with_explanations(
    query: str,
    index: SearchIndex,
    model: NeuralCodeSearchModel,
    tokenizer: UniversalTokenizer,
    top_k: int = 10,
    device: str = 'cpu',
) -> List[Dict[str, Any]]:
    """
    Search with intelligent explanation generation.

    Args:
        query: Natural language or code query
        index: SearchIndex with embeddings
        model: Trained NeuralCodeSearchModel
        tokenizer: UniversalTokenizer
        top_k: Number of results
        device: Device for inference

    Returns:
        List of results with explanations:
        [{
            'filepath': str,
            'start_line': int,
            'end_line': int,
            'code': str,
            'explanation': str,  # Generated by model!
            'relevance': float,
            'metadata': dict,
        }]
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # Encode query
    # Create a dummy chunk for the query
    query_chunk = CodeChunk(
        filepath='<query>',
        start_line=0,
        end_line=0,
        code_text=query,
        explanation_target=None,
        language='text',
        chunk_type='query',
        metadata={}
    )

    query_tokens, _ = tokenizer.encode_chunk(query_chunk, max_len=512)
    query_tokens = query_tokens.unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode query to semantic space
        z_query = model.encode_to_semantic(query_tokens)  # (1, latent)

        # Compute cosine similarity with all embeddings
        query_norm = F.normalize(z_query, p=2, dim=-1)  # (1, latent)
        embeddings_norm = F.normalize(index.embeddings.to(device), p=2, dim=-1)  # (N, latent)

        similarities = (query_norm @ embeddings_norm.T).squeeze(0)  # (N,)

        # Get top-k
        top_scores, top_indices = similarities.topk(top_k)

        # Generate explanations for top-k results
        results = []

        print(f"\n🔎 Generating explanations for top {top_k} results...")
        for score, idx in zip(top_scores, top_indices):
            chunk = index.chunks[int(idx)]

            # Get semantic embedding for this chunk
            z_semantic = index.embeddings[int(idx)].unsqueeze(0).to(device)

            # Generate explanation
            explanation_tokens = model.generate_explanation(z_semantic, max_len=128)
            explanation = tokenizer.detokenize(explanation_tokens[0].cpu().tolist())

            # Clean up explanation (remove padding, special tokens)
            explanation = explanation.replace('<PAD>', '').replace('<UNK>', '').strip()

            result = {
                'filepath': chunk.filepath,
                'start_line': chunk.start_line,
                'end_line': chunk.end_line,
                'code': chunk.code_text[:500],  # Truncate for display
                'explanation': explanation,
                'relevance': float(score),
                'language': chunk.language,
                'chunk_type': chunk.chunk_type,
                'metadata': chunk.metadata,
            }
            results.append(result)

    return results


# ──────────────────────────────────────────────────────────────────────────────
#  CLI INTERFACE
# ──────────────────────────────────────────────────────────────────────────────

def cmd_index(args):
    """Index a codebase with incremental learning."""
    directory = Path(args.directory)
    output_path = Path(args.output) if args.output else Path("neural_index.pt")

    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return

    print(f"🦝 Neural Code Search - Incremental Learning Mode")
    print(f"📁 Directory: {directory}")
    print(f"💾 Output: {output_path}")

    # Initialize components
    tokenizer = UniversalTokenizer(chunk_size=512, stride=256)
    model = NeuralCodeSearchModel(
        vocab_size=VOCAB_SIZE,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"🤖 Model parameters: {param_count:,}")

    # Build index with learning
    index = build_index_with_learning(
        directory=directory,
        output_path=output_path,
        model=model,
        tokenizer=tokenizer,
        file_extensions=args.extensions.split(','),
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        device='cpu',
    )


def cmd_search(args):
    """Search with explanation generation."""
    index_path = Path(args.index) if args.index else Path("neural_index.pt")

    if not index_path.exists():
        print(f"❌ Index not found: {index_path}")
        print("   Run 'index' command first to create an index")
        return

    # Load index
    print(f"📂 Loading index from {index_path}...")
    index = SearchIndex.load(index_path)

    # Recreate model
    config = index.config
    model = NeuralCodeSearchModel(
        vocab_size=config.get('vocab_size', VOCAB_SIZE),
        embed_dim=64,
        hidden_dim=128,
        latent_dim=config['latent_dim'],
    )
    model.load_state_dict(index.model_state)

    tokenizer = UniversalTokenizer()

    # Search
    query = args.query
    print(f"\n🔎 Query: {query}")
    print(f"{'='*80}")

    results = search_with_explanations(
        query=query,
        index=index,
        model=model,
        tokenizer=tokenizer,
        top_k=args.top_k,
        device='cpu',
    )

    # Display results
    print(f"\n📋 Top {len(results)} Results:\n")
    for i, result in enumerate(results, 1):
        print(f"[{i}] {result['filepath']}:{result['start_line']}-{result['end_line']}")
        print(f"    Relevance: {result['relevance']:.3f}")
        print(f"    Language: {result['language']} | Type: {result['chunk_type']}")
        print(f"\n    💡 Explanation: {result['explanation']}")

        if args.show_code:
            print(f"\n    Code:")
            for line in result['code'].split('\n')[:10]:  # Show first 10 lines
                print(f"      {line}")

        print(f"\n{'-'*80}\n")


def cmd_explain(args):
    """Generate explanation for specific code location."""
    index_path = Path(args.index) if args.index else Path("neural_index.pt")

    if not index_path.exists():
        print(f"❌ Index not found: {index_path}")
        return

    # Parse location
    try:
        filepath, line = args.location.rsplit(':', 1)
        line = int(line)
    except:
        print(f"❌ Invalid location format. Use: <file>:<line>")
        return

    # Load index
    index = SearchIndex.load(index_path)

    # Find chunk at location
    matching_chunks = [
        (i, chunk) for i, chunk in enumerate(index.chunks)
        if chunk.filepath == filepath and chunk.start_line <= line <= chunk.end_line
    ]

    if not matching_chunks:
        print(f"❌ No code found at {filepath}:{line}")
        return

    idx, chunk = matching_chunks[0]

    # Load model
    config = index.config
    model = NeuralCodeSearchModel(
        vocab_size=config.get('vocab_size', VOCAB_SIZE),
        latent_dim=config['latent_dim'],
    )
    model.load_state_dict(index.model_state)
    model.eval()

    tokenizer = UniversalTokenizer()

    # Generate explanation
    print(f"\n📍 Code at {filepath}:{line}")
    print(f"{'='*80}")
    print(chunk.code_text[:500])
    print(f"\n{'-'*80}")

    with torch.no_grad():
        z_semantic = index.embeddings[idx].unsqueeze(0)
        explanation_tokens = model.generate_explanation(z_semantic, max_len=128)
        explanation = tokenizer.detokenize(explanation_tokens[0].tolist())
        explanation = explanation.replace('<PAD>', '').replace('<UNK>', '').strip()

    print(f"\n💡 Generated Explanation:")
    print(f"   {explanation}")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Neural Code Search - Intelligent code search with explanations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Index command
    parser_index = subparsers.add_parser('index', help='Index codebase with learning')
    parser_index.add_argument('directory', help='Directory to index')
    parser_index.add_argument('--output', '-o', help='Output index file (default: neural_index.pt)')
    parser_index.add_argument('--extensions', default='.py,.md,.txt,.json,.yaml,.sh',
                             help='File extensions to index (comma-separated)')
    parser_index.add_argument('--embed-dim', type=int, default=64, help='Embedding dimension')
    parser_index.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser_index.add_argument('--latent-dim', type=int, default=32, help='Latent dimension')
    parser_index.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser_index.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser_index.add_argument('--epochs', type=int, default=3, help='Training epochs')

    # Search command
    parser_search = subparsers.add_parser('search', help='Search with explanations')
    parser_search.add_argument('query', help='Search query (code or natural language)')
    parser_search.add_argument('--index', '-i', help='Index file (default: neural_index.pt)')
    parser_search.add_argument('--top-k', '-k', type=int, default=10, help='Number of results')
    parser_search.add_argument('--show-code', '-c', action='store_true', help='Show code snippets')

    # Explain command
    parser_explain = subparsers.add_parser('explain', help='Explain code at location')
    parser_explain.add_argument('location', help='File location (file:line)')
    parser_explain.add_argument('--index', '-i', help='Index file (default: neural_index.pt)')

    args = parser.parse_args()

    if args.command == 'index':
        cmd_index(args)
    elif args.command == 'search':
        cmd_search(args)
    elif args.command == 'explain':
        cmd_explain(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
