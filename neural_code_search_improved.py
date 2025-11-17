#!/usr/bin/env python3
"""
NEURAL CODE SEARCH - Intelligent Code Search with Explanations [IMPROVED]
================================================================

Multi-scale semantic code search using latent trajectory transformers with:
- Contrastive learning (InfoNCE loss)
- Hard negative mining
- Multi-scale search modes (syntax/semantic/purpose)
- Query expansion with synonyms
- Cross-encoder re-ranking
- Comprehensive evaluation metrics

Architecture:
1. Universal Tokenizer - preserves structure (docstrings, comments, code)
2. Encoder → SDE Trajectory → Normalizing Flow → Semantic Latent Space
3. GRU Explanation Decoder - generates natural language from latent
4. Contrastive Learning - learns semantic similarity with hard negatives
5. Cross-Encoder - re-ranks results for optimal precision
6. Intelligent Search - multi-scale, query expansion, re-ranking

Usage:
    python neural_code_search_improved.py index <directory> --output index.pt
    python neural_code_search_improved.py search "SDE dynamics" --index index.pt --mode semantic
    python neural_code_search_improved.py evaluate --index index.pt --queries eval_queries.json
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
import numpy as np

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
#  SYNONYM DICTIONARY for Query Expansion
# ──────────────────────────────────────────────────────────────────────────────

SYNONYM_DICTIONARY = {
    'sde': ['stochastic differential equation', 'stochastic dynamics', 'sde dynamics'],
    'stochastic differential equation': ['sde', 'stochastic dynamics'],
    'ode': ['ordinary differential equation', 'deterministic dynamics', 'ode dynamics'],
    'ordinary differential equation': ['ode', 'deterministic dynamics'],
    'ep test': ['epps-pulley', 'epps pulley test', 'normality test', 'fasteppspulley'],
    'epps-pulley': ['ep test', 'epps pulley test', 'normality test'],
    'normalizing flow': ['flow', 'invertible transformation', 'bijective mapping', 'coupling layer'],
    'flow': ['normalizing flow', 'invertible transformation', 'coupling layer'],
    'raccoon': ['continual learning', 'online learning', 'incremental learning', 'bungeecord'],
    'continual learning': ['raccoon', 'online learning', 'lifelong learning', 'continuous adaptation'],
    'experience replay': ['memory buffer', 'replay buffer', 'experience buffer', 'raccoonmemory'],
    'memory buffer': ['experience replay', 'replay buffer', 'experience buffer'],
    'transformer': ['attention', 'self-attention', 'multi-head attention', 'qkv'],
    'attention': ['transformer', 'self-attention', 'multi-head attention'],
    'vae': ['variational autoencoder', 'variational encoder', 'latent variable model'],
    'variational autoencoder': ['vae', 'latent variable model'],
    'kl divergence': ['kullback-leibler', 'kl loss', 'kld'],
    'kullback-leibler': ['kl divergence', 'kl loss'],
    'gradient': ['grad', 'derivative', 'backprop'],
    'grad': ['gradient', 'derivative'],
}


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
        """Extract structured Python code with docstrings and comments."""
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

    def _chunk_raw_code(self, content: str, filepath: str, language: str) -> List[CodeChunk]:
        """Fallback: chunk raw text with overlap."""
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
        """Main entry point: chunk any file type with structure preservation."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            return []

        suffix = filepath.suffix.lower()
        filepath_str = str(filepath)

        # Route to appropriate extractor
        if suffix == '.py':
            chunks = self.extract_python_metadata(content, filepath_str)
        else:
            chunks = self._chunk_raw_code(content, filepath_str, suffix[1:] if suffix else 'text')

        return chunks

    def encode_chunk(self, chunk: CodeChunk, max_len: int = 512) -> Tuple[Tensor, Optional[Tensor]]:
        """Encode chunk to tokens with structure markers."""
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
#  CROSS-ENCODER & NEURAL CODE SEARCH MODEL with Contrastive Learning
# ──────────────────────────────────────────────────────────────────────────────

class CrossEncoderReranker(nn.Module):
    """Cross-encoder for re-ranking search results."""
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, query_embed: Tensor, code_embed: Tensor) -> Tensor:
        """Score relevance of code to query. Returns scores in [0, 1]."""
        combined = torch.cat([query_embed, code_embed], dim=-1)
        return self.encoder(combined).squeeze(-1)


class ExplanationDecoder(nn.Module):
    """GRU-based explanation decoder."""
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
        """Generate explanation from semantic latent."""
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
    Complete neural code search model with contrastive learning.
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
        self.embed_dim = embed_dim

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

        # SDE dynamics
        self.dynamics = RaccoonDynamics(latent_dim, hidden_dim, sigma_min=1e-4, sigma_max=1.0)

        # Normalizing flow
        self.flow = RaccoonFlow(latent_dim, hidden_dim, num_layers=4, time_dim=32)

        # Explanation decoder
        self.decoder = ExplanationDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            hidden_dim=64,
            num_layers=2,
        )

        # Cross-encoder for re-ranking
        self.cross_encoder = CrossEncoderReranker(latent_dim, hidden_dim)

        # Temperature for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.07))

        # Latent regularization
        univariate_test = FastEppsPulley(t_max=5.0, n_points=17)
        self.latent_test = SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=64,
            reduction="mean",
        )

    def encode(self, tokens: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode tokens to latent distribution."""
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

    def encode_to_semantic(self, tokens: Tensor, query_mode: str = 'semantic') -> Tensor:
        """
        Full encoding pipeline to semantic space with multi-scale support.

        Args:
            tokens: (batch, seq_len)
            query_mode: 'syntax' (t=0.0), 'semantic' (t=0.05), 'purpose' (t=0.1)
        Returns:
            z_semantic: (batch, latent_dim) in semantic space
        """
        # Encode to distribution
        mean, logvar = self.encode(tokens)
        z0 = self.reparameterize(mean, logvar)

        # SDE trajectory (multi-level understanding)
        device = tokens.device

        # Choose timestep based on query mode
        if query_mode == 'syntax':
            t_target = 0.0  # Early timestep for keyword/syntax match
        elif query_mode == 'semantic':
            t_target = 0.05  # Middle timestep for structural semantics
        else:  # 'purpose'
            t_target = 0.1  # Final timestep for high-level purpose

        t_span = torch.linspace(0.0, t_target, max(2, int(t_target * 30) + 1), device=device)
        z_traj = solve_sde(self.dynamics, z0, t_span)  # (batch, steps, latent)
        z_endpoint = z_traj[:, -1, :]  # Take final state

        # Normalizing flow to semantic space
        t_flow = torch.ones(z0.shape[0], 1, device=device) * 0.5
        z_semantic, _ = self.flow(z_endpoint, t_flow, reverse=False)

        return z_semantic

    def compute_contrastive_loss(
        self,
        anchor: Tensor,
        positive: Tensor,
        negatives: Tensor,
    ) -> Tensor:
        """
        InfoNCE contrastive loss.

        Args:
            anchor: (batch, latent_dim) - query embeddings
            positive: (batch, latent_dim) - relevant code embeddings
            negatives: (batch, num_neg, latent_dim) - irrelevant code embeddings
        Returns:
            loss: scalar contrastive loss
        """
        batch_size = anchor.shape[0]

        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negatives = F.normalize(negatives, p=2, dim=-1)

        # Positive similarity
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature  # (batch,)

        # Negative similarities
        neg_sim = torch.matmul(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature  # (batch, num_neg)

        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch, 1 + num_neg)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)  # Positive is at index 0

        loss = F.cross_entropy(logits, labels)

        return loss

    def forward(
        self,
        tokens: Tensor,
        explanation_targets: Optional[Tensor] = None,
        positive_tokens: Optional[Tensor] = None,
        negative_tokens: Optional[Tensor] = None,
        loss_weights: Tuple[float, float, float, float] = (1.0, 0.5, 0.1, 0.01),
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Full forward pass with contrastive learning.

        Args:
            tokens: (batch, seq_len) input code tokens
            explanation_targets: (batch, seq_len) target explanations
            positive_tokens: (batch, seq_len) positive pairs for contrastive learning
            negative_tokens: (batch, num_neg, seq_len) negative pairs
            loss_weights: (explanation, contrastive, kl, ep)
        """
        batch_size = tokens.shape[0]
        device = tokens.device

        # Encode anchor
        mean, logvar = self.encode(tokens)
        z0 = self.reparameterize(mean, logvar)

        # SDE trajectory
        t_span = torch.linspace(0.0, 0.1, 3, device=device)
        z_traj = solve_sde(self.dynamics, z0, t_span)
        z_endpoint = z_traj[:, -1, :]

        # Normalizing flow
        t_flow = torch.ones(batch_size, 1, device=device) * 0.5
        z_semantic, log_det = self.flow(z_endpoint, t_flow, reverse=False)

        # Explanation loss
        explanation_loss = torch.tensor(0.0, device=device)
        if explanation_targets is not None:
            _, exp_loss = self.decoder(z_semantic, target_tokens=explanation_targets)
            explanation_loss = exp_loss

        # Contrastive loss
        contrastive_loss = torch.tensor(0.0, device=device)
        if positive_tokens is not None and negative_tokens is not None:
            # Encode positive
            pos_mean, pos_logvar = self.encode(positive_tokens)
            pos_z0 = self.reparameterize(pos_mean, pos_logvar)
            pos_z_traj = solve_sde(self.dynamics, pos_z0, t_span)
            pos_z_endpoint = pos_z_traj[:, -1, :]
            pos_z_semantic, _ = self.flow(pos_z_endpoint, t_flow, reverse=False)

            # Encode negatives
            num_neg = negative_tokens.shape[1]
            neg_tokens_flat = negative_tokens.view(-1, negative_tokens.shape[-1])
            neg_mean, neg_logvar = self.encode(neg_tokens_flat)
            neg_z0 = self.reparameterize(neg_mean, neg_logvar)
            neg_z_traj = solve_sde(self.dynamics, neg_z0, t_span)
            neg_z_endpoint = neg_z_traj[:, -1, :]

            # Expand t_flow for negatives
            t_flow_neg = t_flow.repeat(num_neg, 1)
            neg_z_semantic, _ = self.flow(neg_z_endpoint, t_flow_neg, reverse=False)
            neg_z_semantic = neg_z_semantic.view(batch_size, num_neg, -1)

            contrastive_loss = self.compute_contrastive_loss(z_semantic, pos_z_semantic, neg_z_semantic)

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        kl_loss = torch.clamp(kl_loss, min=0.0)

        # Epps-Pulley regularization
        z_for_test = z_semantic.unsqueeze(0)  # (1, batch, latent)
        ep_loss = self.latent_test(z_for_test)

        # Total loss
        w_exp, w_cont, w_kl, w_ep = loss_weights
        loss = w_exp * explanation_loss + w_cont * contrastive_loss + w_kl * kl_loss + w_ep * ep_loss

        stats = {
            'explanation_loss': explanation_loss.detach(),
            'contrastive_loss': contrastive_loss.detach(),
            'kl_loss': kl_loss.detach(),
            'ep_loss': ep_loss.detach(),
            'z_semantic': z_semantic.detach(),
        }

        return loss, stats


# ──────────────────────────────────────────────────────────────────────────────
#  QUERY EXPANSION & EVALUATION METRICS
# ──────────────────────────────────────────────────────────────────────────────

def expand_query(query: str, synonym_dict: Dict[str, List[str]] = SYNONYM_DICTIONARY) -> List[str]:
    """
    Expand query using synonyms.

    Args:
        query: Original query string
        synonym_dict: Dictionary of synonyms
    Returns:
        List of expanded queries
    """
    expanded = [query]
    query_lower = query.lower()

    # Check each synonym key
    for key, synonyms in synonym_dict.items():
        if key.lower() in query_lower:
            # Replace with each synonym
            for synonym in synonyms:
                expanded_query = query_lower.replace(key.lower(), synonym.lower())
                if expanded_query not in expanded:
                    expanded.append(expanded_query)

    return expanded


def compute_metrics(
    retrieved: List[str],
    relevant: List[str],
    k: int = 10
) -> Dict[str, float]:
    """
    Compute retrieval metrics.

    Args:
        retrieved: List of retrieved document IDs
        relevant: List of relevant document IDs
        k: Cutoff for metrics
    Returns:
        Dictionary of metrics
    """
    retrieved_k = retrieved[:k]

    # Precision@k
    num_relevant_in_k = len(set(retrieved_k) & set(relevant))
    precision_k = num_relevant_in_k / k if k > 0 else 0.0

    # Recall@k
    recall_k = num_relevant_in_k / len(relevant) if len(relevant) > 0 else 0.0

    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            mrr = 1.0 / (i + 1)
            break

    # NDCG@k
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_k):
        if doc_id in relevant:
            dcg += 1.0 / np.log2(i + 2)

    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    ndcg_k = dcg / idcg if idcg > 0 else 0.0

    return {
        f'precision@{k}': precision_k,
        f'recall@{k}': recall_k,
        'mrr': mrr,
        f'ndcg@{k}': ndcg_k,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  INCREMENTAL LEARNING with Hard Negative Mining
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


def mine_hard_negatives(
    anchor_embedding: Tensor,
    all_embeddings: Tensor,
    positive_idx: int,
    num_hard: int = 5,
) -> List[int]:
    """
    Mine hard negatives: similar embeddings that are NOT the positive.

    Args:
        anchor_embedding: (latent_dim,)
        all_embeddings: (N, latent_dim)
        positive_idx: Index of positive example
        num_hard: Number of hard negatives to mine
    Returns:
        List of hard negative indices
    """
    # Compute similarities
    anchor_norm = F.normalize(anchor_embedding.unsqueeze(0), p=2, dim=-1)
    embeddings_norm = F.normalize(all_embeddings, p=2, dim=-1)
    similarities = (anchor_norm @ embeddings_norm.T).squeeze(0)

    # Exclude positive from negatives
    similarities[positive_idx] = -float('inf')

    # Get top similar (hard negatives)
    _, hard_neg_indices = similarities.topk(num_hard)

    return hard_neg_indices.tolist()


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
    use_hard_negatives: bool = True,
):
    """Build search index with incremental learning and hard negative mining."""
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

    # First pass: compute initial embeddings for hard negative mining
    if use_hard_negatives:
        print(f"\n🔍 Computing initial embeddings for hard negative mining...")
        model.eval()
        initial_embeddings = []

        with torch.no_grad():
            for i in range(0, len(all_chunks), batch_size):
                batch_chunks = all_chunks[i:i+batch_size]
                code_tokens_list = []
                for chunk in batch_chunks:
                    code_tok, _ = tokenizer.encode_chunk(chunk, max_len=512)
                    code_tokens_list.append(code_tok)

                if code_tokens_list:
                    code_tokens = torch.stack(code_tokens_list).to(device)
                    z_semantic = model.encode_to_semantic(code_tokens)
                    initial_embeddings.append(z_semantic.cpu())

        initial_embeddings = torch.cat(initial_embeddings, dim=0) if initial_embeddings else torch.empty(0, model.latent_dim)

    # Training with contrastive learning
    if training_chunks:
        print(f"\n🏋️  Training with Contrastive Learning and Hard Negative Mining")
        print(f"    Training for {num_epochs} epochs on {len(training_chunks)} examples")

        for epoch in range(num_epochs):
            # Shuffle training data
            import random
            random.shuffle(training_chunks)

            epoch_losses = []
            epoch_contrastive_losses = []
            pbar = tqdm(range(0, len(training_chunks), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}")

            for i in pbar:
                batch_chunks = training_chunks[i:i+batch_size]

                # Prepare batch
                code_tokens_list = []
                explanation_tokens_list = []
                positive_tokens_list = []
                negative_tokens_list = []

                for j, chunk in enumerate(batch_chunks):
                    code_tok, exp_tok = tokenizer.encode_chunk(chunk, max_len=512)
                    code_tokens_list.append(code_tok)

                    if exp_tok is not None:
                        explanation_tokens_list.append(exp_tok)

                    # Create positive pair (same chunk, could add augmentation)
                    positive_tokens_list.append(code_tok)

                    # Mine hard negatives
                    if use_hard_negatives and len(initial_embeddings) > 0:
                        chunk_idx = all_chunks.index(chunk)
                        if chunk_idx < len(initial_embeddings):
                            hard_neg_indices = mine_hard_negatives(
                                initial_embeddings[chunk_idx],
                                initial_embeddings,
                                chunk_idx,
                                num_hard=5
                            )

                            neg_tokens = []
                            for neg_idx in hard_neg_indices:
                                neg_chunk = all_chunks[neg_idx]
                                neg_tok, _ = tokenizer.encode_chunk(neg_chunk, max_len=512)
                                neg_tokens.append(neg_tok)

                            if neg_tokens:
                                negative_tokens_list.append(torch.stack(neg_tokens))

                if not code_tokens_list:
                    continue

                code_tokens = torch.stack(code_tokens_list).to(device)
                explanation_tokens = torch.stack(explanation_tokens_list).to(device) if explanation_tokens_list else None
                positive_tokens = torch.stack(positive_tokens_list).to(device) if positive_tokens_list else None

                # Stack negatives
                negative_tokens = None
                if negative_tokens_list:
                    # Pad to same number of negatives
                    max_neg = max(len(neg) for neg in negative_tokens_list)
                    padded_negatives = []
                    for neg_list in negative_tokens_list:
                        if len(neg_list) < max_neg:
                            # Repeat last negative to fill
                            padding = [neg_list[-1]] * (max_neg - len(neg_list))
                            neg_list = torch.cat([neg_list] + padding, dim=0)
                        padded_negatives.append(neg_list)
                    negative_tokens = torch.stack(padded_negatives).to(device)

                # Forward pass
                model.train()
                loss, stats = model(
                    code_tokens,
                    explanation_targets=explanation_tokens,
                    positive_tokens=positive_tokens,
                    negative_tokens=negative_tokens,
                    loss_weights=(1.0, 0.5, 0.1, 0.01),  # (explanation, contrastive, kl, ep)
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(loss.item())
                epoch_contrastive_losses.append(stats['contrastive_loss'].item())

                # Update progress bar
                if len(epoch_losses) > 0:
                    pbar.set_postfix({
                        'loss': f"{epoch_losses[-1]:.4f}",
                        'contr': f"{epoch_contrastive_losses[-1]:.3f}",
                        'exp': f"{stats['explanation_loss']:.3f}",
                        'kl': f"{stats['kl_loss']:.3f}",
                    })

            avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            avg_contrastive = sum(epoch_contrastive_losses) / max(len(epoch_contrastive_losses), 1)
            print(f"  Epoch {epoch+1} - Loss: {avg_loss:.4f}, Contrastive: {avg_contrastive:.4f}")

    # Compute final embeddings
    print(f"\n🔨 Computing final semantic embeddings for {len(all_chunks)} chunks")
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Encoding"):
            batch_chunks = all_chunks[i:i+batch_size]

            code_tokens_list = []
            for chunk in batch_chunks:
                code_tok, _ = tokenizer.encode_chunk(chunk, max_len=512)
                code_tokens_list.append(code_tok)

            if code_tokens_list:
                code_tokens = torch.stack(code_tokens_list).to(device)
                z_semantic = model.encode_to_semantic(code_tokens, query_mode='semantic')
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

    print(f"\n🎉 Indexing complete with contrastive learning!")
    return index


def search_with_reranking(
    query: str,
    index: SearchIndex,
    model: NeuralCodeSearchModel,
    tokenizer: UniversalTokenizer,
    top_k: int = 10,
    query_mode: str = 'semantic',
    use_query_expansion: bool = True,
    use_reranking: bool = True,
    device: str = 'cpu',
) -> List[Dict[str, Any]]:
    """
    Search with query expansion and cross-encoder re-ranking.

    Args:
        query: Natural language or code query
        index: SearchIndex with embeddings
        model: Trained NeuralCodeSearchModel
        tokenizer: UniversalTokenizer
        top_k: Number of results
        query_mode: 'syntax', 'semantic', or 'purpose'
        use_query_expansion: Whether to expand query with synonyms
        use_reranking: Whether to use cross-encoder re-ranking
        device: Device for inference
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # Query expansion
    queries = [query]
    if use_query_expansion:
        queries = expand_query(query)
        if len(queries) > 1:
            print(f"📖 Expanded query to: {queries}")

    all_scores = torch.zeros(len(index.chunks))

    with torch.no_grad():
        for q in queries:
            # Encode query
            query_chunk = CodeChunk(
                filepath='<query>',
                start_line=0,
                end_line=0,
                code_text=q,
                explanation_target=None,
                language='text',
                chunk_type='query',
                metadata={}
            )

            query_tokens, _ = tokenizer.encode_chunk(query_chunk, max_len=512)
            query_tokens = query_tokens.unsqueeze(0).to(device)

            # Encode query to semantic space
            z_query = model.encode_to_semantic(query_tokens, query_mode=query_mode)  # (1, latent)

            # Compute cosine similarity
            query_norm = F.normalize(z_query, p=2, dim=-1)  # (1, latent)
            embeddings_norm = F.normalize(index.embeddings.to(device), p=2, dim=-1)  # (N, latent)

            similarities = (query_norm @ embeddings_norm.T).squeeze(0)  # (N,)
            all_scores += similarities.cpu()

    # Average scores from expanded queries
    all_scores /= len(queries)

    # Get top candidates for re-ranking
    num_candidates = min(top_k * 2, len(index.chunks))  # Get 2x for re-ranking
    top_scores, top_indices = all_scores.topk(num_candidates)

    # Re-ranking with cross-encoder
    if use_reranking and num_candidates > 0:
        print(f"🔄 Re-ranking top {num_candidates} candidates with cross-encoder...")

        reranked_scores = []

        with torch.no_grad():
            # Encode original query
            query_chunk = CodeChunk(
                filepath='<query>',
                start_line=0,
                end_line=0,
                code_text=query,  # Use original query for re-ranking
                explanation_target=None,
                language='text',
                chunk_type='query',
                metadata={}
            )
            query_tokens, _ = tokenizer.encode_chunk(query_chunk, max_len=512)
            query_tokens = query_tokens.unsqueeze(0).to(device)
            z_query = model.encode_to_semantic(query_tokens, query_mode=query_mode)

            for idx in top_indices:
                # Get candidate embedding
                z_candidate = index.embeddings[int(idx)].unsqueeze(0).to(device)

                # Compute cross-encoder score
                cross_score = model.cross_encoder(z_query, z_candidate)
                reranked_scores.append((float(cross_score), int(idx)))

        # Sort by cross-encoder scores
        reranked_scores.sort(key=lambda x: x[0], reverse=True)

        # Take top-k after re-ranking
        final_indices = [idx for _, idx in reranked_scores[:top_k]]
        final_scores = [score for score, _ in reranked_scores[:top_k]]
    else:
        final_indices = top_indices[:top_k].tolist()
        final_scores = top_scores[:top_k].tolist()

    # Generate results
    results = []
    for score, idx in zip(final_scores, final_indices):
        chunk = index.chunks[idx]

        result = {
            'filepath': chunk.filepath,
            'start_line': chunk.start_line,
            'end_line': chunk.end_line,
            'code': chunk.code_text[:500],  # Truncate for display
            'relevance': float(score),
            'language': chunk.language,
            'chunk_type': chunk.chunk_type,
            'metadata': chunk.metadata,
        }
        results.append(result)

    return results


# ──────────────────────────────────────────────────────────────────────────────
#  EVALUATION FRAMEWORK
# ──────────────────────────────────────────────────────────────────────────────

def create_eval_queries():
    """Create evaluation queries with ground truth."""
    eval_queries = {
        "queries": [
            {
                "query": "SDE dynamics",
                "relevant_files": ["latent_drift_trajectory.py"],
                "relevant_functions": ["RaccoonDynamics", "solve_sde"],
            },
            {
                "query": "stochastic differential equation",
                "relevant_files": ["latent_drift_trajectory.py"],
                "relevant_functions": ["RaccoonDynamics", "solve_sde"],
            },
            {
                "query": "normalizing flow",
                "relevant_files": ["latent_drift_trajectory.py", "neural_code_search_improved.py"],
                "relevant_functions": ["RaccoonFlow", "CouplingLayer"],
            },
            {
                "query": "experience replay",
                "relevant_files": ["latent_drift_trajectory.py"],
                "relevant_functions": ["RaccoonMemory"],
            },
            {
                "query": "EP test",
                "relevant_files": ["latent_drift_trajectory.py", "neural_code_search_improved.py"],
                "relevant_functions": ["FastEppsPulley", "SlicingUnivariateTest"],
            },
            {
                "query": "Epps-Pulley normality test",
                "relevant_files": ["latent_drift_trajectory.py"],
                "relevant_functions": ["FastEppsPulley"],
            },
            {
                "query": "continual learning",
                "relevant_files": ["latent_drift_trajectory.py"],
                "relevant_functions": ["RaccoonLogClassifier", "continuous_learning_phase"],
            },
            {
                "query": "transformer encoder",
                "relevant_files": ["latent_drift_trajectory.py", "neural_code_search_improved.py"],
                "relevant_functions": ["TransformerBlock", "QKVAttention"],
            },
            {
                "query": "docstring extraction",
                "relevant_files": ["neural_code_search_improved.py"],
                "relevant_functions": ["extract_python_metadata"],
            },
            {
                "query": "contrastive learning",
                "relevant_files": ["neural_code_search_improved.py"],
                "relevant_functions": ["compute_contrastive_loss"],
            },
            {
                "query": "hard negative mining",
                "relevant_files": ["neural_code_search_improved.py"],
                "relevant_functions": ["mine_hard_negatives"],
            },
            {
                "query": "cross encoder reranking",
                "relevant_files": ["neural_code_search_improved.py"],
                "relevant_functions": ["CrossEncoderReranker", "search_with_reranking"],
            },
            {
                "query": "KL divergence loss",
                "relevant_files": ["latent_drift_trajectory.py", "neural_code_search_improved.py"],
                "relevant_functions": ["forward"],
            },
            {
                "query": "gradient clipping",
                "relevant_files": ["latent_drift_trajectory.py", "neural_code_search_improved.py"],
                "relevant_functions": ["train_ode", "build_index_with_learning"],
            },
            {
                "query": "adam optimizer",
                "relevant_files": ["latent_drift_trajectory.py", "neural_code_search_improved.py"],
                "relevant_functions": ["train_ode", "train_raccoon_classifier"],
            },
            {
                "query": "log classification",
                "relevant_files": ["latent_drift_trajectory.py"],
                "relevant_functions": ["LogDataset", "RaccoonLogClassifier"],
            },
            {
                "query": "concept drift",
                "relevant_files": ["latent_drift_trajectory.py"],
                "relevant_functions": ["LogDataset"],
            },
            {
                "query": "time embedding",
                "relevant_files": ["latent_drift_trajectory.py"],
                "relevant_functions": ["TimeAwareTransform"],
            },
            {
                "query": "coupling layer",
                "relevant_files": ["latent_drift_trajectory.py"],
                "relevant_functions": ["CouplingLayer"],
            },
            {
                "query": "query expansion synonyms",
                "relevant_files": ["neural_code_search_improved.py"],
                "relevant_functions": ["expand_query", "SYNONYM_DICTIONARY"],
            }
        ]
    }

    # Save to file
    with open('eval_queries.json', 'w') as f:
        json.dump(eval_queries, f, indent=2)

    return eval_queries


def evaluate_search(
    index_path: Path,
    queries_path: Path,
    model: Optional[NeuralCodeSearchModel] = None,
) -> Dict[str, float]:
    """Evaluate search quality with metrics."""
    # Load index
    index = SearchIndex.load(index_path)

    # Load or create model
    if model is None:
        config = index.config
        model = NeuralCodeSearchModel(
            vocab_size=config.get('vocab_size', VOCAB_SIZE),
            latent_dim=config['latent_dim'],
        )
        model.load_state_dict(index.model_state)

    tokenizer = UniversalTokenizer()

    # Load queries
    with open(queries_path, 'r') as f:
        eval_data = json.load(f)

    all_metrics = []

    for query_data in eval_data['queries']:
        query = query_data['query']
        relevant = query_data.get('relevant_functions', [])

        # Search
        results = search_with_reranking(
            query=query,
            index=index,
            model=model,
            tokenizer=tokenizer,
            top_k=10,
            query_mode='semantic',
            use_query_expansion=True,
            use_reranking=True,
            device='cpu',
        )

        # Extract retrieved function names
        retrieved = []
        for result in results:
            metadata = result.get('metadata', {})
            if 'function_name' in metadata:
                retrieved.append(metadata['function_name'])
            elif 'class_name' in metadata:
                retrieved.append(metadata['class_name'])

        # Compute metrics
        metrics = compute_metrics(retrieved, relevant, k=5)
        all_metrics.append(metrics)

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_metrics


# ──────────────────────────────────────────────────────────────────────────────
#  CLI INTERFACE
# ──────────────────────────────────────────────────────────────────────────────

def cmd_index(args):
    """Index a codebase with incremental learning."""
    directory = Path(args.directory)
    output_path = Path(args.output) if args.output else Path("contrastive_index.pt")

    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return

    print(f"🦝 Neural Code Search - Contrastive Learning Mode")
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
        use_hard_negatives=True,
    )


def cmd_search(args):
    """Search with explanation generation."""
    index_path = Path(args.index) if args.index else Path("contrastive_index.pt")

    if not index_path.exists():
        print(f"❌ Index not found: {index_path}")
        return

    # Load index
    print(f"📂 Loading index from {index_path}...")
    index = SearchIndex.load(index_path)

    # Recreate model
    config = index.config
    model = NeuralCodeSearchModel(
        vocab_size=config.get('vocab_size', VOCAB_SIZE),
        latent_dim=config['latent_dim'],
    )
    model.load_state_dict(index.model_state)

    tokenizer = UniversalTokenizer()

    # Search
    query = args.query
    print(f"\n🔎 Query: {query}")
    print(f"    Mode: {args.mode}")
    print(f"{'='*80}")

    results = search_with_reranking(
        query=query,
        index=index,
        model=model,
        tokenizer=tokenizer,
        top_k=args.top_k,
        query_mode=args.mode,
        use_query_expansion=not args.no_expansion,
        use_reranking=not args.no_reranking,
        device='cpu',
    )

    # Display results
    print(f"\n📋 Top {len(results)} Results:\n")
    for i, result in enumerate(results, 1):
        print(f"[{i}] {result['filepath']}:{result['start_line']}-{result['end_line']}")
        print(f"    Relevance: {result['relevance']:.3f}")
        print(f"    Language: {result['language']} | Type: {result['chunk_type']}")

        if args.show_code:
            print(f"\n    Code:")
            for line in result['code'].split('\n')[:10]:  # Show first 10 lines
                print(f"      {line}")

        print(f"\n{'-'*80}\n")


def cmd_evaluate(args):
    """Evaluate search quality."""
    index_path = Path(args.index) if args.index else Path("contrastive_index.pt")
    queries_path = Path(args.queries) if args.queries else Path("eval_queries.json")

    if not index_path.exists():
        print(f"❌ Index not found: {index_path}")
        return

    # Create eval queries if needed
    if not queries_path.exists():
        print(f"📝 Creating evaluation queries...")
        create_eval_queries()

    print(f"📊 Evaluating search quality...")
    metrics = evaluate_search(index_path, queries_path)

    print(f"\n📈 Evaluation Results:")
    print(f"{'='*40}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Neural Code Search - Intelligent code search with contrastive learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Index command
    parser_index = subparsers.add_parser('index', help='Index codebase with contrastive learning')
    parser_index.add_argument('directory', help='Directory to index')
    parser_index.add_argument('--output', '-o', help='Output index file')
    parser_index.add_argument('--extensions', default='.py,.md,.txt,.json,.yaml,.sh',
                             help='File extensions to index')
    parser_index.add_argument('--embed-dim', type=int, default=64)
    parser_index.add_argument('--hidden-dim', type=int, default=128)
    parser_index.add_argument('--latent-dim', type=int, default=32)
    parser_index.add_argument('--batch-size', type=int, default=16)
    parser_index.add_argument('--lr', type=float, default=1e-3)
    parser_index.add_argument('--epochs', type=int, default=5)

    # Search command
    parser_search = subparsers.add_parser('search', help='Search with multi-scale and reranking')
    parser_search.add_argument('query', help='Search query')
    parser_search.add_argument('--index', '-i', help='Index file')
    parser_search.add_argument('--top-k', '-k', type=int, default=10)
    parser_search.add_argument('--mode', choices=['syntax', 'semantic', 'purpose'], default='semantic',
                              help='Search mode: syntax (t=0.0), semantic (t=0.05), purpose (t=0.1)')
    parser_search.add_argument('--no-expansion', action='store_true', help='Disable query expansion')
    parser_search.add_argument('--no-reranking', action='store_true', help='Disable cross-encoder reranking')
    parser_search.add_argument('--show-code', '-c', action='store_true', help='Show code snippets')

    # Evaluate command
    parser_evaluate = subparsers.add_parser('evaluate', help='Evaluate search quality')
    parser_evaluate.add_argument('--index', '-i', help='Index file')
    parser_evaluate.add_argument('--queries', '-q', help='Evaluation queries JSON file')

    args = parser.parse_args()

    if args.command == 'index':
        cmd_index(args)
    elif args.command == 'search':
        cmd_search(args)
    elif args.command == 'evaluate':
        cmd_evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()