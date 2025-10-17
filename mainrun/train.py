import utils
import math, random, time
from dataclasses import dataclass
import json
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm
import structlog
from torch.utils.tensorboard import SummaryWriter

@dataclass
class Hyperparameters:
    block_size: int = 128
    batch_size: int = 64
    vocab_size: int = 16_000
    n_layer: int = 6  # Revert to 6 layers (best peak perf per Exp14)
    n_head: int = 8
    d_model: int = 512  # Reverted from 768 to 512 (optimal from Experiment 15)
    dropout: float = 0.1
    lr: float = 6e-3
    weight_decay: float = 0.0
    evals_per_epoch: int = 3
    
    # Gradient Accumulation: accumulate gradients over multiple micro-batches
    # WHAT: Process multiple small batches before updating weights
    # WHY: Simulates larger effective batch size for better gradient estimates
    # IMPACT: Better gradient estimates, improved convergence, memory efficient
    grad_accum_steps: int = 4  # Accumulate over 4 micro-batches
    
    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = "./logs/mainrun.log"

class WarmupCosineWithFloor:
    """Learning rate scheduler with warmup followed by cosine decay to a non-zero floor.
    
    WHAT: Single warmup phase followed by smooth cosine decay to eta_min
    WHY: Prevents early instability, provides smooth decay, maintains learning at end
    IMPACT: Better convergence, no late-epoch rebounds, stable final performance
    """
    def __init__(self, base_lr, eta_min, warmup_steps, total_steps):
        self.base_lr = base_lr
        self.eta_min = eta_min
        self.warmup_steps = max(1, warmup_steps)
        self.total = total_steps
        self.step_idx = 0
    
    def get_lr(self):
        """Get current learning rate based on step count."""
        t = self.step_idx
        if t < self.warmup_steps:
            # Linear warmup: lr = base_lr * (t + 1) / warmup_steps
            return self.base_lr * (t + 1) / self.warmup_steps
        
        # Cosine decay: progress from 0 to 1 over remaining steps
        progress = (t - self.warmup_steps) / max(1, self.total - self.warmup_steps)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return self.eta_min + (self.base_lr - self.eta_min) * cos
    
    def step(self):
        """Increment step counter."""
        self.step_idx += 1

class WarmupCosineWithTailSqueeze:
    """LR scheduler with warmup, cosine decay, and optional tail squeeze to 0.
    
    WHAT: Warmup → cosine decay to eta_min → linear decay to 0 in final tail_pct
    WHY: Prevents rebound by eliminating residual learning rate in final steps
    IMPACT: Smoother convergence, monotonic validation loss, eliminates late-epoch instability
    """
    def __init__(self, base_lr, eta_min, warmup_steps, total_steps, tail_squeeze_steps):
        self.base_lr = base_lr
        self.eta_min = eta_min
        self.warmup_steps = max(1, warmup_steps)
        self.total = total_steps
        self.tail_start = total_steps - tail_squeeze_steps
        self.tail_squeeze_steps = max(1, tail_squeeze_steps)
        self.step_idx = 0
    
    def get_lr(self):
        t = self.step_idx
        if t < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (t + 1) / self.warmup_steps
        elif t < self.tail_start:
            # Cosine decay to eta_min
            progress = (t - self.warmup_steps) / max(1, self.tail_start - self.warmup_steps)
            cos = 0.5 * (1 + math.cos(math.pi * progress))
            return self.eta_min + (self.base_lr - self.eta_min) * cos
        else:
            # Tail squeeze: linear decay from eta_min to 0
            progress = (t - self.tail_start) / self.tail_squeeze_steps
            return self.eta_min * (1 - progress)
    
    def step(self):
        self.step_idx += 1

def configure_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = open(log_file, 'w')
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    class DualLogger:
        def __init__(self, file_handler):
            self.file_handler = file_handler
            self.logger = structlog.get_logger()
            
        def log(self, event, **kwargs):
            log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
            self.file_handler.write(log_entry + "\n")
            self.file_handler.flush()
            
            if kwargs.get("prnt", True):
                if "step" in kwargs and "max_steps" in kwargs:
                    tqdm.write(f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={kwargs.get('loss', 'N/A'):.6f} time={kwargs.get('elapsed_time', 0):.2f}s")
                else:
                    parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ["prnt", "timestamp"]]
                    if parts:
                        tqdm.write(f"{event}: {', '.join(parts)}")
                    else:
                        tqdm.write(event)
    
    return DualLogger(file_handler)

logger = None

def get_titles(num_titles: int, seed: int, val_frac: float) -> str:
    ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir="./data").shuffle(seed=seed)
    titles = [row["title"].strip() for row in ds.take(num_titles)]
    n = int(num_titles * (1 - val_frac))
    return titles[:n], titles[n:]

def get_batch(split_ids: torch.Tensor, ptr: int, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    if ptr + span >= len(split_ids):
        ptr = 0
    batch = split_ids[ptr: ptr + span]
    x = batch[:-1].view(batch_size, block_size).to(device)
    y = batch[1:].view(batch_size, block_size).to(device)
    return x, y, ptr + block_size * batch_size

def iter_full_split(split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr: ptr + span]
        x = batch[:-1].view(batch_size, block_size).to(device)
        y = batch[1:].view(batch_size, block_size).to(device)
        yield x, y

def train_tokenizer(titles: list[str], vocab_size: int, unk_token: str = "<unk>", pad_token: str = "<pad>", eos_token: str = "<eos>") -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[pad_token, eos_token, unk_token]
    )
    tokenizer.train_from_iterator(titles, trainer)
    return tokenizer

# --- Rules guard: verify constraints (DO NOT REMOVE) ---
import json, pathlib, subprocess, sys
def _guard_rules(train_cfg):
    payload = json.dumps({"train_cfg": {
        "epochs": train_cfg["epochs"],
        "seed": train_cfg["seed"],
        "val_fraction": train_cfg["val_fraction"],
    }})
    # WHAT: Resolve path to rules/check_rules.py relative to train.py's location
    # WHY: rules/ folder is in the same directory as train.py
    rules_py = str((pathlib.Path(__file__).parent / "rules" / "check_rules.py").resolve())
    p = subprocess.run([sys.executable, rules_py, "--payload", payload])
    if p.returncode != 0: sys.exit(p.returncode)
# --- end guard ---

class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer
        self.stoi = {tok: i for tok, i in tokenizer.get_vocab().items()}
        self.itos = {i: tok for tok, i in tokenizer.get_vocab().items()}

    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self): return self.tk.get_vocab_size()

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float
    # Optional knobs (default keep current behavior)
    residual_scale: bool = False
    mlp_activation: str = "gelu"  # "gelu" | "swiglu"
    pre_ln: bool = False  # Pre-LN vs Post-LN architecture
    norm_type: str = "layernorm"  # "layernorm" | "rmsnorm"
    attention_type: str = "mha"  # "mha" | "mqa" | "gqa"
    pos_encoding: str = "learned"  # "learned" | "rope"

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # WHAT: RMSNorm normalizes by root mean square instead of mean and variance
        # WHY: More stable than LayerNorm, used in modern transformers (LLaMA, PaLM)
        # IMPACT: Better gradient flow and training stability
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.weight

def create_norm_layer(cfg: GPTConfig, d_model: int) -> nn.Module:
    """Create normalization layer based on config"""
    if cfg.norm_type == "rmsnorm":
        return RMSNorm(d_model)
    else:  # layernorm
        return nn.LayerNorm(d_model)

class RoPE(nn.Module):
    """Rotary Position Embeddings (RoPE)
    
    WHAT: Applies rotary position embeddings to query and key vectors
    WHY: More effective than learned positional embeddings, used in LLaMA, PaLM
    IMPACT: Better position encoding, improved performance, better extrapolation
    """
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Pre-compute frequency matrix for head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        # WHAT: Apply rotary position embeddings to input tensor
        # WHY: Encodes position information directly into the attention mechanism
        # IMPACT: Better position awareness, improved performance on longer sequences
        
        # x shape: (batch, n_heads, seq_len, head_dim)
        # We need to apply RoPE to the last dimension (head_dim)
        
        # Create position indices
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        
        # Create rotation matrix
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        # Apply rotation to even and odd dimensions of head_dim
        x_even = x[..., ::2]  # (batch, n_heads, seq_len, head_dim//2)
        x_odd = x[..., 1::2]  # (batch, n_heads, seq_len, head_dim//2)
        
        # Rotate
        x_rotated = torch.zeros_like(x)
        x_rotated[..., ::2] = x_even * cos - x_odd * sin
        x_rotated[..., 1::2] = x_even * sin + x_odd * cos
        
        return x_rotated

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head   = cfg.n_head
        self.attention_type = cfg.attention_type
        self.pos_encoding = cfg.pos_encoding
        
        # WHAT: Support different attention mechanisms (MHA, MQA, GQA)
        # WHY: MQA reduces memory usage and can improve performance
        # IMPACT: Enables modern attention patterns used in LLaMA, PaLM
        
        # WHAT: Initialize RoPE for position encoding
        # WHY: RoPE provides better position encoding than learned embeddings
        # IMPACT: Improved performance and better extrapolation to longer sequences
        if self.pos_encoding == "rope":
            self.rope = RoPE(self.head_dim, cfg.block_size)
        else:
            self.rope = None
        
        if cfg.attention_type == "mqa":
            # Multi-Query Attention: single key/value head, multiple query heads
            self.q = nn.Linear(cfg.d_model, cfg.d_model)  # n_head * head_dim
            self.k = nn.Linear(cfg.d_model, self.head_dim)  # 1 * head_dim
            self.v = nn.Linear(cfg.d_model, self.head_dim)  # 1 * head_dim
        elif cfg.attention_type == "gqa":
            # Grouped Query Attention: fewer key/value heads than query heads
            n_kv_heads = max(1, cfg.n_head // 4)  # 4:1 ratio like LLaMA-2
            self.n_kv_heads = n_kv_heads
            self.q = nn.Linear(cfg.d_model, cfg.d_model)
            self.k = nn.Linear(cfg.d_model, n_kv_heads * self.head_dim)
            self.v = nn.Linear(cfg.d_model, n_kv_heads * self.head_dim)
        else:  # mha - Multi-Head Attention (original)
            self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
            
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop= nn.Dropout(cfg.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        
        if self.attention_type == "mqa":
            # Multi-Query Attention
            q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = self.k(x).view(B, T, 1, self.head_dim).transpose(1, 2)
            v = self.v(x).view(B, T, 1, self.head_dim).transpose(1, 2)
            
            # Repeat k, v for all query heads
            k = k.repeat_interleave(self.n_head, dim=1)
            v = v.repeat_interleave(self.n_head, dim=1)
            
        elif self.attention_type == "gqa":
            # Grouped Query Attention
            q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = self.k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            
            # Repeat k, v for grouped query heads
            repeat_factor = self.n_head // self.n_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
            
        else:  # mha - Multi-Head Attention (original)
            qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
            q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        
        # WHAT: Apply RoPE to query and key vectors if enabled
        # WHY: RoPE provides better position encoding than learned embeddings
        # IMPACT: Improved position awareness and performance
        if self.rope is not None:
            # Apply RoPE to query and key vectors
            q = self.rope(q, T)
            k = self.rope(k, T)
        
        # Attention computation (same for all types)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.activation = cfg.mlp_activation
        if self.activation == "swiglu":
            # SwiGLU: two linear projections with gated SiLU
            self.w1 = nn.Linear(cfg.d_model, 2 * 4 * cfg.d_model)
            self.w2 = nn.Linear(4 * cfg.d_model, cfg.d_model)
            self.drop = nn.Dropout(cfg.dropout)
        else:
            self.net = nn.Sequential(
                nn.Linear(cfg.d_model, 4 * cfg.d_model),
                nn.GELU(),
                nn.Linear(4 * cfg.d_model, cfg.d_model),
                nn.Dropout(cfg.dropout),
            )
    def forward(self, x):
        if self.activation == "swiglu":
            u, v = self.w1(x).chunk(2, dim=-1)
            x = self.w2(F.silu(u) * v)
            return self.drop(x)
        return self.net(x)

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # WHAT: Use configurable normalization (LayerNorm or RMSNorm)
        # WHY: RMSNorm often provides better training stability and performance
        # IMPACT: Enables modern normalization techniques used in LLaMA, PaLM
        self.ln1 = create_norm_layer(cfg, cfg.d_model)
        self.ln2 = create_norm_layer(cfg, cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp  = MLP(cfg)
        self.use_residual_scale = cfg.residual_scale
        self.pre_ln = cfg.pre_ln
        if self.use_residual_scale:
            self.attn_alpha = nn.Parameter(torch.ones(1))
            self.mlp_alpha = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if self.pre_ln:
            # Pre-LN: normalize before attention/MLP, then add residual
            if self.use_residual_scale:
                x = x + self.attn_alpha * self.attn(self.ln1(x))
                x = x + self.mlp_alpha * self.mlp(self.ln2(x))
            else:
                x = x + self.attn(self.ln1(x))
                x = x + self.mlp(self.ln2(x))
        else:
            # Post-LN: apply attention/MLP, then normalize and add residual (original behavior)
            if self.use_residual_scale:
                x = x + self.attn_alpha * self.ln1(self.attn(x))
                x = x + self.mlp_alpha * self.ln2(self.mlp(x))
            else:
                x = x + self.ln1(self.attn(x))
                x = x + self.ln2(self.mlp(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        # WHAT: Default label smoothing factor for training-time CE loss only
        # WHY: Allows enabling regularization without affecting evaluation metrics
        # IMPACT: Improves generalization; evaluate() remains unchanged
        self.label_smoothing: float = 0.0
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        
        # WHAT: Conditional positional encoding based on config
        # WHY: RoPE is more effective than learned positional embeddings
        # IMPACT: Better position encoding, improved performance
        if cfg.pos_encoding == "rope":
            self.pos_emb = None  # RoPE is applied in attention layers
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
            
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        # WHAT: Use configurable normalization for final layer norm
        # WHY: Consistent normalization throughout the model
        # IMPACT: Enables RMSNorm for the final normalization layer
        self.ln_f      = create_norm_layer(cfg, cfg.d_model)
        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        tok = self.token_emb(idx)
        
        # WHAT: Apply positional encoding based on config
        # WHY: RoPE is applied in attention layers, learned embeddings are added here
        # IMPACT: Proper position encoding for the model
        if self.pos_emb is not None:
            pos = self.pos_emb[:, :T, :]
            x = self.drop(tok + pos)
        else:
            x = self.drop(tok)  # RoPE will be applied in attention layers
            
        for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            # WHAT: Apply label smoothing only during training forward pass
            # WHY: Regularize predictions; evaluation uses its own CE loss (untouched)
            # IMPACT: Potentially reduces overconfidence, improves val loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='mean',
                label_smoothing=getattr(self, 'label_smoothing', 0.0)
            )
        return logits, loss

def main():
    # Parse command line arguments for LR sweep
    parser = argparse.ArgumentParser(description='Train GPT-2 style model with LR optimization')
    parser.add_argument('--lr', type=float, default=6e-3, help='Base learning rate')
    parser.add_argument('--eta_min_factor', type=float, default=0.2, help='LR floor as fraction of base LR')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--warmup_pct', type=float, default=0.20, help='Warmup as percentage of total steps')
    parser.add_argument('--tail_squeeze', action='store_true', help='Enable linear decay to 0 in final 10% of steps')
    parser.add_argument('--tail_squeeze_pct', type=float, default=0.10, help='Percentage of steps for tail squeeze')
    parser.add_argument('--beta2_tail', type=float, default=None, help='Target beta2 for late training (None = no damping)')
    parser.add_argument('--tail_beta2_start_pct', type=float, default=0.30, help='Start beta2 interpolation at this fraction of total steps')
    parser.add_argument('--sweep_mode', action='store_true', help='Run LR sweep and exit')
    # Additional sweep flags (safe defaults = no behavior change unless provided)
    parser.add_argument('--residual_scale', action='store_true', help='Enable residual scaling (LayerScale-style)')
    parser.add_argument('--mlp_activation', type=str, default='gelu', choices=['gelu','swiglu'], help='MLP activation')
    parser.add_argument('--pre_ln', action='store_true', help='Use Pre-LN architecture (normalize before attention/MLP)')
    parser.add_argument('--norm_type', type=str, default='layernorm', choices=['layernorm','rmsnorm'], help='Normalization type')
    parser.add_argument('--attention_type', type=str, default='mha', choices=['mha','mqa','gqa'], help='Attention mechanism type')
    parser.add_argument('--pos_encoding', type=str, default='learned', choices=['learned','rope'], help='Position encoding type')
    parser.add_argument('--pack_tokens', action='store_true', help='Enable dynamic token packing (placeholder, no-op)')
    parser.add_argument('--dropout', type=float, default=None, help='Override dropout if provided')
    parser.add_argument('--weight_decay', type=float, default=None, help='Override weight decay if provided')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing epsilon for training loss (eval unaffected)')
    cli_args = parser.parse_args()
    
    # Create hyperparameters with CLI overrides
    args = Hyperparameters()
    args.lr = cli_args.lr
    args.grad_accum_steps = cli_args.grad_accum_steps
    # Optional overrides
    if cli_args.dropout is not None:
        args.dropout = cli_args.dropout
    if cli_args.weight_decay is not None:
        args.weight_decay = cli_args.weight_decay
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # --- Rules guard: verify constraints (DO NOT REMOVE) ---
    train_cfg = {
        "epochs": args.epochs,
        "seed": args.seed,
        "val_fraction": args.val_frac,
    }
    _guard_rules(train_cfg)
    # --- end guard ---
    
    global logger
    logger = configure_logging(args.log_file)
    
    hyperparams_dict = vars(args)
    logger.log("hyperparameters_configured", **hyperparams_dict)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log("device_info", device=device)

    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)
    
    eos_token = "<eos>"
    tok = BPETokenizer(train_tokenizer(train_titles+val_titles, args.vocab_size, eos_token=eos_token))
    train_text = eos_token.join(train_titles) + eos_token
    val_text = eos_token.join(val_titles) + eos_token
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    
    # Calculate batches and steps with gradient accumulation
    # WHAT: Adjust batch calculation for gradient accumulation
    # WHY: Maintain same effective batch size while using accumulation
    # IMPACT: More micro-batches per epoch, fewer gradient updates per epoch
    micro_batch_size = args.batch_size // args.grad_accum_steps  # 64 // 4 = 16
    batches = len(train_ids) // (args.block_size * micro_batch_size)  # More batches per epoch
    max_steps = args.epochs * batches // args.grad_accum_steps  # Steps = gradient updates (fewer)
    eval_interval = (batches // args.evals_per_epoch) // args.grad_accum_steps  # Adjust eval frequency
    logger.log("dataset_info",
               titles_count=len(train_titles),
               epochs=args.epochs,
               batches_per_epoch=batches,
               tokens_per_epoch=len(train_ids),
               vocab_size=tok.vocab_size)

    cfg = GPTConfig(
        vocab_size = tok.vocab_size,
        block_size = args.block_size,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        d_model    = args.d_model,
        dropout    = args.dropout,
        residual_scale = bool(cli_args.residual_scale),
        mlp_activation = cli_args.mlp_activation,
        pre_ln = bool(cli_args.pre_ln),
        norm_type = cli_args.norm_type,
        attention_type = cli_args.attention_type,
        pos_encoding = cli_args.pos_encoding,
    )
    model = GPT(cfg).to(device)
    # WHAT: Configure training-only label smoothing from CLI
    # WHY: Allows quick regularization without touching evaluate()
    # IMPACT: Potential generalization gains with minimal risk
    model.label_smoothing = max(0.0, float(cli_args.label_smoothing))
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)
    
    # set up a per-run directory under mainrun/runs with a stable symlink 'latest'
    # This does not change training logic; only organizes outputs for TB/reporting.
    # Resolve repo root so paths are stable regardless of current working directory
    repo_root = Path(__file__).resolve().parent.parent
    runs_root = repo_root / "mainrun" / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Initialize TensorBoard SummaryWriter to this specific run directory
    writer = SummaryWriter(log_dir=run_dir)
    # Maintain/update a 'latest' symlink for convenience (best-effort on non-Windows)
    latest_link = runs_root / "latest"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(run_dir, target_is_directory=True)
    except Exception:
        # On filesystems without symlink support, skip silently
        pass
    
    # once at startup
    param_count = sum(p.numel() for p in model.parameters())
    writer.add_text("model/param_count", f"{param_count:,}")
    writer.add_text("run/seed", str(args.seed))
    
    # Optimizer: switch to AdamW with decoupled weight decay and explicit no-decay groups
    # WHAT: AdamW typically stabilizes transformer training versus SGD; we also
    #       exclude bias/LayerNorm/Embedding from weight decay as common practice.
    # WHY: Expected faster convergence and better generalization with the same 7 epochs.
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("bias") or "ln" in name.lower() or "layernorm" in name.lower() or "emb" in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95)
    )
    
    # AMP: Configure mixed precision
    # WHAT: Prefer bf16 (no GradScaler needed) when supported; else fp16 + GradScaler
    # WHY: bf16 is numerically more stable and avoids scaler-related instabilities
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = None if use_bf16 else GradScaler()

    # LR schedule: single warmup → cosine decay to non-zero floor (optional tail squeeze)
    # WHAT: Configurable warmup percentage and optional tail squeeze for gentler convergence
    # WHY: Prevents late-epoch rebounds, maintains learning at end, smooth convergence
    # BUGFIX: Use actual number of optimizer steps, not calculated micro-batch steps
    total_optim_steps = batches * args.epochs  # Actual optimizer steps taken during training
    warmup_steps = round(cli_args.warmup_pct * total_optim_steps)
    eta_min = max(0.0, cli_args.eta_min_factor * args.lr)  # Allow 0 as valid floor
    tail_squeeze_steps = round(cli_args.tail_squeeze_pct * total_optim_steps) if cli_args.tail_squeeze else 0
    
    # Create LR scheduler instance
    if cli_args.tail_squeeze:
        lr_scheduler = WarmupCosineWithTailSqueeze(
            base_lr=args.lr,
            eta_min=eta_min,
            warmup_steps=warmup_steps,
            total_steps=total_optim_steps,
            tail_squeeze_steps=tail_squeeze_steps
        )
    else:
        lr_scheduler = WarmupCosineWithFloor(
            base_lr=args.lr,
            eta_min=eta_min,
            warmup_steps=warmup_steps,
            total_steps=total_optim_steps
        )

    # Persist immutable run metadata for reporting
    try:
        (run_dir / "run.json").write_text(json.dumps({
            "run_id": run_id,
            "device": device,
            "epochs": args.epochs,
            "seed": args.seed,
            "val_fraction": args.val_frac,
            "block_size": args.block_size,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size,  # Same due to accumulation
            "micro_batch_size": micro_batch_size,
            "vocab_size": tok.vocab_size,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "d_model": args.d_model,
            "dropout": args.dropout,
            "lr": args.lr,
            "eta_min_factor": cli_args.eta_min_factor,
            "eta_min": eta_min,
            "warmup_pct": cli_args.warmup_pct,
            "tail_squeeze": cli_args.tail_squeeze,
            "tail_squeeze_pct": cli_args.tail_squeeze_pct if cli_args.tail_squeeze else 0,
            "beta2_tail": cli_args.beta2_tail,
            "tail_beta2_start_pct": cli_args.tail_beta2_start_pct,
            "weight_decay": args.weight_decay,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2))
    except Exception:
        pass

    def evaluate():
        model.eval()
        losses = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
        model.train()
        return losses / len(val_text)

    ptr = 0
    step = 0
    t0 = time.time()
    best_val = float("inf")
    
    # Gradient Accumulation: Process multiple micro-batches before updating
    # WHAT: Accumulate gradients over grad_accum_steps micro-batches
    # WHY: Simulates larger effective batch size for better gradient estimates
    # IMPACT: Better gradient estimates, improved convergence, memory efficient
    # EMA for evaluation-only: maintain shadow weights for smoother validation curves
    # WHAT: Track EMA of parameters; swap to EMA weights only during evaluation
    # WHY: Produces smoother, more reliable val metrics without changing training dynamics
    ema_decay = 0.999
    ema_state = [{"name": n, "data": p.data.detach().clone()} for n, p in model.named_parameters() if p.requires_grad]

    skipped_update_count = 0
    effective_updates = 0
    
    # LR tracking for enhanced logging
    lr_history = []
    min_lr = float('inf')
    max_lr = 0.0

    for epoch in range(1, args.epochs + 1):
        for batch_idx in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.epochs}"):
            # Process micro-batches for gradient accumulation
            # WHAT: Process 4 micro-batches of size 16 before updating weights
            # WHY: Simulates batch size of 64 with memory efficiency
            accumulated_loss = 0.0
            # Always clear grads at the start of an accumulation window to avoid leakage
            opt.zero_grad(set_to_none=True)
            for micro_batch in range(args.grad_accum_steps):
                xb, yb, ptr = get_batch(train_ids, ptr, args.block_size, micro_batch_size, device)
                
                # AMP: Wrap forward pass with autocast for mixed precision
                # WHY: Enables 16-bit operations where safe, 32-bit where needed
                with autocast(dtype=amp_dtype):
                    _, loss = model(xb, yb)
                
                # Scale loss by accumulation steps to maintain correct gradient magnitude
                # WHAT: Divide loss by grad_accum_steps to prevent gradient explosion
                # WHY: Each micro-batch contributes 1/4 of the total gradient
                scaled_loss = loss / args.grad_accum_steps
                accumulated_loss += loss.item()
                
                # Backward pass with or without GradScaler depending on precision mode
                if scaler is None:  # bf16 path (no scaler)
                    scaled_loss.backward()
                else:  # fp16 path with scaler
                    scaler.scale(scaled_loss).backward()
            
            # Update weights after accumulating gradients from all micro-batches
            # WHAT: Apply accumulated gradients to model parameters with proper stepping order
            # WHY: LR scheduler should only step after successful optimizer updates
            step_successful = False
            if scaler is None:
                # bf16 path: clip and step directly
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                # Skip update if gradients are non-finite
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                if not torch.isfinite(total_norm):
                    opt.zero_grad(set_to_none=True)
                    skipped_update_count += 1
                else:
                    opt.step()
                    effective_updates += 1
                    step_successful = True
                    # Update EMA state after successful optimizer step
                    with torch.no_grad():
                        for (n, p), s in zip(model.named_parameters(), ema_state):
                            if p.requires_grad:
                                s["data"].mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)
            else:
                # fp16 path: unscale, clip, sanity-check, then step via scaler
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                # If any grad is non-finite after unscale, skip this step
                found_inf = False
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        found_inf = True
                        break
                if found_inf:
                    scaler.update()  # still update internal scale to recover
                    opt.zero_grad(set_to_none=True)
                    skipped_update_count += 1
                else:
                    scaler.step(opt)
                    scaler.update()
                    effective_updates += 1
                    step_successful = True
                    # Update EMA state after successful optimizer step
                    with torch.no_grad():
                        for (n, p), s in zip(model.named_parameters(), ema_state):
                            if p.requires_grad:
                                s["data"].mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)
            
            # Only increment step counter and update LR after successful optimizer step
            if step_successful:
                step += 1
                # Update LR using new scheduler (call step() after optimizer.step())
                lr_scheduler.step()
                current_lr = lr_scheduler.get_lr()
                for pg in opt.param_groups:
                    pg["lr"] = current_lr
                
                # Track LR for enhanced logging
                lr_history.append(current_lr)
                min_lr = min(min_lr, current_lr)
                max_lr = max(max_lr, current_lr)
                
                # Beta2 damping: reduce second-moment estimates in late training
                # WHAT: Linearly interpolate beta2 from base (0.95) to target (beta2_tail) 
                # WHY: Prevents optimizer from being too conservative near end of training
                # IMPACT: Better generalization and convergence in final epochs
                if cli_args.beta2_tail is not None:
                    beta2_start_step = int(cli_args.tail_beta2_start_pct * total_optim_steps)
                    if step >= beta2_start_step:
                        # Linear interpolation from base beta2 (0.95) to beta2_tail
                        progress = (step - beta2_start_step) / (total_optim_steps - beta2_start_step)
                        current_beta2 = 0.95 + (cli_args.beta2_tail - 0.95) * progress
                        # Update optimizer beta2 for all parameter groups
                        for pg in opt.param_groups:
                            pg['betas'] = (0.9, current_beta2)

            elapsed = time.time() - t0
            
            # Calculate tokens per second based on effective batch size
            effective_batch_size = args.batch_size  # Same as before due to accumulation
            tokens_per_sec = (effective_batch_size * args.block_size) / elapsed if elapsed > 0 else 0
            writer.add_scalar("loss/train", float(accumulated_loss / args.grad_accum_steps), step)
            writer.add_scalar("lr", opt.param_groups[0]["lr"], step)
            writer.add_scalar("perf/tokens_per_sec", tokens_per_sec, step)
            # Stability counters (cumulative)
            writer.add_scalar("train/skipped_update_cum", skipped_update_count, step)
            writer.add_scalar("train/effective_updates_cum", effective_updates, step)
            
            logger.log("training_step",
                      step=step,
                      max_steps=max_steps,
                      loss=accumulated_loss / args.grad_accum_steps,
                      elapsed_time=elapsed,
                      prnt=False)

            if step == 1 or step % eval_interval == 0 or step == max_steps:
                # Swap to EMA weights for evaluation, then swap back
                # Save current params
                with torch.no_grad():
                    current_weights = [p.data.detach().clone() for p in model.parameters() if p.requires_grad]
                    # Load EMA
                    i = 0
                    for p in model.parameters():
                        if p.requires_grad:
                            p.data.copy_(ema_state[i]["data"]) ; i += 1
                val_loss = evaluate()
                # Restore current params
                with torch.no_grad():
                    i = 0
                    for p in model.parameters():
                        if p.requires_grad:
                            p.data.copy_(current_weights[i]) ; i += 1
                if val_loss < best_val:
                    best_val = val_loss
                
                # after each validation
                val_ppl = math.exp(val_loss)
                writer.add_scalar("loss/val", float(val_loss), step)
                writer.add_scalar("metrics/perplexity", val_ppl, step)
                
                logger.log("validation_step",
                          step=step,
                          max_steps=max_steps,
                          loss=val_loss,
                          elapsed_time=elapsed)

    # Log final statistics and warnings
    final_val_loss = evaluate()  # Get final validation loss
    
    # Persist best validation result for this run
    try:
        (run_dir / "result.json").write_text(json.dumps({
            "best_val_loss": float(best_val),
            "final_val_loss": float(final_val_loss),
            "rebound": float(final_val_loss - best_val),
            "skipped_updates": int(skipped_update_count),
            "lr_min": float(min_lr if lr_history else 0.0),
            "lr_max": float(max_lr if lr_history else 0.0),
            "last5_lrs": [float(x) for x in lr_history[-5:]],
            "config": {
                "lr": float(args.lr),
                "eta_min_factor": float(cli_args.eta_min_factor),
                "warmup_pct": float(cli_args.warmup_pct),
                "tail_squeeze": bool(cli_args.tail_squeeze),
                "tail_squeeze_pct": float(cli_args.tail_squeeze_pct if cli_args.tail_squeeze else 0.0),
                "grad_accum_steps": int(args.grad_accum_steps),
                "beta2_tail": None if cli_args.beta2_tail is None else float(cli_args.beta2_tail),
                "tail_beta2_start_pct": float(cli_args.tail_beta2_start_pct),
                "residual_scale": bool(cli_args.residual_scale),
                "mlp_activation": str(cli_args.mlp_activation),
                "pack_tokens": bool(cli_args.pack_tokens),
                "dropout": float(args.dropout),
                "weight_decay": float(args.weight_decay)
            }
        }, indent=2))
    except Exception:
        pass
    logger.log("training_complete",
               final_val_loss=final_val_loss,
               best_val_loss=best_val,
               skipped_updates=skipped_update_count,
               effective_updates=effective_updates,
               total_steps=step)
    
    # Warning: Check for late-epoch rebound
    if final_val_loss > best_val + 0.03:
        logger.log("warning_late_rebound",
                   final_val=final_val_loss,
                   best_val=best_val,
                   difference=final_val_loss - best_val,
                   prnt=True)
    
    # Print summary table for LR sweep
    print(f"\n=== Training Summary ===")
    print(f"Base LR: {args.lr:.2e}")
    print(f"Eta Min: {eta_min:.2e} ({cli_args.eta_min_factor:.1f}x base)")
    print(f"Grad Accum Steps: {args.grad_accum_steps}")
    print(f"Total Optim Steps: {total_optim_steps}")
    print(f"Warmup Steps: {warmup_steps}")
    print(f"LR Range: {min_lr:.6f} - {max_lr:.6f}")
    print(f"Last 5 LRs: {[f'{lr:.6f}' for lr in lr_history[-5:]]}")
    if cli_args.beta2_tail is not None:
        final_beta2 = opt.param_groups[0]['betas'][1]
        print(f"Final Beta2: {final_beta2:.4f}")
    print(f"Final Val Loss: {final_val_loss:.6f}")
    print(f"Best Val Loss: {best_val:.6f}")
    print(f"Skipped Updates: {skipped_update_count}")
    print(f"Effective Updates: {effective_updates}")
    
    # Log to sweep file if in sweep mode
    if cli_args.sweep_mode:
        sweep_log = Path("logs/lr_sweep.jsonl")
        sweep_log.parent.mkdir(parents=True, exist_ok=True)
        with open(sweep_log, "a") as f:
            f.write(json.dumps({
                "base_lr": args.lr,
                "eta_min": eta_min,
                "accum": args.grad_accum_steps,
                "final_val": final_val_loss,
                "best_val": best_val,
                "skipped_updates": skipped_update_count
            }) + "\n")

if __name__ == "__main__":
    try:
        main()
    finally:
        if logger and hasattr(logger, 'file_handler'):
            logger.file_handler.close()
