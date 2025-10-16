import utils
import math, random, time
from dataclasses import dataclass
import json
from pathlib import Path

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
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 512
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
    # WHAT: Resolve absolute path from repository root, not from mainrun/
    # WHY: train.py runs from mainrun/, so relative path needs to go up one level and be absolute
    rules_py = str((pathlib.Path(__file__).parent.parent / "rules" / "check_rules.py").resolve())
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

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head   = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop= nn.Dropout(cfg.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
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
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp  = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f      = nn.LayerNorm(cfg.d_model)
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
        pos = self.pos_emb[:, :T, :]
        x = self.drop(tok + pos)
        for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
        return logits, loss

def main():
    args = Hyperparameters()
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
    )
    model = GPT(cfg).to(device)
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

    # LR schedule: linear warmup then cosine decay to an LR floor
    # WHY: Warmup avoids early instability; LR floor prevents the LR from collapsing to zero.
    # Stability: increase warmup when using gradient accumulation
    # WHAT: Use longer warmup (20%) when grad_accum is active to smooth early updates
    # WHY: Accumulation changes update cadence; a gentler ramp reduces early spikes
    warmup_ratio = 0.2 if args.grad_accum_steps > 1 else 0.1
    warmup_steps = max(1, min(1000, int(warmup_ratio * max_steps)))
    lr_min = args.lr * 0.10  # 10% floor
    def _lr_schedule(step_idx: int) -> float:
        if step_idx <= warmup_steps:
            return args.lr * (step_idx / warmup_steps)
        progress = (step_idx - warmup_steps) / max(1, (max_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min + (args.lr - lr_min) * cosine

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
            # WHAT: Apply accumulated gradients to model parameters
            # WHY: Single update per effective batch instead of per micro-batch
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
                    # Update EMA state after successful optimizer step
                    with torch.no_grad():
                        for (n, p), s in zip(model.named_parameters(), ema_state):
                            if p.requires_grad:
                                s["data"].mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)
            
            # Only increment step counter after full gradient update
            step += 1
            
            # Manual LR schedule application (logs reflect current LR)
            for pg in opt.param_groups:
                pg["lr"] = _lr_schedule(step)

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
                import math
                val_ppl = math.exp(val_loss)
                writer.add_scalar("loss/val", float(val_loss), step)
                writer.add_scalar("metrics/perplexity", val_ppl, step)
                
                logger.log("validation_step",
                          step=step,
                          max_steps=max_steps,
                          loss=val_loss,
                          elapsed_time=elapsed)

    # Persist best validation result for this run
    try:
        (run_dir / "result.json").write_text(json.dumps({
            "best_val_loss": float(best_val)
        }, indent=2))
    except Exception:
        pass

if __name__ == "__main__":
    try:
        main()
    finally:
        if logger and hasattr(logger, 'file_handler'):
            logger.file_handler.close()
