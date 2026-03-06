"""
EMA (Exponential Moving Average) for Diffusion Model Training
==============================================================
Based on best practices from EDM2 (Karras et al., arXiv:2312.02696).

EMA smooths the stochastic weight oscillations inherent to diffusion
training, producing 5-10× better sample quality at inference.

Usage:
    ema = EMAModel(model, decay=0.9999, use_warmup=True)
    for step, batch in enumerate(dataloader):
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
        ema.update(model, step=step)

    # For evaluation:
    ema.apply_to(model)  # copy EMA weights to model
    evaluate(model)
    ema.restore(model)   # restore training weights
"""

import torch
import copy


class EMAModel:
    """
    Exponential Moving Average of model parameters.

    Features:
    - Warmup-aware decay: decay_t = min(decay, (1+t)/(10+t))
    - CPU storage option to save GPU memory during training
    - Apply/restore for evaluation
    """

    def __init__(self, model, decay=0.9999, use_warmup=True, store_on_cpu=False):
        """
        Args:
            model: The training model
            decay: Target EMA decay rate (0.9999 recommended for 100K-500K steps)
            use_warmup: If True, use adaptive decay that starts low and increases
            store_on_cpu: If True, store EMA params on CPU to save GPU memory
        """
        self.decay = decay
        self.use_warmup = use_warmup
        self.store_on_cpu = store_on_cpu

        # Create shadow copy of model parameters
        self.shadow_params = {}
        self.backup_params = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                shadow = param.data.clone()
                if store_on_cpu:
                    shadow = shadow.cpu()
                self.shadow_params[name] = shadow

    def _get_decay(self, step):
        """Compute the decay rate, optionally with warmup."""
        if self.use_warmup:
            return min(self.decay, (1 + step) / (10 + step))
        return self.decay

    @torch.no_grad()
    def update(self, model, step=0):
        """Update EMA parameters after an optimizer step."""
        decay = self._get_decay(step)

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                src = param.data
                if self.store_on_cpu:
                    src = src.cpu()
                self.shadow_params[name].mul_(decay).add_(src, alpha=1 - decay)

    @torch.no_grad()
    def apply_to(self, model):
        """Copy EMA parameters to the model (for evaluation)."""
        self.backup_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.backup_params[name] = param.data.clone()
                shadow = self.shadow_params[name]
                if self.store_on_cpu:
                    shadow = shadow.to(param.device)
                param.data.copy_(shadow)

    @torch.no_grad()
    def restore(self, model):
        """Restore training parameters after evaluation."""
        for name, param in model.named_parameters():
            if name in self.backup_params:
                param.data.copy_(self.backup_params[name])
        self.backup_params = {}

    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {
            'shadow_params': self.shadow_params,
            'decay': self.decay,
            'use_warmup': self.use_warmup,
        }

    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.shadow_params = state_dict['shadow_params']
        self.decay = state_dict['decay']
        self.use_warmup = state_dict.get('use_warmup', True)
