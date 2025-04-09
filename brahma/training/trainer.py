import os
import sys
import math
import time
import logging
import torch
import wandb
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Tuple, Any
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import BrahmaForCausalLM, BrahmaConfig

class BrahmaTrainer:
    def __init__(
        self,
        model: BrahmaForCausalLM,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        max_steps: int = 100000,
        max_epochs: Optional[int] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        gradient_accumulation_steps: int = 1,
        gradient_clipping: float = 1.0,
        mixed_precision: bool = True,
        log_interval: int = 10,
        eval_interval: int = 500,
        save_interval: int = 1000,
        save_dir: str = "checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_wandb: bool = False,
        wandb_project: str = "brahma-llm",
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
    ):
        """
        Trainer for Brahma LLM.
        
        Args:
            model: Model to train
            train_dataloader: DataLoader for training
            eval_dataloader: DataLoader for evaluation
            optimizer: Optimizer to use (if None, AdamW is used)
            lr_scheduler: Learning rate scheduler (if None, cosine with warmup is used)
            max_steps: Maximum number of training steps
            max_epochs: Maximum number of training epochs (if set, overrides max_steps)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps for LR scheduler
            gradient_accumulation_steps: Number of steps to accumulate gradients
            gradient_clipping: Maximum gradient norm for clipping
            mixed_precision: Whether to use mixed precision training
            log_interval: Interval for logging training metrics
            eval_interval: Interval for evaluation
            save_interval: Interval for saving checkpoints
            save_dir: Directory to save checkpoints
            device: Device to use for training
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: W&B project name
            wandb_run_name: W&B run name
            wandb_config: Additional W&B config
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Move model to device
        self.device = device
        self.model.to(self.device)
        
        # Set up optimizer
        if optimizer is None:
            self.optimizer = self._create_optimizer(learning_rate, weight_decay)
        else:
            self.optimizer = optimizer
        
        # Set up learning rate scheduler
        if lr_scheduler is None:
            self.lr_scheduler = self._create_lr_scheduler(warmup_steps, max_steps)
        else:
            self.lr_scheduler = lr_scheduler
        
        # Training settings
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clipping = gradient_clipping
        
        # Mixed precision
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None
        
        # Logging and evaluation
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        
        # Checkpointing
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Weights & Biases
        self.use_wandb = use_wandb
        if use_wandb:
            wandb_config = wandb_config or {}
            wandb_config.update({
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "warmup_steps": warmup_steps,
                "max_steps": max_steps,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "mixed_precision": mixed_precision,
            })
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=wandb_config,
            )
        
        # State tracking
        self.global_step = 0
        self.epoch = 0
        
        # Set up logger
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_optimizer(self, learning_rate: float, weight_decay: float) -> Optimizer:
        """Create AdamW optimizer with weight decay applied only to non-bias and non-LayerNorm weights."""
        no_decay = ["bias", "LayerNorm.weight", "norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    def _create_lr_scheduler(self, warmup_steps: int, max_steps: int) -> LambdaLR:
        """Create a learning rate scheduler with linear warmup and cosine decay."""
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def save_checkpoint(self, path: str):
        """Save model, optimizer, and trainer state."""
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model, optimizer, and trainer state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        if checkpoint["lr_scheduler"] and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        
        if checkpoint["scaler"] and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler"])
        
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        
        self.logger.info(f"Loaded checkpoint from {path}")
    
    def train(self):
        """Start the training process."""
        self.logger.info("***** Starting training *****")
        self.logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        self.logger.info(f"  Num Epochs = {self.max_epochs if self.max_epochs else 'N/A'}")
        self.logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.max_steps}")
        
        self.model.train()
        
        # Calculate total steps based on epochs if provided
        if self.max_epochs is not None:
            total_steps = len(self.train_dataloader) * self.max_epochs
            self.max_steps = min(self.max_steps, total_steps)
        
        # Training loop
        accumulated_loss = 0
        step_count = 0
        
        while self.global_step < self.max_steps:
            self.epoch += 1
            
            for step, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with mixed precision if enabled
                with autocast(enabled=self.mixed_precision):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"] if "attention_mask" in batch else None,
                        labels=batch["labels"],
                    )
                    
                    loss = outputs[0] if isinstance(outputs, tuple) else outputs
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with mixed precision if enabled
                if self.mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update metrics
                accumulated_loss += loss.item()
                step_count += 1
                
                # Optimizer step and logging every gradient_accumulation_steps
                if (step + 1) % self.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                    # Gradient clipping
                    if self.gradient_clipping > 0:
                        if self.mixed_precision:
                            self.scaler.unscale_(self.optimizer)
                        
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clipping
                        )
                    
                    # Optimizer step
                    if self.mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # Learning rate scheduler step
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Update global step
                    self.global_step += 1
                    
                    # Log metrics
                    if self.global_step % self.log_interval == 0:
                        avg_loss = accumulated_loss / step_count
                        lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else 0
                        
                        self.logger.info(
                            f"Step: {self.global_step}, Loss: {avg_loss:.4f}, LR: {lr:.9f}"
                        )
                        
                        if self.use_wandb:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/learning_rate": lr,
                                "train/epoch": self.epoch,
                                "train/step": self.global_step,
                            })
                        
                        accumulated_loss = 0
                        step_count = 0
                    
                    # Evaluation
                    if self.eval_dataloader is not None and self.global_step % self.eval_interval == 0:
                        eval_metrics = self.evaluate()
                        
                        # Log evaluation metrics
                        self.logger.info(f"Evaluation results at step {self.global_step}:")
                        for metric_name, metric_value in eval_metrics.items():
                            self.logger.info(f"  {metric_name} = {metric_value:.4f}")
                        
                        if self.use_wandb:
                            wandb.log({
                                f"eval/{metric_name}": metric_value
                                for metric_name, metric_value in eval_metrics.items()
                            })
                        
                        # Set model back to training mode
                        self.model.train()
                    
                    # Save checkpoint
                    if self.global_step % self.save_interval == 0:
                        checkpoint_path = os.path.join(
                            self.save_dir, f"checkpoint-{self.global_step}.pt"
                        )
                        self.save_checkpoint(checkpoint_path)
                
                # Break if we've reached max steps
                if self.global_step >= self.max_steps:
                    break
            
            # Break if we've reached max steps or max epochs
            if self.global_step >= self.max_steps:
                break
        
        # Final save
        checkpoint_path = os.path.join(self.save_dir, "checkpoint-final.pt")
        self.save_checkpoint(checkpoint_path)
        
        self.logger.info("***** Training complete *****")
        
        return self.global_step, self.epoch
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate the model on the evaluation dataset."""
        self.logger.info("***** Running evaluation *****")
        
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"] if "attention_mask" in batch else None,
                labels=batch["labels"],
            )
            
            loss = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # Update metrics
            batch_size = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        perplexity = math.exp(avg_loss)
        
        metrics = {
            "loss": avg_loss,
            "perplexity": perplexity,
        }
        
        return metrics
