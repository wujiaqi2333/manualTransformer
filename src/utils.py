import math

import matplotlib.pyplot as plt
import pandas as pd
import torch


class LearningRateScheduler:
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.d_model ** -0.5 * min(
            self.current_step ** -0.5,
            self.current_step * self.warmup_steps ** -1.5
        )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class TrainingLogger:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

    def update(self, train_loss: float, val_loss: float, lr: float):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)

    def plot_training_curves(self, save_path: str = None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot losses
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot learning rate
        ax2.plot(epochs, self.learning_rates, 'g-')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_to_csv(self, save_path: str):
        df = pd.DataFrame({
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'learning_rate': self.learning_rates
        })
        df.to_csv(save_path, index=False)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.current_step if scheduler else 0,
        'loss': loss
    }, path)


def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.current_step = checkpoint['scheduler_state_dict']
    return checkpoint['epoch'], checkpoint['loss']


def calculate_perplexity(loss: float) -> float:
    return math.exp(loss)


def model_size_in_mb(model) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 ** 2
    return size_mb