import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import Transformer
from src.data_loader import load_wikitext2, create_mask
from src.utils import LearningRateScheduler, TrainingLogger, save_checkpoint, load_checkpoint


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, clip_grad=1.0):
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch_idx, (src, tgt) in enumerate(tqdm(dataloader, desc="Training")):
        src, tgt = src.to(device), tgt.to(device)

        # Create masks
        src_mask, tgt_mask = create_mask(src, tgt)

        # Forward pass
        optimizer.zero_grad()

        if model.use_decoder:
            # 对于encoder-decoder架构
            # 确保tgt输入比目标序列少一个token（用于teacher forcing）
            output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])
            # 计算损失 - 目标应该是tgt[:, 1:]
            output_flat = output.contiguous().view(-1, output.size(-1))
            target_flat = tgt[:, 1:].contiguous().view(-1)
        else:
            # 对于encoder-only架构（语言建模）
            output = model(src, src_mask=src_mask)
            # 计算损失 - 目标应该是整个tgt（因为预测下一个token）
            output_flat = output.contiguous().view(-1, output.size(-1))
            target_flat = tgt.contiguous().view(-1)

        loss = criterion(output_flat, target_flat)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item() * src.size(0)
        total_tokens += src.size(0)

    return total_loss / total_tokens


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating"):
            src, tgt = src.to(device), tgt.to(device)

            # 确保序列长度一致
            seq_len = min(src.size(1), tgt.size(1))
            src = src[:, :seq_len]
            tgt = tgt[:, :seq_len]

            src_mask, tgt_mask = create_mask(src, tgt)

            if model.use_decoder:
                output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])
                output_flat = output.contiguous().view(-1, output.size(-1))
                target_flat = tgt[:, 1:].contiguous().view(-1)
            else:
                output = model(src, src_mask=src_mask)
                output_flat = output.contiguous().view(-1, output.size(-1))
                target_flat = tgt.contiguous().view(-1)

            # 确保输出和目标有相同数量的元素
            min_len = min(output_flat.size(0), target_flat.size(0))
            output_flat = output_flat[:min_len]
            target_flat = target_flat[:min_len]

            loss = criterion(output_flat, target_flat)

            total_loss += loss.item() * src.size(0)
            total_tokens += src.size(0)

    return total_loss / total_tokens


def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description='Transformer Training on WikiText-2')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--d-ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--warmup-steps', type=int, default=4000)
    parser.add_argument('--attention-type', choices=['standard', 'linear'], default='standard')
    parser.add_argument('--use-relative-pos', action='store_true')
    parser.add_argument('--use-decoder', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    # 添加数据集路径参数
    parser.add_argument('--data-dir', type=str, default='dataset/wikitext-2',
                        help='Path to the dataset directory')

    # 解析命令行参数
    args = parser.parse_args()

    # 如果没有命令行参数（直接main运行），设置与bash命令相同的默认值
    if len(sys.argv) == 1:  # 只有脚本名，没有其他参数
        print("No command line arguments provided, using default values from bash command...")
        # 设置与bash命令相同的参数
        args.batch_size = 32
        args.seq_len = 128
        args.epochs = 20
        args.d_model = 512
        args.n_heads = 8
        args.num_layers = 6
        args.d_ff = 2048
        args.dropout = 0.1
        args.lr = 0.0001
        args.clip_grad = 1.0
        args.warmup_steps = 4000
        args.seed = 42
        args.use_decoder = True  # 对应 --use-decoder
        args.save_dir = 'checkpoints/base_model'
        args.attention_type = 'standard'
        args.use_relative_pos = False
        args.resume = None
        args.data_dir = '../dataset/wikitext-2'  # 添加默认数据路径

    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 打印数据集路径以便调试
    print(f"Dataset directory: {args.data_dir}")
    print(f"Dataset exists: {os.path.exists(args.data_dir)}")

    # 列出数据集目录中的文件
    if os.path.exists(args.data_dir):
        print(f"Files in dataset directory: {os.listdir(args.data_dir)}")

    # Load data
    print("Loading WikiText-2 dataset...")

    # 修改数据加载函数调用，传递数据集路径
    try:
        data_info = load_wikitext2(args.batch_size, args.seq_len, device, data_dir=args.data_dir)
    except TypeError:
        # 如果load_wikitext2不接受data_dir参数，我们需要修改数据加载方式
        print("Trying alternative data loading method...")
        data_info = load_wikitext2_with_custom_path(args.batch_size, args.seq_len, device, args.data_dir)

    vocab_size = data_info['vocab_size']

    print(f"Vocabulary size: {vocab_size}")
    print(f"Train batches: {len(data_info['train'])}")
    print(f"Val batches: {len(data_info['val'])}")

    # Initialize model
    model = Transformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        attention_type=args.attention_type,
        use_relative_pos=args.use_relative_pos,
        use_decoder=args.use_decoder
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=1)  # ignore padding
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LearningRateScheduler(optimizer, args.d_model, args.warmup_steps)

    start_epoch = 0
    best_val_loss = float('inf')

    # Resume from checkpoint if provided
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, args.resume)
        start_epoch += 1

    # Training logger
    logger = TrainingLogger()

    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_epoch(
            model, data_info['train'], criterion, optimizer, scheduler,
            device, args.clip_grad
        )

        # Validate
        val_loss = evaluate(model, data_info['val'], criterion, device)

        # Calculate perplexity
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)

        # Log metrics
        current_lr = scheduler.get_lr() if scheduler else args.lr
        logger.update(train_loss, val_loss, current_lr)

        print(f"Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f}")
        print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
        print(f"Learning Rate: {current_lr:.6f}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.save_dir, f'best_model_epoch_{epoch + 1}.pt')
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_path)

    # Plot training curves
    os.makedirs('results/training_curves', exist_ok=True)
    logger.plot_training_curves('results/training_curves/training_curves.png')
    logger.save_to_csv('results/training_curves/training_log.csv')

    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    test_loss = evaluate(model, data_info['test'], criterion, device)
    test_ppl = math.exp(test_loss)
    print(f"Test Loss: {test_loss:.4f}, Test PPL: {test_ppl:.2f}")


# 添加一个辅助函数来处理数据加载
def load_wikitext2_with_custom_path(batch_size, seq_len, device, data_dir):
    """使用自定义路径加载WikiText-2数据集，兼容新版torchtext"""
    import os
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def load_text_file(file_path):
        """读取文本文件并返回token列表"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # 简单的空格分词，你可以根据需要替换为更复杂的分词器
        tokens = text.split()
        return tokens

    def build_vocab(tokens):
        """构建词汇表"""
        vocab = {}
        vocab['<pad>'] = 0  # 填充token
        vocab['<unk>'] = 1  # 未知token
        vocab['<eos>'] = 2  # 句子结束token

        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

        return vocab

    def tokens_to_ids(tokens, vocab):
        """将tokens转换为id序列"""
        return [vocab.get(token, vocab['<unk>']) for token in tokens]

    def create_sequences(token_ids, seq_len):
        """创建训练序列"""
        sequences = []
        targets = []

        for i in range(0, len(token_ids) - seq_len, seq_len):
            seq = token_ids[i:i + seq_len]
            target = token_ids[i + 1:i + seq_len + 1]

            # 确保序列长度一致
            if len(seq) == seq_len and len(target) == seq_len:
                sequences.append(seq)
                targets.append(target)

        return torch.tensor(sequences), torch.tensor(targets)

    # 加载训练、验证和测试数据
    train_path = os.path.join(data_dir, 'wiki.train.tokens')
    valid_path = os.path.join(data_dir, 'wiki.valid.tokens')
    test_path = os.path.join(data_dir, 'wiki.test.tokens')

    print(f"Loading training data from: {train_path}")
    train_tokens = load_text_file(train_path)
    print(f"Training tokens: {len(train_tokens)}")

    print(f"Loading validation data from: {valid_path}")
    valid_tokens = load_text_file(valid_path)
    print(f"Validation tokens: {len(valid_tokens)}")

    print(f"Loading test data from: {test_path}")
    test_tokens = load_text_file(test_path)
    print(f"Test tokens: {len(test_tokens)}")

    # 构建词汇表（基于训练数据）
    print("Building vocabulary...")
    vocab = build_vocab(train_tokens)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # 转换为id序列
    train_ids = tokens_to_ids(train_tokens, vocab)
    valid_ids = tokens_to_ids(valid_tokens, vocab)
    test_ids = tokens_to_ids(test_tokens, vocab)

    # 创建序列
    print("Creating sequences...")
    train_sequences, train_targets = create_sequences(train_ids, seq_len)
    valid_sequences, valid_targets = create_sequences(valid_ids, seq_len)
    test_sequences, test_targets = create_sequences(test_ids, seq_len)

    print(f"Train sequences: {train_sequences.shape}")
    print(f"Valid sequences: {valid_sequences.shape}")
    print(f"Test sequences: {test_sequences.shape}")

    # 创建DataLoader
    train_dataset = TensorDataset(train_sequences, train_targets)
    val_dataset = TensorDataset(valid_sequences, valid_targets)
    test_dataset = TensorDataset(test_sequences, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'vocab_size': vocab_size
    }


if __name__ == "__main__":
    main()