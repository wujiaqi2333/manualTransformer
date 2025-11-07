import torch
import os
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Iterator


class WikiText2Dataset(Dataset):
    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        sequence = self.data[start:end]
        target = self.data[start + 1:end + 1]
        return sequence, target


def read_wikitext2_file(file_path: str) -> Iterator[str]:
    """
    读取WikiText-2文件并返回文本行的迭代器
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 跳过空行和文档标题行（以 = 开头的行）
            if line.strip() and not line.startswith(' ='):
                yield line.strip()


def load_wikitext2_local(batch_size: int = 32, seq_len: int = 128,
                         device: torch.device = torch.device('cpu')) -> dict:
    """
    从本地dataset文件夹加载WikiText-2数据集

    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        device: 设备

    Returns:
        包含数据加载器和词汇表信息的字典
    """

    # 数据集路径
    dataset_path = os.path.join('dataset', 'wikitext-2')
    train_path = os.path.join(dataset_path, 'wiki.train.tokens')
    valid_path = os.path.join(dataset_path, 'wiki.valid.tokens')
    test_path = os.path.join(dataset_path, 'wiki.test.tokens')

    # 检查文件是否存在
    for path in [train_path, valid_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据集文件不存在: {path}")

    print("从本地文件加载WikiText-2数据集...")

    # 获取tokenizer
    tokenizer = get_tokenizer('basic_english')

    def yield_tokens(file_path: str) -> Iterator:
        """从文件生成token"""
        for text in read_wikitext2_file(file_path):
            yield tokenizer(text)

    # 从训练数据构建词汇表
    print("构建词汇表...")
    vocab = build_vocab_from_iterator(
        yield_tokens(train_path),
        specials=['<unk>', '<pad>']
    )
    vocab.set_default_index(vocab['<unk>'])

    def data_process(file_path: str) -> torch.Tensor:
        """处理数据文件并返回数值化的张量"""
        data_list = []
        for text in read_wikitext2_file(file_path):
            tokens = tokenizer(text)
            indices = [vocab[token] for token in tokens]
            if indices:  # 确保列表不为空
                data_list.append(torch.tensor(indices, dtype=torch.long))

        if data_list:
            return torch.cat(data_list)
        else:
            return torch.tensor([], dtype=torch.long)

    # 处理所有数据集
    print("处理训练数据...")
    train_data = data_process(train_path)
    print("处理验证数据...")
    val_data = data_process(valid_path)
    print("处理测试数据...")
    test_data = data_process(test_path)

    # 创建数据集
    train_dataset = WikiText2Dataset(train_data, seq_len)
    val_dataset = WikiText2Dataset(val_data, seq_len)
    test_dataset = WikiText2Dataset(test_data, seq_len)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True if device.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"数据集加载完成!")
    print(f"训练集大小: {len(train_data)} tokens")
    print(f"验证集大小: {len(val_data)} tokens")
    print(f"测试集大小: {len(test_data)} tokens")
    print(f"词汇表大小: {len(vocab)}")

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'vocab': vocab,
        'vocab_size': len(vocab),
        'pad_idx': vocab['<pad>']
    }


def load_wikitext2(batch_size: int = 32, seq_len: int = 128,
                   device: torch.device = torch.device('cpu')) -> dict:
    """
    加载WikiText-2数据集的主函数（从本地文件）

    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        device: 设备

    Returns:
        包含数据加载器和词汇表信息的字典
    """
    return load_wikitext2_local(batch_size, seq_len, device)


def create_mask(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    为源序列和目标序列创建mask

    Args:
        src: 源序列 [batch_size, src_len]
        tgt: 目标序列 [batch_size, tgt_len]
        pad_idx: padding的索引

    Returns:
        src_mask: 源序列mask [batch_size, 1, 1, src_len]
        tgt_mask: 目标序列mask [batch_size, 1, tgt_len, tgt_len]
    """
    batch_size, src_len = src.shape
    _, tgt_len = tgt.shape

    # 源序列mask: 非padding位置为True
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]

    # 目标序列mask: 结合padding mask和因果mask
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, tgt_len]

    # 创建下三角矩阵（因果mask）
    seq_len = tgt.size(1)
    nopeak_mask = torch.tril(torch.ones(1, seq_len, seq_len, device=src.device)).bool()  # [1, tgt_len, tgt_len]
    tgt_mask = tgt_pad_mask & nopeak_mask  # [batch_size, 1, tgt_len, tgt_len]

    return src_mask, tgt_mask