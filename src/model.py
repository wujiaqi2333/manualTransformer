import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, q_seq_len = q.size(0), q.size(1)
        k_seq_len = k.size(1)
        v_seq_len = v.size(1)

        # Linear projections with proper reshaping
        Q = self.w_q(q).view(batch_size, q_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, k_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, v_seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            # Ensure mask has the right shape
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # Add head dimension if needed
            scores.masked_fill_(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, self.d_model
        )

        output = self.w_o(attn_output)
        return output, attn_weights

class LinearAttention(nn.Module):
    """Linear attention variant for computational efficiency"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, q_seq_len = q.size(0), q.size(1)
        k_seq_len = k.size(1)
        v_seq_len = v.size(1)

        Q = self.w_q(q).view(batch_size, q_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, k_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, v_seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Linear attention approximation
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1

        # Compute attention using associative property
        KV = torch.einsum("bhnd,bhnk->bhdk", K, V)
        Z = 1.0 / (torch.einsum("bhnd,bhd->bhn", Q, K.sum(dim=2)) + 1e-6)
        V = torch.einsum("bhnd,bhdk,bhn->bhnd", Q, KV, Z)

        V = V.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.d_model)
        output = self.w_o(V)

        # Return dummy attention weights for compatibility
        attn_weights = torch.ones(batch_size, self.n_heads, q_seq_len, k_seq_len, device=q.device) / k_seq_len
        return output, attn_weights


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        return x + self.dropout(sublayer(self.layer_norm(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding as in Transformer-XL"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Learnable relative position embeddings
        self.embeddings = nn.Embedding(2 * max_len + 1, d_model)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = x.size(0), x.size(1)

        if positions is None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

        # Create relative position indices
        rel_pos = positions.unsqueeze(2) - positions.unsqueeze(1)
        rel_pos = rel_pos.clamp(-self.max_len, self.max_len) + self.max_len

        rel_pos_emb = self.embeddings(rel_pos)
        return x + rel_pos_emb.mean(dim=2)  # Simplified integration


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 attention_type: str = "standard", use_relative_pos: bool = False):
        super().__init__()

        self.attention_type = attention_type
        if attention_type == "standard":
            self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        elif attention_type == "linear":
            self.self_attention = LinearAttention(d_model, n_heads, dropout)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        self.feed_forward = PositionWiseFFN(d_model, d_ff, dropout)
        self.residual_norm1 = ResidualLayerNorm(d_model, dropout)
        self.residual_norm2 = ResidualLayerNorm(d_model, dropout)
        self.use_relative_pos = use_relative_pos

        if use_relative_pos:
            self.relative_pos_encoding = RelativePositionalEncoding(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_relative_pos:
            x = self.relative_pos_encoding(x)

        # Self attention with residual connection and layer norm
        x = self.residual_norm1(x, lambda x: self.self_attention(x, x, x, mask)[0])
        # Feed forward with residual connection and layer norm
        x = self.residual_norm2(x, self.feed_forward)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFFN(d_model, d_ff, dropout)

        self.residual_norm1 = ResidualLayerNorm(d_model, dropout)
        self.residual_norm2 = ResidualLayerNorm(d_model, dropout)
        self.residual_norm3 = ResidualLayerNorm(d_model, dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self attention
        x = self.residual_norm1(x, lambda x: self.self_attention(x, x, x, tgt_mask)[0])
        # Cross attention
        x = self.residual_norm2(x, lambda x: self.cross_attention(x, memory, memory, memory_mask)[0])
        # Feed forward
        x = self.residual_norm3(x, self.feed_forward)
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8,
                 num_layers: int = 6, d_ff: int = 2048, max_seq_len: int = 5000,
                 dropout: float = 0.1, attention_type: str = "standard",
                 use_relative_pos: bool = False, use_decoder: bool = False):
        super().__init__()

        self.d_model = d_model
        self.use_decoder = use_decoder

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, attention_type, use_relative_pos)
            for _ in range(num_layers)
        ])

        # Decoder layers (if needed)
        if use_decoder:
            self.decoder_layers = nn.ModuleList([
                TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])
            self.output_projection = nn.Linear(d_model, vocab_size)
        else:
            # For language modeling, project encoder output to vocabulary
            self.output_projection = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # Initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=self.d_model ** -0.5)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.weight, 1)
            torch.nn.init.constant_(module.bias, 0)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        # Encoder forward pass
        src_emb = self.token_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)

        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        enc_output = self.layer_norm(enc_output)

        if not self.use_decoder:
            # For encoder-only model (e.g., language modeling)
            output = self.output_projection(enc_output)
            return output

        # Decoder forward pass (if using decoder)
        if tgt is not None:
            tgt_emb = self.token_embedding(tgt) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoding(tgt_emb)
            tgt_emb = self.dropout(tgt_emb)

            dec_output = tgt_emb
            for layer in self.decoder_layers:
                dec_output = layer(dec_output, enc_output, tgt_mask, src_mask)
            dec_output = self.layer_norm(dec_output)

            output = self.output_projection(dec_output)
            return output

        return enc_output
