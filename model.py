import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# INDAI sLLM v0.1
# Decoder-Only Transformer
# ============================================================

class IndAIConfig:

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    vocab_size = 32000

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    max_seq_len = 2048

    # --------------------------------------------------------
    # Transformer
    # --------------------------------------------------------

    d_model = 256
    n_layers = 6
    n_heads = 8
    d_ff = 1024

    # --------------------------------------------------------
    # Regularization
    # --------------------------------------------------------

    dropout = 0.1


# ============================================================
# CAUSAL SELF-ATTENTION
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, config):

        super().__init__()

        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(
            config.d_model,
            3 * config.d_model
        )

        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model
        )

        self.dropout = nn.Dropout(config.dropout)

        # ----------------------------------------------------
        # Causal mask
        # ----------------------------------------------------

        mask = torch.tril(
            torch.ones(
                config.max_seq_len,
                config.max_seq_len
            )
        )

        self.register_buffer(
            "mask",
            mask.view(
                1,
                1,
                config.max_seq_len,
                config.max_seq_len
            )
        )

    def forward(self, x):

        B, T, C = x.shape

        # ----------------------------------------------------
        # QKV
        # ----------------------------------------------------

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        # ----------------------------------------------------
        # Reshape into attention heads
        # ----------------------------------------------------

        q = q.view(
            B,
            T,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        # ----------------------------------------------------
        # Attention scores
        # ----------------------------------------------------

        attention_scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # ----------------------------------------------------
        # Causal masking
        #
        # Token at position i cannot see future tokens.
        # ----------------------------------------------------

        attention_scores = attention_scores.masked_fill(
            self.mask[:, :, :T, :T] == 0,
            float("-inf")
        )

        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )

        attention_weights = self.dropout(
            attention_weights
        )

        # ----------------------------------------------------
        # Weighted values
        # ----------------------------------------------------

        y = attention_weights @ v

        # ----------------------------------------------------
        # Merge heads
        # ----------------------------------------------------

        y = y.transpose(1, 2).contiguous()

        y = y.view(
            B,
            T,
            C
        )

        return self.out_proj(y)


# ============================================================
# FEED FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                config.d_model,
                config.d_ff
            ),

            nn.GELU(),

            nn.Linear(
                config.d_ff,
                config.d_model
            ),

            nn.Dropout(
                config.dropout
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.ln1 = nn.LayerNorm(
            config.d_model
        )

        self.attention = CausalSelfAttention(
            config
        )

        self.ln2 = nn.LayerNorm(
            config.d_model
        )

        self.feed_forward = FeedForward(
            config
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Pre-Norm + Residual
        # ----------------------------------------------------

        x = x + self.attention(
            self.ln1(x)
        )

        x = x + self.feed_forward(
            self.ln2(x)
        )

        return x


# ============================================================
# INDAI TRANSFORMER
# ============================================================

class IndAI(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.config = config

        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        # ----------------------------------------------------
        # Positional embedding
        # ----------------------------------------------------

        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model
        )

        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.ln_f = nn.LayerNorm(
            config.d_model
        )

        # ----------------------------------------------------
        # Language-model head
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )

        # ----------------------------------------------------
        # Weight tying
        #
        # Input embedding and output projection share weights.
        # This reduces parameters and is common in language models.
        # ----------------------------------------------------

        self.lm_head.weight = self.token_embedding.weight

        # ----------------------------------------------------
        # Initialize weights
        # ----------------------------------------------------

        self.apply(self._init_weights)

    # ========================================================
    # WEIGHT INITIALIZATION
    # ========================================================

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        input_ids,
        targets=None
    ):

        B, T = input_ids.shape

        if T > self.config.max_seq_len:

            raise ValueError(
                f"Sequence length {T} exceeds "
                f"maximum {self.config.max_seq_len}"
            )

        # ----------------------------------------------------
        # Positions
        # ----------------------------------------------------

        positions = torch.arange(
            0,
            T,
            device=input_ids.device
        )

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        token_embeddings = self.token_embedding(
            input_ids
        )

        position_embeddings = self.position_embedding(
            positions
        )

        x = (
            token_embeddings
            + position_embeddings
        )

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        x = F.dropout(
            x,
            p=self.config.dropout,
            training=self.training
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        for block in self.blocks:

            x = block(x)

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        x = self.ln_f(x)

        # ----------------------------------------------------
        # Vocabulary logits
        # ----------------------------------------------------

        logits = self.lm_head(x)

        loss = None

        # ----------------------------------------------------
        # Next-token prediction loss
        # ----------------------------------------------------

        if targets is not None:

            loss = F.cross_entropy(
                logits.view(
                    -1,
                    logits.size(-1)
                ),
                targets.view(-1)
            )

        return logits, loss


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 45)
    print("INDAI sLLM v0.1")
    print("DECODER-ONLY TRANSFORMER")
    print("=" * 45)

    config = IndAIConfig()

    model = IndAI(config)

    parameters = count_parameters(model)

    print()
    print("MODEL CONFIG")
    print("-" * 30)

    print(
        f"Vocabulary       : {config.vocab_size:,}"
    )

    print(
        f"Context length   : {config.max_seq_len:,}"
    )

    print(
        f"Embedding size   : {config.d_model}"
    )

    print(
        f"Transformer layers: {config.n_layers}"
    )

    print(
        f"Attention heads  : {config.n_heads}"
    )

    print(
        f"FFN size         : {config.d_ff}"
    )

    print()
    print(
        f"Parameters       : {parameters:,}"
    )

    print(
        f"Parameters (M)   : {parameters / 1_000_000:.2f}M"
    )

    # --------------------------------------------------------
    # Dummy batch
    # --------------------------------------------------------

    batch_size = 2
    sequence_length = 2048

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (
            batch_size,
            sequence_length
        )
    )

    targets = torch.randint(
        0,
        config.vocab_size,
        (
            batch_size,
            sequence_length
        )
    )

    print()
    print("FORWARD PASS TEST")
    print("-" * 30)

    print(
        f"Input shape     : {input_ids.shape}"
    )

    logits, loss = model(
        input_ids,
        targets
    )

    print(
        f"Logits shape    : {logits.shape}"
    )

    print(
        f"Loss            : {loss.item():.4f}"
    )

    print()
    print("=" * 45)
    print("INDAI MODEL TEST COMPLETE")
    print("=" * 45)