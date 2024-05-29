import math
import torch
import torch.nn as nn
from typing import Optional


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(
            in_features=input_dim,
            out_features=hidden_dim,
        )
        self.activation = torch.nn.SiLU()
        self.linear2 = nn.Linear(
            in_features=hidden_dim,
            out_features=out_dim,
        )

    def forward(self, x):
        hidden = self.linear1(x)
        hidden = self.activation(hidden)
        out = self.linear2(hidden)
        return out


class PositionalEncoding(nn.Module):
    """Inherit from [OpenNMT](https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/embeddings.py)

    Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dim (int): embedding size
    """

    def __init__(self, d_model: int, enc_type, max_len=512):
        if d_model % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(d_model)
            )
        if enc_type == "SinusoidalInterleaved":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(
                (
                    torch.arange(0, d_model, 2, dtype=torch.float)
                    * -(math.log(10000.0) / d_model)
                )
            )
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
        elif enc_type == "SinusoidalConcat":
            half_dim = d_model // 2
            pe = math.log(10000) / (half_dim - 1)
            pe = torch.exp(torch.arange(half_dim, dtype=torch.float) * -pe)
            pe = torch.arange(max_len, dtype=torch.float).unsqueeze(1) * pe.unsqueeze(0)
            pe = torch.cat([torch.sin(pe), torch.cos(pe)], dim=1).view(max_len, -1)
        else:
            raise ValueError(
                "Choice of Position encoding is SinusoidalInterleaved or"
                " SinusoidalConcat."
            )
        pe = pe.unsqueeze(1)  # we keep pe (len x batch x dim) for back comp
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.d_model = d_model

    def forward(self, emb, step=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, seq_len, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """
        pe = self.pe.transpose(0, 1)  # (batch x len x dim)
        # emb = emb * math.sqrt(self.d_model)
        step = step or 0
        if pe.size(1) < step + emb.size(1):
            raise ValueError(
                f"Sequence is {emb.size(1) + step} but PositionalEncoding is"
                f" limited to {self.pe.size(1)}. See max_len argument."
            )
        emb = emb + pe[:, step : emb.size(1) + step, :]
        return emb


class GoalGenerationModule(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        feedforward_dim: int,
        activation: str = 'gelu',
    ):
        super().__init__()
        # self.positional_encoding = PositionalEncoding(
        #     d_model=d_model,
        #     enc_type='SinusoidalConcat',
        # )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        layernorm = nn.LayerNorm(normalized_shape=d_model)
        self.module = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            norm=layernorm,
        )

    def forward(
        self,
        decoder_hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: torch.LongTensor,
    ):
        device = decoder_hidden_states.device
        decoder_hidden_shape = decoder_hidden_states.shape[:2]
        encoder_hidden_shape = encoder_hidden_states.shape[:2]
        decoder_padding_mask = torch.zeros(*decoder_hidden_shape, dtype=torch.bool, device=decoder_hidden_states.device)
        encoder_padding_mask = ~encoder_attention_mask.to(dtype=torch.bool)
        decoder_attn_mask = nn.Transformer.generate_square_subsequent_mask(decoder_hidden_states.shape[1]).to(device=device)
        cross_attn_mask = torch.zeros(
            decoder_hidden_shape[1],
            encoder_hidden_shape[1],
            dtype=torch.bool,
            device=encoder_hidden_states.device
        )
        # decoder_input_embeds = self.positional_encoding(decoder_input_embeds)
        attention_outputs = self.module(
            tgt=decoder_hidden_states,
            memory=encoder_hidden_states,
            tgt_mask=decoder_attn_mask,
            memory_mask=cross_attn_mask,
            tgt_key_padding_mask=decoder_padding_mask,
            memory_key_padding_mask=encoder_padding_mask,
        )
        return attention_outputs


class KnowledgeInjectionModule(nn.Module):
    def __init__(
        self,
        encoder_hidden_dim: int,
        inference_hidden_dim: int,
        decoder_hidden_dim: int,
        sep_token_id: int,
    ):
        super().__init__()
        self.sep_token_id = sep_token_id
        self.inference_rnn = nn.LSTMCell(
            input_size=2 * encoder_hidden_dim + decoder_hidden_dim,
            hidden_size=inference_hidden_dim,
        )

    def attention(
        self,
        tensor: torch.FloatTensor,
        encoder_outputs: torch.FloatTensor,
        encoder_outputs_masks: Optional[torch.LongTensor] = None,
    ):
        assert encoder_outputs.shape[:2] == encoder_outputs_masks.shape[:2], "'encoder_outputs' and 'encoder_outputs_masks' shapes don't match."
        similarity = torch.bmm(tensor, encoder_outputs.transpose(1, 2))
        similarity /= math.sqrt(tensor.shape[-1])
        if encoder_outputs_masks is None:
            batch_size, seq_len, _ = encoder_outputs.shape
            encoder_outputs_masks = torch.ones(batch_size, seq_len, device=encoder_outputs.device)
        mask_value = -1e4 if similarity.dtype == torch.float16 else -1e9
        masks = ((1. - encoder_outputs_masks.unsqueeze(1).expand_as(similarity)) * mask_value).to(dtype=similarity.dtype)
        similarity = similarity + masks
        similarity = torch.softmax(similarity, dim=-1)
        attention_output = torch.bmm(similarity, encoder_outputs)
        return attention_output

    def forward_step(
        self,
        decoder_hidden_state: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: torch.LongTensor,
        vision_hidden_states: torch.LongTensor,
        goal_hidden_state: torch.FloatTensor = None,
        last_guiding_hidden_state: Optional[torch.FloatTensor] = None,
        last_guiding_context: Optional[torch.FloatTensor] = None,
    ):
        attention_outputs = self.attention(
            tensor=last_guiding_hidden_state.unsqueeze(1),
            encoder_outputs=encoder_hidden_states,
            encoder_outputs_masks=encoder_attention_mask,
        ).squeeze(1)

        goal_attention_outputs = self.attention(
            tensor=goal_hidden_state.unsqueeze(1),
            encoder_outputs=vision_hidden_states,
            encoder_outputs_masks=vision_hidden_states.new_ones(*vision_hidden_states.shape[:2], dtype=torch.long),
        ).squeeze(1)

        inference_reasons = torch.cat([attention_outputs, goal_attention_outputs, decoder_hidden_state], dim=-1)
        guiding_hidden_state, guiding_context = self.inference_rnn(
            inference_reasons,
            (last_guiding_hidden_state, last_guiding_context)
        )
        return guiding_hidden_state, guiding_context
