import torch
import typing

from .transformer import (
    CouplingEmbedding,
    DecoderLayer,
    EncoderLayer,
    FourierEmbedding,
    MultiHeadAttention,
    PositionalEncoding,
    Transformer,
)

HETERO_TYPES = ("F_NMR", "P_NMR", "B_NMR", "Si_NMR")
SPECTRUM_PAD = -1000.0

def split_hetero(
    src_c_nmr : typing.Dict[str, torch.Tensor]
) -> typing.Tuple[typing.Dict[str, torch.Tensor], typing.Dict[str, typing.Dict[str, torch.Tensor]]]:
    # Heteronuclear inputs travel inside the 13C dict as "<type>/<field>" keys
    carbon = {k: v for k, v in src_c_nmr.items() if "/" not in k}
    hetero : typing.Dict[str, typing.Dict[str, torch.Tensor]] = {}
    for key, value in src_c_nmr.items():
        if "/" in key:
            spectrum_type, field = key.split("/", 1)
            hetero.setdefault(spectrum_type, {})[field] = value
    return carbon, hetero

class HeteroEncoder(torch.nn.Module):
    def __init__(
        self,
        multiplicity_vocab_size : int,
        spectrum_d_ff : int,
        spectrum_hidden_dim : int,
        multiplicity_hidden_dim : int,
        max_couplings : int,
        coupling_d_ff : int,
        coupling_slot_dim : int,
        hidden_dim : int,
        num_heads : int,
        num_layers : int,
        d_ff : int,
        d_model : int,
        max_seq_length : int,
        dropout : float
    ) -> None:
        super(HeteroEncoder, self).__init__()

        # These nuclei have no integrals, so a peak is shift + multiplicity + J
        self.spectrum_emb = FourierEmbedding(d_ff=spectrum_d_ff, output_dim=spectrum_hidden_dim)
        self.multiplicity_emb = torch.nn.Embedding(multiplicity_vocab_size, multiplicity_hidden_dim)
        self.coupling_emb = CouplingEmbedding(
            max_couplings=max_couplings,
            d_ff=coupling_d_ff,
            slot_dim=coupling_slot_dim
        )
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_length)
        self.layers = torch.nn.ModuleList([EncoderLayer(hidden_dim, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.proj = torch.nn.Linear(hidden_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        data : typing.Dict[str, torch.Tensor],
        mask : torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat(
            [
                self.spectrum_emb(data['spectrum']),
                self.multiplicity_emb(data['multiplicity']),
                self.coupling_emb(data['coupling']),
            ],
            dim=-1
        )
        x = self.dropout(self.positional_encoding(x))
        for layer in self.layers:
            x = layer(x, mask)
        return self.proj(x)

class HeteroDecoderLayer(DecoderLayer):
    def __init__(self, d_model, num_heads, d_ff, dropout, hetero_types):
        super(HeteroDecoderLayer, self).__init__(d_model, num_heads, d_ff, dropout)
        self.hetero_cross_attn = torch.nn.ModuleDict(
            {spectrum_type: MultiHeadAttention(d_model, num_heads) for spectrum_type in hetero_types}
        )

    def forward(
        self,
        x : torch.Tensor,
        enc_carbon_output : torch.Tensor,
        enc_hydrogen_output : torch.Tensor,
        src_carbon_mask : torch.Tensor,
        src_hydrogen_mask : torch.Tensor,
        tgt_mask : torch.Tensor,
        enc_hetero_outputs : typing.Dict[str, torch.Tensor],
        src_hetero_masks : typing.Dict[str, torch.Tensor]
    ) -> torch.Tensor:

        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        carbon_attn_output = self.carbon_cross_attn(
            Q=x,
            K=enc_carbon_output,
            V=enc_carbon_output,
            mask=src_carbon_mask
        )

        hydrogen_attn_output = self.hydrogen_cross_attn(
            Q=x,
            K=enc_hydrogen_output,
            V=enc_hydrogen_output,
            mask=src_hydrogen_mask
        )

        carbon_attn_output = carbon_attn_output.masked_fill(~src_carbon_mask.any(dim=-1), 0.)
        hydrogen_attn_output = hydrogen_attn_output.masked_fill(~src_hydrogen_mask.any(dim=-1), 0.)

        residual = x + self.dropout(carbon_attn_output) + self.dropout(hydrogen_attn_output)

        for spectrum_type, enc_hetero_output in enc_hetero_outputs.items():
            src_hetero_mask = src_hetero_masks[spectrum_type]
            hetero_attn_output = self.hetero_cross_attn[spectrum_type](
                Q=x,
                K=enc_hetero_output,
                V=enc_hetero_output,
                mask=src_hetero_mask
            )
            hetero_attn_output = hetero_attn_output.masked_fill(~src_hetero_mask.any(dim=-1), 0.)
            residual = residual + self.dropout(hetero_attn_output)

        x = self.norm2(residual)
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class HeteroTransformer(Transformer):
    def __init__(
        self,
        hetero_types : typing.Sequence[str],
        hetero_spectrum_hidden_dim : int,
        hetero_multiplicity_hidden_dim : int,
        hetero_coupling_slot_dim : int,
        hetero_hidden_dim : int,
        hetero_num_heads : int,
        hetero_num_layers : int,
        hetero_d_ff : int,
        **base_kwargs : typing.Any
    ) -> None:
        super(HeteroTransformer, self).__init__(**base_kwargs)

        self.hetero_types = tuple(hetero_types)
        d_model = base_kwargs['d_model']
        num_heads = base_kwargs['num_heads']
        d_ff = base_kwargs['d_ff']
        dropout = base_kwargs['dropout']

        self.hetero_encoders : torch.nn.ModuleDict[str, torch.nn.Module] = torch.nn.ModuleDict(
            {
                spectrum_type : HeteroEncoder(
                    multiplicity_vocab_size=base_kwargs['multiplicity_vocab_size'],
                    spectrum_d_ff=base_kwargs['spectrum_d_ff'],
                    spectrum_hidden_dim=hetero_spectrum_hidden_dim,
                    multiplicity_hidden_dim=hetero_multiplicity_hidden_dim,
                    max_couplings=base_kwargs['max_couplings'],
                    coupling_d_ff=base_kwargs['coupling_d_ff'],
                    coupling_slot_dim=hetero_coupling_slot_dim,
                    hidden_dim=hetero_hidden_dim,
                    num_heads=hetero_num_heads,
                    num_layers=hetero_num_layers,
                    d_ff=hetero_d_ff,
                    d_model=d_model,
                    max_seq_length=base_kwargs['max_seq_length'],
                    dropout=dropout,
                )
                for spectrum_type in self.hetero_types
            }
        )
        # Encoder output for an empty spectrum: the nucleus is absent from the molecule, unlike a
        # spectrum that was not given at all (no cross-attention then)
        self.hetero_empty = torch.nn.ParameterDict(
            {spectrum_type : torch.nn.Parameter(torch.randn(d_model) * 0.02) for spectrum_type in self.hetero_types}
        )

        self.decoder_layers = torch.nn.ModuleList(
            [HeteroDecoderLayer(d_model, num_heads, d_ff, dropout, self.hetero_types) for _ in range(base_kwargs['num_layers'])]
        )

    def init_from_base(
        self,
        base_state_dict : typing.Dict[str, torch.Tensor]
    ) -> None:
        # All weights of the 1H/13C model are loaded; heteronuclear encoders and cross-attentions
        # stay random, with a zero output projection so the start equals the base model
        missing, unexpected = self.load_state_dict(base_state_dict, strict=False)
        if unexpected:
            raise ValueError(f"Base checkpoint has unknown keys: {unexpected[:5]}")
        new_prefixes = ("hetero_encoders.", "hetero_empty.")
        not_new = [k for k in missing if not (k.startswith(new_prefixes) or ".hetero_cross_attn." in k)]
        if not_new:
            raise ValueError(f"Base checkpoint is missing keys: {not_new[:5]}")

        with torch.no_grad():
            for dec_layer in self.decoder_layers:
                for hetero_cross_attn in dec_layer.hetero_cross_attn.values():
                    hetero_cross_attn.W_o.weight.zero_()
                    hetero_cross_attn.W_o.bias.zero_()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Checkpoints saved before the empty-spectrum embeddings keep their initial values
        state_dict = dict(state_dict)
        for key, value in self.state_dict().items():
            if key.startswith("hetero_empty."):
                state_dict.setdefault(key, value)
        return super(HeteroTransformer, self).load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(
        self,
        src_c_nmr : typing.Dict[str, torch.Tensor],
        src_h_nmr : typing.Dict[str, torch.Tensor],
        tgt : torch.Tensor
    ):
        src_c_nmr, src_hetero = split_hetero(src_c_nmr)

        src_carbon_mask, src_hydrogen_mask, tgt_mask = self.generate_mask(
            src_carbon_spectrum=src_c_nmr['spectrum'],
            src_hydrogen_spectrum=src_h_nmr['spectrum'],
            tgt=tgt
        )

        src_c_nmr_embedded = self.dropout(self.positional_encoding(self.spectrum_embedding(src_c_nmr, spectrum_type='C_NMR')))
        src_h_nmr_embedded = self.dropout(self.positional_encoding(self.spectrum_embedding(src_h_nmr, spectrum_type='H_NMR')))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_carbon_output = src_c_nmr_embedded
        for enc_layer in self.encoder_carbon_layers:
            enc_carbon_output = enc_layer(enc_carbon_output, src_carbon_mask)

        enc_hydrogen_output = src_h_nmr_embedded
        for enc_layer in self.encoder_hydrogen_layers:
            enc_hydrogen_output = enc_layer(enc_hydrogen_output, src_hydrogen_mask)

        enc_hetero_outputs : typing.Dict[str, torch.Tensor] = {}
        src_hetero_masks : typing.Dict[str, torch.Tensor] = {}
        for spectrum_type, data in src_hetero.items():
            # 19F shifts are mostly negative, so padding is found by its value, not by sign
            src_hetero_mask = (data['spectrum'] != SPECTRUM_PAD).unsqueeze(1).unsqueeze(2)
            if not src_hetero_mask.any():
                continue
            enc_hetero_output = self.hetero_encoders[spectrum_type](data, src_hetero_mask)
            if 'empty' in data:
                # An empty spectrum is one placeholder peak; its encoding is replaced by the learned vector
                empty = data['empty'].view(-1, 1, 1)
                enc_hetero_output = torch.where(empty, self.hetero_empty[spectrum_type].to(enc_hetero_output.dtype), enc_hetero_output)
            enc_hetero_outputs[spectrum_type] = enc_hetero_output
            src_hetero_masks[spectrum_type] = src_hetero_mask

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(
                x=dec_output,
                enc_carbon_output=enc_carbon_output,
                enc_hydrogen_output=enc_hydrogen_output,
                src_carbon_mask=src_carbon_mask,
                src_hydrogen_mask=src_hydrogen_mask,
                tgt_mask=tgt_mask,
                enc_hetero_outputs=enc_hetero_outputs,
                src_hetero_masks=src_hetero_masks
            )

        output = self.fc(dec_output)
        return output
