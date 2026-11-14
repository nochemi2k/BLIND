import torch
import typing

from .hetero_transformer import HeteroDecoderLayer, HeteroTransformer, split_hetero
from .transformer import EncoderLayer, FourierEmbedding, MultiHeadAttention

# Index 0 pads; elements outside the list (metals) share <other>
EA_ELEMENTS = ("<PAD>", "C", "H", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si", "B", "Se", "<other>")
EA_ELEMENT2IDX = {element: idx for idx, element in enumerate(EA_ELEMENTS)}
EA_PREFIX = "EA/"

def split_ea(
    src_c_nmr : typing.Dict[str, torch.Tensor]
) -> typing.Tuple[typing.Dict[str, torch.Tensor], typing.Optional[typing.Dict[str, torch.Tensor]]]:
    # The elemental analysis travels inside the 13C dict as "EA/<field>" keys
    carbon = {k: v for k, v in src_c_nmr.items() if not k.startswith(EA_PREFIX)}
    ea = {k[len(EA_PREFIX):]: v for k, v in src_c_nmr.items() if k.startswith(EA_PREFIX)}
    return carbon, ea or None

class EAEncoder(torch.nn.Module):
    def __init__(
        self,
        value_d_ff : int,
        value_hidden_dim : int,
        element_hidden_dim : int,
        hidden_dim : int,
        num_heads : int,
        num_layers : int,
        d_ff : int,
        d_model : int,
        dropout : float
    ) -> None:
        super(EAEncoder, self).__init__()

        # One token per reported element: element + Found mass % + molar % among the reported elements.
        # Both values are in absolute %, the scale of the measurement error, so no per-element scaling.
        # No positional encoding: the elements are a set.
        self.element_emb = torch.nn.Embedding(len(EA_ELEMENTS), element_hidden_dim, padding_idx=0)
        self.mass_emb = FourierEmbedding(d_ff=value_d_ff, output_dim=value_hidden_dim)
        self.molar_emb = FourierEmbedding(d_ff=value_d_ff, output_dim=value_hidden_dim)
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
                self.element_emb(data['element']),
                self.mass_emb(data['mass']),
                self.molar_emb(data['molar']),
            ],
            dim=-1
        )
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.proj(x)

class EADecoderLayer(HeteroDecoderLayer):
    def __init__(self, d_model, num_heads, d_ff, dropout, hetero_types):
        super(EADecoderLayer, self).__init__(d_model, num_heads, d_ff, dropout, hetero_types)
        self.ea_cross_attn = MultiHeadAttention(d_model, num_heads)

    def forward(
        self,
        x : torch.Tensor,
        enc_carbon_output : torch.Tensor,
        enc_hydrogen_output : torch.Tensor,
        src_carbon_mask : torch.Tensor,
        src_hydrogen_mask : torch.Tensor,
        tgt_mask : torch.Tensor,
        enc_hetero_outputs : typing.Dict[str, torch.Tensor],
        src_hetero_masks : typing.Dict[str, torch.Tensor],
        enc_ea_output : typing.Optional[torch.Tensor] = None,
        src_ea_mask : typing.Optional[torch.Tensor] = None
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

        if enc_ea_output is not None:
            ea_attn_output = self.ea_cross_attn(
                Q=x,
                K=enc_ea_output,
                V=enc_ea_output,
                mask=src_ea_mask
            )
            ea_attn_output = ea_attn_output.masked_fill(~src_ea_mask.any(dim=-1), 0.)
            residual = residual + self.dropout(ea_attn_output)

        x = self.norm2(residual)
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class EATransformer(HeteroTransformer):
    """HeteroTransformer (1H, 13C, heteronuclear spectra) + elemental analysis."""

    def __init__(
        self,
        ea_value_hidden_dim : int,
        ea_element_hidden_dim : int,
        ea_hidden_dim : int,
        ea_num_heads : int,
        ea_num_layers : int,
        ea_d_ff : int,
        **hetero_kwargs : typing.Any
    ) -> None:
        super(EATransformer, self).__init__(**hetero_kwargs)

        d_model = hetero_kwargs['d_model']
        num_heads = hetero_kwargs['num_heads']
        d_ff = hetero_kwargs['d_ff']
        dropout = hetero_kwargs['dropout']

        self.ea_encoder = EAEncoder(
            value_d_ff=hetero_kwargs['spectrum_d_ff'],
            value_hidden_dim=ea_value_hidden_dim,
            element_hidden_dim=ea_element_hidden_dim,
            hidden_dim=ea_hidden_dim,
            num_heads=ea_num_heads,
            num_layers=ea_num_layers,
            d_ff=ea_d_ff,
            d_model=d_model,
            dropout=dropout,
        )

        self.decoder_layers = torch.nn.ModuleList(
            [EADecoderLayer(d_model, num_heads, d_ff, dropout, self.hetero_types) for _ in range(hetero_kwargs['num_layers'])]
        )

    def init_from_hetero(
        self,
        hetero_state_dict : typing.Dict[str, torch.Tensor]
    ) -> None:
        # All weights of the model with heteronuclear spectra are loaded; the elemental analysis encoder
        # and cross-attentions stay random, with a zero output projection so the start equals that model
        missing, unexpected = self.load_state_dict(hetero_state_dict, strict=False)
        if unexpected:
            raise ValueError(f"Checkpoint has unknown keys: {unexpected[:5]}")
        not_new = [k for k in missing if not (k.startswith("ea_encoder.") or ".ea_cross_attn." in k)]
        if not_new:
            raise ValueError(f"Checkpoint is missing keys (not a model with heteronuclear spectra?): {not_new[:5]}")

        with torch.no_grad():
            for dec_layer in self.decoder_layers:
                dec_layer.ea_cross_attn.W_o.weight.zero_()
                dec_layer.ea_cross_attn.W_o.bias.zero_()

    def forward(
        self,
        src_c_nmr : typing.Dict[str, torch.Tensor],
        src_h_nmr : typing.Dict[str, torch.Tensor],
        tgt : torch.Tensor
    ):
        # "EA/..." keys first: split_hetero would read them as a spectrum type
        src_c_nmr, src_ea = split_ea(src_c_nmr)
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

        enc_hetero_outputs, src_hetero_masks = self.encode_hetero(src_hetero)

        enc_ea_output, src_ea_mask = None, None
        if src_ea is not None:
            src_ea_mask = (src_ea['element'] != 0).unsqueeze(1).unsqueeze(2)
            if src_ea_mask.any():
                enc_ea_output = self.ea_encoder(src_ea, src_ea_mask)

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
                src_hetero_masks=src_hetero_masks,
                enc_ea_output=enc_ea_output,
                src_ea_mask=src_ea_mask
            )

        output = self.fc(dec_output)
        return output
