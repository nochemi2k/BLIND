import torch
import typing

from .ea_transformer import EADecoderLayer, EATransformer, split_ea
from .hetero_transformer import split_hetero
from .transformer import EncoderLayer, FourierEmbedding, MultiHeadAttention

MS_PREFIX = "MS/"
MS_PAD = -1000.0
MZ_SCALE = 100.0  # m/z enters the Fourier features in units of 100 Da
MS_PEAK, MS_PRECURSOR = 0, 1  # token types

def split_ms(
    src_c_nmr : typing.Dict[str, torch.Tensor]
) -> typing.Tuple[typing.Dict[str, torch.Tensor], typing.Optional[typing.Dict[str, torch.Tensor]]]:
    # The MS2 spectrum travels inside the 13C dict as "MS/<field>" keys
    carbon = {k: v for k, v in src_c_nmr.items() if not k.startswith(MS_PREFIX)}
    ms = {k[len(MS_PREFIX):]: v for k, v in src_c_nmr.items() if k.startswith(MS_PREFIX)}
    return carbon, ms or None

class MSEncoder(torch.nn.Module):
    def __init__(
        self,
        mz_d_ff : int,
        mz_hidden_dim : int,
        intensity_hidden_dim : int,
        num_heads : int,
        num_layers : int,
        d_ff : int,
        d_model : int,
        dropout : float
    ) -> None:
        super(MSEncoder, self).__init__()

        # Tokens: the precursor ion (its m/z, when reported) and the peaks (m/z, intensity); no adduct or
        # instrument metadata. No positional encoding: peaks are sorted by m/z, which the embedding carries.
        hidden_dim = mz_hidden_dim + intensity_hidden_dim
        self.mz_emb = FourierEmbedding(d_ff=mz_d_ff, output_dim=mz_hidden_dim)
        self.intensity_emb = FourierEmbedding(d_ff=mz_d_ff, output_dim=intensity_hidden_dim)
        self.type_emb = torch.nn.Embedding(2, hidden_dim)
        self.layers = torch.nn.ModuleList([EncoderLayer(hidden_dim, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.proj = torch.nn.Linear(hidden_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        data : typing.Dict[str, torch.Tensor],
        mask : torch.Tensor
    ) -> torch.Tensor:
        mz = data['mz'].clamp(min=0) / MZ_SCALE
        x = torch.cat([self.mz_emb(mz), self.intensity_emb(data['intensity'].sqrt())], dim=-1)
        x = self.dropout(x + self.type_emb(data['token_type']))
        for layer in self.layers:
            x = layer(x, mask)
        return self.proj(x)

class MSDecoderLayer(EADecoderLayer):
    def __init__(self, d_model, num_heads, d_ff, dropout, hetero_types):
        super(MSDecoderLayer, self).__init__(d_model, num_heads, d_ff, dropout, hetero_types)
        self.ms_cross_attn = MultiHeadAttention(d_model, num_heads)

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
        src_ea_mask : typing.Optional[torch.Tensor] = None,
        enc_ms_output : typing.Optional[torch.Tensor] = None,
        src_ms_mask : typing.Optional[torch.Tensor] = None
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

        for cross_attn, enc_output, src_mask in (
            (self.ea_cross_attn, enc_ea_output, src_ea_mask),
            (self.ms_cross_attn, enc_ms_output, src_ms_mask),
        ):
            if enc_output is not None:
                attn = cross_attn(Q=x, K=enc_output, V=enc_output, mask=src_mask)
                attn = attn.masked_fill(~src_mask.any(dim=-1), 0.)
                residual = residual + self.dropout(attn)

        x = self.norm2(residual)
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class MSTransformer(EATransformer):
    """EATransformer (1H, 13C, heteronuclear spectra, elemental analysis) + one MS2 spectrum."""

    def __init__(
        self,
        ms_mz_hidden_dim : int,
        ms_intensity_hidden_dim : int,
        ms_num_heads : int,
        ms_num_layers : int,
        ms_d_ff : int,
        **ea_kwargs : typing.Any
    ) -> None:
        super(MSTransformer, self).__init__(**ea_kwargs)

        d_model = ea_kwargs['d_model']
        num_heads = ea_kwargs['num_heads']
        d_ff = ea_kwargs['d_ff']
        dropout = ea_kwargs['dropout']

        self.ms_encoder = MSEncoder(
            mz_d_ff=ea_kwargs['spectrum_d_ff'],
            mz_hidden_dim=ms_mz_hidden_dim,
            intensity_hidden_dim=ms_intensity_hidden_dim,
            num_heads=ms_num_heads,
            num_layers=ms_num_layers,
            d_ff=ms_d_ff,
            d_model=d_model,
            dropout=dropout,
        )

        self.decoder_layers = torch.nn.ModuleList(
            [MSDecoderLayer(d_model, num_heads, d_ff, dropout, self.hetero_types) for _ in range(ea_kwargs['num_layers'])]
        )

    def init_from_ea(
        self,
        ea_state_dict : typing.Dict[str, torch.Tensor]
    ) -> None:
        # All weights of the model with elemental analysis are loaded; the MS2 encoder and cross-attentions
        # stay random, with a zero output projection so the start equals that model
        missing, unexpected = self.load_state_dict(ea_state_dict, strict=False)
        if unexpected:
            raise ValueError(f"Checkpoint has unknown keys: {unexpected[:5]}")
        not_new = [k for k in missing if not (k.startswith("ms_encoder.") or ".ms_cross_attn." in k)]
        if not_new:
            raise ValueError(f"Checkpoint is missing keys (not a model with elemental analysis?): {not_new[:5]}")

        with torch.no_grad():
            for dec_layer in self.decoder_layers:
                dec_layer.ms_cross_attn.W_o.weight.zero_()
                dec_layer.ms_cross_attn.W_o.bias.zero_()

    def forward(
        self,
        src_c_nmr : typing.Dict[str, torch.Tensor],
        src_h_nmr : typing.Dict[str, torch.Tensor],
        tgt : torch.Tensor
    ):
        # "MS/..." and "EA/..." keys first: split_hetero would read them as spectrum types
        src_c_nmr, src_ms = split_ms(src_c_nmr)
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

        enc_ms_output, src_ms_mask = None, None
        if src_ms is not None:
            src_ms_mask = (src_ms['mz'] != MS_PAD).unsqueeze(1).unsqueeze(2)
            if src_ms_mask.any():
                enc_ms_output = self.ms_encoder(src_ms, src_ms_mask)

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
                src_ea_mask=src_ea_mask,
                enc_ms_output=enc_ms_output,
                src_ms_mask=src_ms_mask
            )

        output = self.fc(dec_output)
        return output
