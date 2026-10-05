import os
import vllm

p = os.path.join(os.path.dirname(vllm.__file__), "model_executor/models/pixtral.py")
s = open(p).read()
old = """from transformers.models.pixtral.modeling_pixtral import (
    PixtralRotaryEmbedding,
    apply_rotary_pos_emb,
    position_ids_in_meshgrid,
)
"""
new = """try:
    from transformers.models.pixtral.modeling_pixtral import (
        PixtralRotaryEmbedding,
        apply_rotary_pos_emb,
        position_ids_in_meshgrid,
    )
except ImportError:  # lossless-dec patch: only the HF-format vision tower needs these
    from transformers.models.pixtral.modeling_pixtral import apply_rotary_pos_emb

    def _missing(*args, **kwargs):
        raise NotImplementedError("HF-format Pixtral vision tower unsupported with this transformers")

    PixtralRotaryEmbedding = position_ids_in_meshgrid = _missing
"""
assert old in s, "pixtral import block changed; update patch"
open(p, "w").write(s.replace(old, new, 1))
