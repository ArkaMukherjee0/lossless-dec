# vLLM 0.30: EagleMistralLarge3Model calls nn.Module.__init__ instead of DeepseekV2Model.__init__,
# so the inherited load_weights hits a missing `is_fused_shared_expert_enabled`. Compute it the
# same way DeepseekV2Model.__init__ does when absent.
import os
import vllm

p = os.path.join(os.path.dirname(vllm.__file__), "model_executor/models/deepseek_v2.py")
s = open(p).read()
anchor = "        # Params for weights, fp8 weight scales, fp8 activation scales\n        # (param_name, weight_name, expert_id, shard_id)\n        expert_params_mapping = fused_moe_make_expert_params_mapping("
assert s.count(anchor) == 1, "deepseek_v2.load_weights changed; update patch"
fix = ("        if not hasattr(self, \"is_fused_shared_expert_enabled\"):  # lossless-dec patch (EAGLE Mistral)\n"
       "            self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(\n"
       "                self.layers, DeepseekV2MoE, \"mlp\")\n")
open(p, "w").write(s.replace(anchor, fix + anchor, 1))
