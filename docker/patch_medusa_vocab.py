# vLLM 0.30: the Medusa vocab alignment reads target hf_config.vocab_size, which multimodal
# configs (Qwen3.5/3.8, Gemma 4) only expose under text_config. Use the text config.
import os
import vllm

p = os.path.join(os.path.dirname(vllm.__file__), "config/speculative.py")
s = open(p).read()
old = "target_vocab = self.target_model_config.hf_config.vocab_size"
assert s.count(old) == 1, "speculative.py medusa block changed; update patch"
open(p, "w").write(s.replace(old, "target_vocab = self.target_model_config.hf_text_config.vocab_size", 1))
