# microscale/autopsy.py
"""Model autopsy utilities — analyze safetensors models without loading weights.

Read the safetensors JSON header (tiny, ~35KB) to extract tensor names,
shapes, dtypes, and compute parameter breakdowns — all without loading
a single weight into memory.
"""

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

DTYPE_SIZES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


@dataclass
class TensorInfo:
    name: str
    shape: list[int]
    dtype: str

    @property
    def numel(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def size_bytes(self) -> int:
        return self.numel * DTYPE_SIZES.get(self.dtype, 2)


@dataclass
class ModelAnatomy:
    """Parsed architecture information from a safetensors model."""

    repo: str
    tensors: list[TensorInfo] = field(default_factory=list)

    # Detected architecture
    num_layers: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0
    num_q_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    has_tied_embeddings: bool = False
    has_qk_norm: bool = False
    has_fused_qkv: bool = False
    ffn_type: str = ""  # "swiglu_separate", "swiglu_fused", "standard"

    # Parameter breakdown
    total_params: int = 0
    embedding_params: int = 0
    attention_params: int = 0
    ffn_params: int = 0
    norm_params: int = 0
    other_params: int = 0


def parse_safetensors_header(filepath: str | Path) -> list[TensorInfo]:
    """Read safetensors header from a local file without loading tensors."""
    with open(filepath, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))

    tensors = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        tensors.append(
            TensorInfo(
                name=name,
                shape=info["shape"],
                dtype=info["dtype"],
            )
        )
    return tensors


def parse_safetensors_header_remote(repo_id: str) -> list[TensorInfo]:
    """Read safetensors header from HuggingFace Hub via HTTP Range requests.

    Downloads only the header (~35KB), not the full model weights.
    Works for both single-file and sharded models.
    """
    import requests
    from huggingface_hub import get_token, list_repo_tree

    token = get_token()
    auth = {"Authorization": f"Bearer {token}"} if token else {}

    # Find safetensors files
    files = [f.rfilename for f in list_repo_tree(repo_id) if f.rfilename.endswith(".safetensors")]
    if not files:
        raise ValueError(f"No safetensors files found in {repo_id}")

    all_tensors = []
    for filename in files:
        from huggingface_hub import hf_hub_url

        url = hf_hub_url(repo_id, filename)

        # Read 8-byte length prefix
        resp = requests.get(url, headers={**auth, "Range": "bytes=0-7"}, timeout=30)
        resp.raise_for_status()
        header_size = struct.unpack("<Q", resp.content)[0]

        # Read JSON header
        resp = requests.get(
            url,
            headers={**auth, "Range": f"bytes=8-{8 + header_size - 1}"},
            timeout=30,
        )
        resp.raise_for_status()
        header = json.loads(resp.content)

        for name, info in header.items():
            if name == "__metadata__":
                continue
            all_tensors.append(
                TensorInfo(
                    name=name,
                    shape=info["shape"],
                    dtype=info["dtype"],
                )
            )

    return all_tensors


def analyze_architecture(
    tensors: list[TensorInfo],
    repo: str = "",
) -> ModelAnatomy:
    """Detect architecture details from tensor names and shapes."""
    anatomy = ModelAnatomy(repo=repo, tensors=tensors)

    # Total parameters (raw count, may include tied duplicates — corrected below)
    anatomy.total_params = sum(t.numel for t in tensors)

    # Detect number of layers
    layer_indices = set()
    for t in tensors:
        match = re.search(r"layers\.(\d+)\.", t.name)
        if match:
            layer_indices.add(int(match.group(1)))
    anatomy.num_layers = len(layer_indices)

    # Detect vocab and hidden size from embedding
    for t in tensors:
        if "embed_tokens" in t.name and len(t.shape) == 2:
            anatomy.vocab_size = t.shape[0]
            anatomy.hidden_size = t.shape[1]
            break

    # Detect tied embeddings: if lm_head.weight is absent, embeddings are tied
    has_lm_head = any("lm_head" in t.name for t in tensors)
    # Also check config — some models (Qwen3) store lm_head even when tied
    # We detect by checking if lm_head shape matches embed_tokens shape
    if has_lm_head:
        lm_head_shape = None
        embed_shape = None
        for t in tensors:
            if "lm_head" in t.name:
                lm_head_shape = t.shape
            if "embed_tokens" in t.name:
                embed_shape = t.shape
        anatomy.has_tied_embeddings = lm_head_shape == embed_shape
    else:
        anatomy.has_tied_embeddings = True

    # Detect attention config from layer 0
    # Build a lookup of layer 0 tensors for easier access
    layer0 = {t.name: t for t in tensors if "layers.0." in t.name}

    # Check for QK-Norm (gives us head_dim directly)
    for name in layer0:
        if "q_norm" in name:
            anatomy.has_qk_norm = True
            anatomy.head_dim = layer0[name].shape[0]
            break

    # Separate Q/K/V projections
    q_proj_key = next((n for n in layer0 if "q_proj.weight" in n and "qkv" not in n), None)
    k_proj_key = next((n for n in layer0 if "k_proj.weight" in n and "qkv" not in n), None)

    if q_proj_key:
        q_out = layer0[q_proj_key].shape[0]
        if anatomy.head_dim == 0:
            for hd in [128, 96, 80, 64]:
                if q_out % hd == 0:
                    anatomy.head_dim = hd
                    break
        if anatomy.head_dim > 0:
            anatomy.num_q_heads = q_out // anatomy.head_dim

    if k_proj_key and anatomy.head_dim > 0:
        anatomy.num_kv_heads = layer0[k_proj_key].shape[0] // anatomy.head_dim

    # Fused QKV projection (e.g. Phi-4)
    qkv_key = next((n for n in layer0 if "qkv_proj.weight" in n), None)
    if qkv_key:
        anatomy.has_fused_qkv = True
        qkv_out = layer0[qkv_key].shape[0]
        if anatomy.head_dim == 0:
            anatomy.head_dim = 128
        # qkv_out = (num_q_heads + 2 * num_kv_heads) * head_dim
        # We know hidden_size, and for most models: num_q_heads * head_dim >= hidden_size
        # Try common GQA ratios: 3:1, 4:1, 2:1, 1:1
        for kv_h in [8, 4, 16, 32]:
            q_h = (qkv_out // anatomy.head_dim) - 2 * kv_h
            if q_h > 0 and q_h * anatomy.head_dim >= anatomy.hidden_size:
                anatomy.num_q_heads = q_h
                anatomy.num_kv_heads = kv_h
                break

    # FFN type
    for name in layer0:
        if "gate_proj.weight" in name and "gate_up" not in name:
            anatomy.ffn_type = "swiglu_separate"
            anatomy.intermediate_size = layer0[name].shape[0]
        elif "gate_up_proj.weight" in name:
            anatomy.ffn_type = "swiglu_fused"
            anatomy.intermediate_size = layer0[name].shape[0] // 2
        elif "fc1.weight" in name:
            anatomy.ffn_type = "standard"
            anatomy.intermediate_size = layer0[name].shape[0]

    # Classify parameters by component
    for t in tensors:
        n = t.numel
        name = t.name

        if "embed_tokens" in name:
            anatomy.embedding_params += n
        elif "lm_head" in name:
            # Don't double-count if tied
            if not anatomy.has_tied_embeddings:
                anatomy.embedding_params += n
        elif any(
            k in name
            for k in [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "qkv_proj",
                "q_norm",
                "k_norm",
            ]
        ):
            anatomy.attention_params += n
        elif any(
            k in name
            for k in [
                "gate_proj",
                "up_proj",
                "down_proj",
                "gate_up_proj",
                "fc1",
                "fc2",
            ]
        ):
            anatomy.ffn_params += n
        elif "norm" in name or "layernorm" in name:
            anatomy.norm_params += n
        else:
            anatomy.other_params += n

    # Correct total for tied embeddings: subtract the duplicate lm_head
    if anatomy.has_tied_embeddings:
        lm_head_params = sum(t.numel for t in tensors if "lm_head" in t.name)
        anatomy.total_params -= lm_head_params

    return anatomy


def format_params(n: int) -> str:
    """Format parameter count as human-readable string."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)
