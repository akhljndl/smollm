"""Architecture registry — every named config used by the released checkpoints."""

_MODERN = {"swiglu": True, "rope": True}
_WS = {**_MODERN, "weight_sharing": True}
_VANILLA = {"swiglu": False, "rope": False, "weight_sharing": False}
_GRU = {"model_type": "gru"}


ARCHITECTURES: dict[str, dict] = {
    # Vanilla GPT scaling curve
    "L2H2E64": {"n_layer": 2, "n_head": 2, "n_embd": 64, **_VANILLA},
    "L2H2E128": {"n_layer": 2, "n_head": 2, "n_embd": 128, **_VANILLA},
    "L2H2E144": {"n_layer": 2, "n_head": 2, "n_embd": 144, **_VANILLA},  # GPT-527K
    "L4H4E128": {"n_layer": 4, "n_head": 4, "n_embd": 128, **_VANILLA},
    "gpt-3.2m": {"n_layer": 4, "n_head": 4, "n_embd": 256, **_VANILLA},
    "L6H6E384": {"n_layer": 6, "n_head": 6, "n_embd": 384, **_VANILLA},
    # Weight-shared headline configs
    "ws-53k": {"n_layer": 8, "n_head": 4, "n_embd": 64, **_WS},
    "ws-206k": {"n_layer": 8, "n_head": 4, "n_embd": 128, **_WS},
    # GRU baselines (param-matched)
    "gru-53k": {"n_layers": 2, "hidden_size": 64, **_GRU},
    "gru-206k": {"n_layers": 3, "hidden_size": 105, **_GRU},
}
