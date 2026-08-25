"""Ablation config generation (IMPLEMENTATION CONTRACT #35). Disabled
filters are always None -- never a sentinel threshold like -1 -- exactly
like every other SignalDiscoveryConfig in this pipeline."""
from __future__ import annotations

from dataclasses import replace

from research.discovery_types import SignalDiscoveryConfig, deterministic_id

# Each entry: (step_name, fields_to_copy_from_base_when_enabled)
FEATURE_STEPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("momentum", ()),  # always on -- the baseline every step includes
    ("flow", ("flow_window_ms", "flow_threshold")),
    ("imbalance", ("imbalance_depth", "imbalance_threshold")),
    ("volume", ("volume_window_ms", "volume_z_threshold")),
    ("poly_lag", ("poly_lag_window_ms", "max_poly_reprice")),
)


def generate_ablation_sequence(
    base: SignalDiscoveryConfig, steps: tuple[str, ...] = tuple(name for name, _ in FEATURE_STEPS),
) -> list[SignalDiscoveryConfig]:
    """Returns one config per prefix of `steps`: momentum-only, momentum+flow,
    momentum+flow+imbalance, ... -- always momentum plus whichever features
    have been "turned on" so far, with every not-yet-enabled feature's
    fields explicitly None (disabled), not the base's value."""
    step_fields = dict(FEATURE_STEPS)
    all_ablatable_fields = {field for fields in step_fields.values() for field in fields}

    out: list[SignalDiscoveryConfig] = []
    enabled_fields: set[str] = set()
    for step in steps:
        enabled_fields |= set(step_fields[step])
        overrides = {
            field: (getattr(base, field) if field in enabled_fields else None)
            for field in all_ablatable_fields
        }
        cfg = replace(base, id="", **overrides)
        cfg = replace(cfg, id=deterministic_id("ablcfg", cfg))
        out.append(cfg)
    return out
