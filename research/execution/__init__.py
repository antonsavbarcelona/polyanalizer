"""Exit-policy engine: builds a signal's future book-path once per
(signal, execution_config), then evaluates arbitrary exit configs against
that cached path with zero further DB queries (task §22/§70)."""
