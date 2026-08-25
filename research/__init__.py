"""Offline Strategy Discovery Analyzer.

Replays raw collector data (see `poly_analyzer.discovery`) through a grid of
signal / execution / exit-policy configurations to discover which
combinations give a robust, executable Polymarket edge. This package never
opens a socket and never modifies `poly_analyzer/` -- it wraps and reuses it.
"""
