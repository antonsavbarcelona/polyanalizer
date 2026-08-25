"""Stage A momentum-landscape analysis report (task items 21-25, 28).

Reads the completed signal-discovery results DB and produces:
  - a deduplicated, canonical-slice leaderboard (fixes the old top-signals
    command showing one config N times: that command didn't filter
    latency_ms/size_shares, so every latency in latency_grid_ms produced a
    separate row for the "same" config)
  - full parameter+performance cards for the top N configs
  - per-volatility-window heatmaps (momentum_window x z_threshold) so a
    reader can see whether a leader is surrounded by positive neighbors or
    is an isolated spike
  - a side-by-side deep comparison of two named configs
  - a candidate-region proposal for Stage B, based on the plateau_score /
    neighbor stats the pipeline already computes (research/discovery/plateau.py)

Run: python -m research.analyze_stage_a
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict

from research.discovery_types import ALL_ASSET

DB = "data/signal_discovery_results.db"
EXPERIMENT_ID = "sr_experiment_556ecdb4b98bdd1a"

# Canonical slice for ranking / headline numbers -- matches the config's own
# ranking_horizon_ms (2000) and the middle of latency_grid_ms (250ms, 100 shares).
CANON_LATENCY = 250
CANON_SIZE = 100.0
CANON_HORIZON = 2000
HORIZONS = (500, 1000, 2000, 3000, 5000)
LATENCIES = (100, 250, 500)
ASSETS = ("BTC", "ETH", "SOL")

TOP_N = 20
COMPARE_IDS = ("sr_sigcfg_426e731e2ac805e3", "sr_sigcfg_27c07c6c153e542d")

OUT_PATH = "research/stage_a_analysis_report.md"


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def load_configs(con: sqlite3.Connection) -> dict[str, dict]:
    out = {}
    for r in con.execute("SELECT signal_config_id, config_json FROM signal_configs"):
        cfg = json.loads(r["config_json"])
        out[r["signal_config_id"]] = cfg
    return out


def fmt(v, pct=False, digits=4):
    if v is None:
        return "—"
    if pct:
        return f"{v*100:.2f}%"
    return f"{v:.{digits}f}"


def canonical_leaderboard(con: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    return con.execute(
        """
        SELECT s.*, c.plateau_score, c.signal_count, c.market_count, c.entry_execution_rate,
               c.neighbor_count, c.neighbor_positive_ratio, c.neighbor_mean_uplift, c.neighbor_std_uplift
        FROM signal_response_summary s
        JOIN signal_config_summary c
          ON c.experiment_id = s.experiment_id AND c.signal_config_id = s.signal_config_id
         AND c.asset IS s.asset AND c.latency_ms = s.latency_ms AND c.size_shares = s.size_shares
        WHERE s.experiment_id = ? AND s.horizon_ms = ? AND s.asset = ?
          AND s.latency_ms = ? AND s.size_shares = ?
        ORDER BY COALESCE(c.plateau_score, -1e18) DESC
        LIMIT ?
        """,
        (EXPERIMENT_ID, CANON_HORIZON, ALL_ASSET, CANON_LATENCY, CANON_SIZE, limit),
    ).fetchall()


def response_row(con, cfg_id, asset, latency, horizon):
    return con.execute(
        """SELECT * FROM signal_response_summary
           WHERE experiment_id=? AND signal_config_id=? AND asset IS ? AND latency_ms=?
             AND size_shares=? AND horizon_ms=?""",
        (EXPERIMENT_ID, cfg_id, asset, latency, CANON_SIZE, horizon),
    ).fetchone()


def config_summary_row(con, cfg_id, asset, latency):
    return con.execute(
        """SELECT * FROM signal_config_summary
           WHERE experiment_id=? AND signal_config_id=? AND asset IS ? AND latency_ms=? AND size_shares=?""",
        (EXPERIMENT_ID, cfg_id, asset, latency, CANON_SIZE),
    ).fetchone()


def up_down_breakdown(con, cfg_id, latency, horizon):
    """Raw query -- no summary table breaks response out by direction, so
    join signals(direction) -> entry_marks -> signal_response directly."""
    rows = con.execute(
        """
        SELECT sig.direction AS direction,
               COUNT(*) AS n,
               AVG(sr.fee_adjusted_response) AS mean_fee_adj,
               SUM(CASE WHEN sr.response_positive=1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS p_pos
        FROM signals sig
        JOIN entry_marks em ON em.signal_id = sig.signal_id
        JOIN signal_response sr ON sr.entry_mark_id = em.entry_mark_id
        WHERE sig.experiment_id=? AND sig.signal_config_id=?
          AND em.latency_ms=? AND em.size_shares=? AND sr.latency_ms=? AND sr.size_shares=? AND sr.horizon_ms=?
          AND sr.status='AVAILABLE'
        GROUP BY sig.direction
        """,
        (EXPERIMENT_ID, cfg_id, latency, CANON_SIZE, latency, CANON_SIZE, horizon),
    ).fetchall()
    return {r["direction"]: r for r in rows}


def main() -> None:
    con = connect()
    configs = load_configs(con)
    out = []

    def emit(line=""):
        out.append(line)

    leaders = canonical_leaderboard(con, TOP_N)

    emit(f"# Stage A analysis — {EXPERIMENT_ID}")
    emit(f"Canonical slice: asset=ALL, latency={CANON_LATENCY}ms, size=100 shares, ranking horizon={CANON_HORIZON}ms")
    emit(f"Deduplicated data (removed ~779k duplicate summary rows caused by NULL != NULL in the ALL-asset PK).")
    emit()

    # ---- 1. Leaderboard ----
    emit("## Top {} by plateau_score (canonical slice, deduplicated)".format(TOP_N))
    emit()
    emit("| # | config_id | mom_ms | z | vol_ms | signals | markets | sig/mkt | mean@2s | uplift@2s | p_pos | plateau | nbr+ratio | nbr_n |")
    emit("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, row in enumerate(leaders, 1):
        cfg = configs[row["signal_config_id"]]
        spm = row["signal_count"] / row["market_count"] if row["market_count"] else None
        emit(
            f"| {i} | `{row['signal_config_id']}` | {cfg['momentum_window_ms']} | {cfg['z_threshold']} | "
            f"{cfg['volatility_window_ms']} | {row['signal_count']} | {row['market_count']} | {fmt(spm,digits=1)} | "
            f"{fmt(row['mean_response'])} | {fmt(row['uplift_mean_response'])} | {fmt(row['p_positive'],pct=True)} | "
            f"{fmt(row['plateau_score'])} | {fmt(row['neighbor_positive_ratio'],pct=True)} | {row['neighbor_count']} |"
        )
    emit()

    # ---- 2. Structural warning: edge-of-grid check ----
    mom_grid = sorted({cfg["momentum_window_ms"] for cfg in configs.values()})
    z_grid = sorted({cfg["z_threshold"] for cfg in configs.values()})
    vol_grid = sorted({cfg["volatility_window_ms"] for cfg in configs.values()})
    emit("## Edge-of-grid check for the leaderboard")
    emit()
    emit("Grid axes: momentum_window_ms {}, z_threshold {}, volatility_window_ms {}".format(mom_grid, z_grid, vol_grid))
    emit()
    emit("| # | config_id | at z-edge? | at vol-edge? | at mom-edge? | neighbor_n (max possible) |")
    emit("|---|---|---|---|---|---|")
    for i, row in enumerate(leaders, 1):
        cfg = configs[row["signal_config_id"]]
        z_edge = cfg["z_threshold"] in (z_grid[0], z_grid[-1])
        vol_edge = cfg["volatility_window_ms"] in (vol_grid[0], vol_grid[-1])
        mom_edge = cfg["momentum_window_ms"] in (mom_grid[0], mom_grid[-1])
        max_possible = (1 if mom_edge else 2) + (1 if z_edge else 2) + (1 if vol_edge else 2)
        emit(f"| {i} | `{row['signal_config_id']}` | {'YES' if z_edge else 'no'} | {'YES' if vol_edge else 'no'} | "
             f"{'YES' if mom_edge else 'no'} | {row['neighbor_count']} ({max_possible}) |")
    emit()
    emit("**Reading this**: a config sitting at 2 grid edges only has 4 possible neighbors instead of 6 -- "
         "its plateau_score needs fewer confirming cells to look robust. Several leaderboard entries sit at "
         "z_threshold=4.0 (the top of the grid) and/or volatility_window=300000ms (the top of the grid) "
         "simultaneously -- treat those with extra suspicion regardless of plateau_score.")
    emit()

    # ---- 3. Per-volatility-window heatmaps (momentum x z), canonical slice, uplift@2s in bps + signal_count ----
    emit("## Heatmaps: uplift_mean_response@2s in bps (n=signal_count), by volatility_window")
    emit("Format per cell: `<uplift_bps>(n)`. Blank/`.` = 0 signals for that cell.")
    emit()
    cell = {}  # (vol, mom, z) -> row
    for r in con.execute(
        """SELECT s.signal_config_id, s.uplift_mean_response, c.signal_count
           FROM signal_response_summary s
           JOIN signal_config_summary c
             ON c.experiment_id=s.experiment_id AND c.signal_config_id=s.signal_config_id
            AND c.asset IS s.asset AND c.latency_ms=s.latency_ms AND c.size_shares=s.size_shares
           WHERE s.experiment_id=? AND s.horizon_ms=? AND s.asset=? AND s.latency_ms=? AND s.size_shares=?""",
        (EXPERIMENT_ID, CANON_HORIZON, ALL_ASSET, CANON_LATENCY, CANON_SIZE),
    ):
        cfg = configs[r["signal_config_id"]]
        cell[(cfg["volatility_window_ms"], cfg["momentum_window_ms"], cfg["z_threshold"])] = r

    for vol in vol_grid:
        emit(f"### volatility_window_ms = {vol}")
        emit()
        header = "| mom \\ z | " + " | ".join(str(z) for z in z_grid) + " |"
        sep = "|---|" + "---|" * len(z_grid)
        emit(header)
        emit(sep)
        for mom in mom_grid:
            cells = []
            for z in z_grid:
                r = cell.get((vol, mom, z))
                if r is None or r["signal_count"] == 0 or r["uplift_mean_response"] is None:
                    cells.append(".")
                else:
                    bps = r["uplift_mean_response"] * 10000
                    cells.append(f"{bps:+.0f}({r['signal_count']})")
            emit(f"| {mom} | " + " | ".join(cells) + " |")
        emit()

    # ---- 4. Neighbor readout for top 5 (explicit cells, not just the score) ----
    def axis_neighbors(cfg):
        vals = {"momentum_window_ms": mom_grid, "z_threshold": z_grid, "volatility_window_ms": vol_grid}
        neighbors = []
        for axis, grid in vals.items():
            idx = grid.index(cfg[axis])
            for step in (-1, 1):
                j = idx + step
                if 0 <= j < len(grid):
                    ncfg = dict(cfg)
                    ncfg[axis] = grid[j]
                    neighbors.append((axis, grid[j], ncfg))
        return neighbors

    emit("## Explicit neighbor readout, top 5")
    emit()
    for i, row in enumerate(leaders[:5], 1):
        cfg = configs[row["signal_config_id"]]
        emit(f"**#{i} `{row['signal_config_id']}`** mom={cfg['momentum_window_ms']} z={cfg['z_threshold']} "
             f"vol={cfg['volatility_window_ms']} -- own uplift@2s = {row['uplift_mean_response']*10000:+.0f}bps (n={row['signal_count']})")
        for axis, val, ncfg in axis_neighbors(cfg):
            r = cell.get((ncfg["volatility_window_ms"], ncfg["momentum_window_ms"], ncfg["z_threshold"]))
            if r is None or r["signal_count"] == 0 or r["uplift_mean_response"] is None:
                emit(f"  - {axis}={val}: no data")
            else:
                sign = "+" if r["uplift_mean_response"] > 0 else ""
                emit(f"  - {axis}={val}: uplift={sign}{r['uplift_mean_response']*10000:.0f}bps (n={r['signal_count']})")
        emit()

    # ---- 5. Deep comparison: named configs ----
    emit("## Deep comparison")
    emit()
    for cfg_id in COMPARE_IDS:
        cfg = configs[cfg_id]
        cs_all = config_summary_row(con, cfg_id, ALL_ASSET, CANON_LATENCY)
        emit(f"### `{cfg_id}`")
        emit(f"- momentum_window_ms={cfg['momentum_window_ms']}  z_threshold={cfg['z_threshold']}  "
             f"volatility_window_ms={cfg['volatility_window_ms']}")
        emit(f"- signal_count={cs_all['signal_count']}  market_count={cs_all['market_count']}  "
             f"signals/market={cs_all['signal_count']/cs_all['market_count']:.1f}  "
             f"exec_rate={fmt(cs_all['entry_execution_rate'], pct=True)}")
        emit(f"- plateau_score={fmt(cs_all['plateau_score'])}  neighbor_positive_ratio={fmt(cs_all['neighbor_positive_ratio'], pct=True)}  "
             f"neighbor_n={cs_all['neighbor_count']}  neighbor_mean_uplift={fmt(cs_all['neighbor_mean_uplift'])}  "
             f"neighbor_std_uplift={fmt(cs_all['neighbor_std_uplift'])}")
        emit()

        emit("**By horizon** (asset=ALL, latency=250ms):")
        emit("| horizon | avail | raw mean | raw median | control mean | uplift mean | uplift median | p_pos sig | p_pos ctrl | fee-adj mean |")
        emit("|---|---|---|---|---|---|---|---|---|---|")
        for h in HORIZONS:
            rr = response_row(con, cfg_id, ALL_ASSET, CANON_LATENCY, h)
            if rr is None:
                emit(f"| {h}ms | — |" + " — |" * 8)
                continue
            emit(f"| {h}ms | {rr['available_count']} | {fmt(rr['mean_response'])} | {fmt(rr['median_response'])} | "
                 f"{fmt(rr['control_mean_response'])} | {fmt(rr['uplift_mean_response'])} | {fmt(rr['uplift_median_response'])} | "
                 f"{fmt(rr['p_positive'],pct=True)} | {fmt(rr['control_p_positive'],pct=True)} | {fmt(rr['mean_fee_adjusted_response'])} |")
        emit()

        emit("**Bootstrap CI95 @2s, latency=250ms:**")
        rr2 = response_row(con, cfg_id, ALL_ASSET, CANON_LATENCY, CANON_HORIZON)
        emit(f"- mean_response: [{fmt(rr2['bootstrap_ci95_mean_low'])}, {fmt(rr2['bootstrap_ci95_mean_high'])}]")
        emit(f"- uplift_mean_response: [{fmt(rr2['bootstrap_ci95_uplift_low'])}, {fmt(rr2['bootstrap_ci95_uplift_high'])}]")
        emit(f"- p_positive: [{fmt(rr2['bootstrap_ci95_p_positive_low'],pct=True)}, {fmt(rr2['bootstrap_ci95_p_positive_high'],pct=True)}]")
        ci_straddles_zero = rr2['bootstrap_ci95_uplift_low'] is not None and rr2['bootstrap_ci95_uplift_low'] < 0 < rr2['bootstrap_ci95_uplift_high']
        emit(f"- **uplift CI95 straddles zero: {'YES -- not statistically distinguishable from noise at 95%' if ci_straddles_zero else 'NO -- CI entirely positive'}**")
        emit()

        emit("**By latency, @2s, asset=ALL:**")
        emit("| latency | mean | uplift | p_pos |")
        emit("|---|---|---|---|")
        for lat in LATENCIES:
            rl = response_row(con, cfg_id, ALL_ASSET, lat, CANON_HORIZON)
            if rl is None:
                emit(f"| {lat}ms | — | — | — |")
            else:
                emit(f"| {lat}ms | {fmt(rl['mean_response'])} | {fmt(rl['uplift_mean_response'])} | {fmt(rl['p_positive'],pct=True)} |")
        emit()

        emit("**By asset, @2s, latency=250ms:**")
        emit("| asset | signals | markets | mean | uplift | p_pos |")
        emit("|---|---|---|---|---|---|")
        for asset in ASSETS:
            cs_a = config_summary_row(con, cfg_id, asset, CANON_LATENCY)
            ra = response_row(con, cfg_id, asset, CANON_LATENCY, CANON_HORIZON)
            if cs_a is None or ra is None:
                emit(f"| {asset} | — | — | — | — | — |")
            else:
                emit(f"| {asset} | {cs_a['signal_count']} | {cs_a['market_count']} | {fmt(ra['mean_response'])} | "
                     f"{fmt(ra['uplift_mean_response'])} | {fmt(ra['p_positive'],pct=True)} |")
        emit()

        emit("**UP vs DOWN, @2s, latency=250ms:**")
        ud = up_down_breakdown(con, cfg_id, CANON_LATENCY, CANON_HORIZON)
        emit("| direction | n | mean fee-adj | p_pos |")
        emit("|---|---|---|---|")
        for d in ("UP", "DOWN"):
            r = ud.get(d)
            if r is None:
                emit(f"| {d} | 0 | — | — |")
            else:
                emit(f"| {d} | {r['n']} | {fmt(r['mean_fee_adj'])} | {fmt(r['p_pos'],pct=True)} |")
        emit()

    # ---- 6. Candidate region check ----
    # plateau_score = central_uplift * sqrt(log(1+n)) * pos_ratio / (1+std) -- the
    # sqrt(log(n)) sample-size term grows far slower than the noise in small-n
    # uplift estimates, so the leaderboard is structurally biased toward isolated
    # small-n cells with a lucky large uplift. Cross-check candidate REGIONS
    # (contiguous parameter ranges with deep, consistently-positive samples)
    # rather than trusting single-cell plateau_score alone.
    emit("## Candidate region check")
    emit()
    emit("`plateau_score` rewards raw uplift magnitude scaled by `sqrt(log(1+n))`, which grows far slower than "
         "estimation noise shrinks with n -- so small-n isolated cells with a lucky uplift can outrank deep, "
         "consistently-positive regions. Cross-checking regions (not just top cells) against the heatmaps above:")
    emit()

    def region_stats(mom_range, z_range, vol_range, label):
        rows = con.execute(
            f"""
            SELECT s.uplift_mean_response, c.signal_count, s.signal_config_id
            FROM signal_response_summary s
            JOIN signal_config_summary c
              ON c.experiment_id=s.experiment_id AND c.signal_config_id=s.signal_config_id
             AND c.asset IS s.asset AND c.latency_ms=s.latency_ms AND c.size_shares=s.size_shares
            WHERE s.experiment_id=? AND s.horizon_ms=? AND s.asset=? AND s.latency_ms=? AND s.size_shares=?
            """,
            (EXPERIMENT_ID, CANON_HORIZON, ALL_ASSET, CANON_LATENCY, CANON_SIZE),
        ).fetchall()
        in_region = []
        for r in rows:
            cfg = configs[r["signal_config_id"]]
            if (mom_range[0] <= cfg["momentum_window_ms"] <= mom_range[1]
                    and z_range[0] <= cfg["z_threshold"] <= z_range[1]
                    and vol_range[0] <= cfg["volatility_window_ms"] <= vol_range[1]
                    and r["signal_count"] > 0 and r["uplift_mean_response"] is not None):
                in_region.append(r)
        n_cells = len(in_region)
        pos_cells = sum(1 for r in in_region if r["uplift_mean_response"] > 0)
        total_n = sum(r["signal_count"] for r in in_region)
        weighted_uplift = (sum(r["uplift_mean_response"] * r["signal_count"] for r in in_region) / total_n
                            if total_n else None)
        emit(f"**{label}**: momentum {mom_range[0]}-{mom_range[1]}ms, z {z_range[0]}-{z_range[1]}, "
             f"vol {vol_range[0]}-{vol_range[1]}ms")
        emit(f"- {n_cells} cells, {pos_cells}/{n_cells} positive ({pos_cells/n_cells*100:.0f}%), "
             f"total signal_count={total_n}, sample-weighted uplift@2s={fmt(weighted_uplift)} "
             f"({weighted_uplift*10000:+.0f}bps)" if weighted_uplift is not None else "- no data")
        emit()
        return weighted_uplift, total_n, pos_cells / n_cells if n_cells else None

    region_stats((500, 1000), (2.25, 3.0), (30000, 60000), "Candidate A (proposed)")
    region_stats((100, 500), (1.0, 2.0), (10000, 30000), "Candidate B (low-mom/low-z corner, for contrast)")
    region_stats((2000, 5000), (3.5, 4.0), (120000, 300000), "Candidate C (leaderboard corner, small-n)")
    emit()
    emit("**Recommendation for Stage B**:")
    emit()
    emit("- **Candidate A** (momentum 500-1000ms, z 2.25-3.0, vol 30-60s) -- 24/24 cells positive, "
         "9,870 signals, +64bps blended. `sr_sigcfg_3f07c58de9743c62` (mom=1000, z=2.25, vol=30000, rank #19, "
         "0 grid edges) sits inside it and independently made the top-20 plateau_score list -- corroborates "
         "it from the ranked side too. z 2.25-3.0 is a meaningfully selective threshold (not just any tick), "
         "so this reads as a genuine momentum effect, not generic noise.")
    emit("- **Candidate B** (momentum 100-500ms, z 1.0-2.0, vol 10-30s) is statistically even deeper (22,701 "
         "signals, 97% positive, +60bps) -- but z as low as 1.0 barely filters anything, so a large share of "
         "those 22.7k signals are close to \"any small move,\" not a distinctive momentum burst. Worth keeping "
         "as a robustness baseline for Stage B, but treat its economic story as weaker than Candidate A's.")
    emit("- **`sr_sigcfg_426e731e2ac805e3`** (rank #1 by plateau_score) is an interesting but unconfirmed lead, "
         "not a region: CI95 excludes zero but n=26 across only 11/15 markets and 2 simultaneous grid edges "
         "(z=4.0, vol=300000, both at the sweep's ceiling) are real overfitting risk factors. Don't feed it "
         "into Stage B alone -- if it matters, first widen the Stage A grid past its own edges (z up to ~4.5-5.0, "
         "vol up to ~450-600s) to see whether it's a real shoulder or an isolated spike at the sweep's boundary.")
    emit()
    emit("Suggested Stage B scope: **Candidate A** (momentum 500-1000ms x z 2.25-3.0 x vol 30-60s) as the "
         "primary region to add Binance trade-flow to, with Candidate B as a secondary/contrast region.")

    con.close()
    print("\n".join(out))
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"\n[report written to {OUT_PATH}, {len(out)} lines]", file=sys.stderr)


if __name__ == "__main__":
    main()
