#!/usr/bin/env python3
"""
make_latex_scaling_plot.py
------------------------------------
Generate a compilable LaTeX/pgfplots capacity-scaling figure.

Usage:
  python make_latex_scaling_plot.py
  pdflatex results/size_scaling_figure.tex

Output: results/size_scaling_figure.tex
"""
import os
import json
import numpy as np
import argparse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"
OUTPUT_TEX  = os.path.join(RESULTS_DIR, "size_scaling_figure.tex")

FIXED_N = 10

# Plotting order: baselines first, OVI last (drawn on top).
ALG_CONFIGS = [
    ("alg5", "sm", None),   # SPOIL  – index 0
    ("alg7", None, None),   # DAgger – index 1
    ("alg6", None, None),   # BC     – index 2
    ("alg4", "sm", 50),     # OVI    – index 3 (plotted on top)
]

# SPOIL=blue, DAgger=purple, BC=green, OVI=red
COLORS_HEX     = ["3B7BC8", "7B6EBF", "389C56", "E63946"]
LINE_OPACITIES = [1.0,      1.0,      1.0,      1.0     ]
LINE_WIDTHS    = [1.0,      1.0,      1.0,      1.5     ]   # OVI slightly thicker
# solid=online (OVI, DAgger), dashed=offline (SPOIL, BC)
LINE_STYLES    = ["dashed", "solid",  "dashed", "solid" ]
# filled circle=value-based (OVI, SPOIL), hollow circle=policy-based (DAgger, BC)
MARKERS        = ["*",      "o",      "o",      "*"     ]
MARK_SIZES     = [2.0,      2.0,      2.0,      2.0     ]

BAND_OPACITY = 0.15

EXPERT_WIDTH = 64   # neuron count of the expert network

SIZE_MAPPING = {
    "tiny":          2,
    "veryverysmall": 4,
    "verysmall":     8,
    "small":         16,
    "large":         64,
}
SIZE_ORDER = ["tiny", "veryverysmall", "verysmall", "small", "large"]

EXPERIMENTS = [
    ("Acrobot_v1",         "_sub15"),
    ("CartPole_v1",        "_sub25"),
    ("Pendulum_v1_11bins", "_sub5"),
    ("LunarLander_v2",     "_sub5"),
]

ENV_TITLES = {
    "Acrobot_v1":          "Acrobot-v1",
    "CartPole_v1":         "CartPole-v1",
    "Pendulum_v1_11bins":  "Pendulum-v1",
    "LunarLander_v2":      "LunarLander-v2",
}

# Algorithm names with \textsf (not \texttt); colored with their respective line color
ALG_LABELS = {
    ("alg4", "sm", 50):   r"\shortstack[c]{{\large\textcolor{algcol3}{\textbf{\textsf{OVI (Ours)}}}}\\(online, value-based)}",
    ("alg5", "sm", None): r"\shortstack[c]{{\large\textcolor{algcol0}{\textbf{\textsf{SPOIL}}}}\\(offline, value-based)}",
    ("alg6", None, None): r"\shortstack[c]{{\large\textcolor{algcol2}{\textbf{\textsf{BC}}}}\\(offline, policy-based)}",
    ("alg7", None, None): r"\shortstack[c]{{\large\textcolor{algcol1}{\textbf{\textsf{DAgger}}}}\\(online, policy-based)}",
}

UNIFORM_RETURN = {
    "Acrobot_v1":         -500.0,
    "CartPole_v1":           8.0,
    "Pendulum_v1_11bins": -1650.0,
    "LunarLander_v2":      -600.0,
}

# Expert return line: solid gray, same weight as the old vertical line
EXPERT_LINE_STYLE = "gray, solid, line width=1.5pt"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _make_full_key(alg, mt, L, size):
    if alg == "alg4":               return f"alg4_{mt}_L{L}_{size}"
    elif alg == "alg5":             return f"alg5_{mt}_{size}"
    else:                           return f"{alg}_{size}"

def load_data(env_tag, freq_tag):
    prefix = f"{env_tag}{freq_tag}_"
    results = {}
    expert_path = os.path.join(RESULTS_DIR, f"{prefix}expert.json")
    if os.path.exists(expert_path):
        with open(expert_path) as f:
            results["expert"] = json.load(f)["expert_return"]

    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.startswith(prefix) or not fname.endswith(".json"):
            continue
        sa = fname[len(prefix):-len(".json")]
        if sa == "expert": continue

        parts = sa.split("_")
        try:
            if parts[0] == "alg4":              alg, mt, L = "alg4", parts[1], int(parts[2][1:])
            elif parts[0] == "alg5":            alg, mt, L = "alg5", parts[1], None
            elif parts[0] in ("alg6", "alg7"):  alg, mt, L = parts[0], None, None
            else: continue
        except: continue

        with open(os.path.join(RESULTS_DIR, fname)) as f:
            fd = json.load(f)
            for k, v in fd.items():
                if k == "expert":
                    results.setdefault("expert", v)
                elif isinstance(v, dict):
                    results[_make_full_key(alg, mt, L, k)] = v
    return results

def get_size_stats(results, alg, mt, L, n_val, use_sem=False):
    xs, ms, ss = [], [], []
    for sz_key in SIZE_ORDER:
        key = _make_full_key(alg, mt, L, sz_key)
        d = results.get(key, {})
        vals = d.get(str(n_val), [])
        if vals:
            xs.append(SIZE_MAPPING[sz_key])
            ms.append(float(np.mean(vals)))
            std_val = float(np.std(vals))
            ss.append(std_val / np.sqrt(len(vals)) if use_sem else std_val)
    return xs, ms, ss

# ---------------------------------------------------------------------------
# LaTeX Generation
# ---------------------------------------------------------------------------

def _ratio_label(width, expert_width=EXPERT_WIDTH):
    """Return a LaTeX fraction string for width/expert_width in lowest terms."""
    from math import gcd
    g = gcd(width, expert_width)
    num, den = width // g, expert_width // g
    if den == 1:
        return f"${num}$"
    return f"$\\frac{{{num}}}{{{den}}}$"

def build_tex(all_data, use_sem=False):
    color_defs = "\n".join([
        f"\\definecolor{{algcol{i}}}{{HTML}}{{{COLORS_HEX[i]}}}"
        for i in range(len(COLORS_HEX))
    ])
    xtick_str = ",".join([str(v) for v in SIZE_MAPPING.values()])
    # Ratio labels: each width divided by EXPERT_WIDTH
    xticklabels_str = ", ".join(
        _ratio_label(v) for v in SIZE_MAPPING.values()
    )

    ovi_idx = next(i for i, (a, _, __) in enumerate(ALG_CONFIGS) if a == "alg4")
    ovi_clr = f"algcol{ovi_idx}"
    ovi_lw  = LINE_WIDTHS[ovi_idx]
    ovi_ls  = LINE_STYLES[ovi_idx]
    ovi_ms  = MARK_SIZES[ovi_idx]
    ovi_mrk = MARKERS[ovi_idx]
    ovi_lbl = ALG_LABELS[("alg4", "sm", 50)]

    preamble = (
        r"\documentclass[border=8pt]{standalone}" "\n"
        r"\usepackage{xcolor}" "\n"
        r"\usepackage{pgfplots}" "\n"
        r"\usepgfplotslibrary{fillbetween, groupplots}" "\n"
        r"\usetikzlibrary{positioning, calc}" "\n"
        r"\pgfplotsset{compat=1.18}" "\n"
        r"\usepackage{amsmath}" "\n"
        r"\usepackage{graphicx}" "\n"
        "\n"
        + color_defs + "\n"
        "\n"
        r"\pgfplotsset{" "\n"
        r"  size plot/.style={" "\n"
        r"    width=5.8cm," "\n"
        r"    height=5.8cm," "\n"
        r"    grid=both," "\n"
        r"    xmode=log," "\n"
        r"    log basis x={2}," "\n"
        f"    xmin={2**0.7:.4f}," "\n"
        f"    xmax={2**6.3:.4f}," "\n"
        r"    grid style={line width=0.35pt, draw=gray!35}," "\n"
        f"    xtick={{{xtick_str}}}," "\n"
        f"    xticklabels={{{xticklabels_str}}}," "\n"
        r"    xticklabel style={font=\normalsize}," "\n"
        r"    yticklabel style={font=\normalsize}," "\n"
        r"    ylabel style={font=\normalsize\bfseries, align=center}," "\n"
        r"    title style={font=\footnotesize}," "\n"
        r"  }" "\n"
        r"}" "\n"
        "\n"
        r"\begin{document}" "\n"
        r"  \begin{tikzpicture}" "\n"
        r"  \begin{groupplot}[" "\n"
        r"    size plot," "\n"
        r"    group style={" "\n"
        r"      group size=4 by 1," "\n"
        r"      horizontal sep=0.2cm," "\n"
        r"      xlabels at=edge bottom," "\n"
        r"      ylabels at=edge left," "\n"
        r"    }," "\n"
        r"  ]" "\n"
    )

    body_lines = []
    path_id = 0
    legend_placed = False

    for col, (env_tag, freq_tag) in enumerate(EXPERIMENTS):
        data = all_data[(env_tag, freq_tag)]
        is_first_col = (col == 0)

        ax_opts = [
            f"title={{{ENV_TITLES[env_tag]}}}",
            "ymin=0", "ymax=1.1",
        ]

        if is_first_col:
            ax_opts.append(r"ylabel={Normalized return}")
        else:
            ax_opts.append("yticklabels={}")

        if not legend_placed:
            ax_opts += [
                "legend to name=sharedlegend",
                "legend columns=4",
                (r"legend style={"
                 r"draw=none,"
                 r"font=\small,"
                 r"/tikz/every even column/.append style={column sep=0.5cm}"
                 r"}"),
            ]

        opt_block = ",\n    ".join(ax_opts)
        body_lines.append(f"  \\nextgroupplot[\n    {opt_block}\n  ]")

        # Pre-place OVI legend entry so it appears first (row 1, col 1)
        if not legend_placed:
            body_lines.append(
                f"    \\addlegendimage{{{ovi_clr}, {ovi_ls}, mark={ovi_mrk}, mark size={ovi_ms}pt, line width={ovi_lw}pt}}"
            )
            body_lines.append(f"    \\addlegendentry{{{ovi_lbl}}}")

        expert   = data.get("expert", 1.0)
        baseline = UNIFORM_RETURN.get(env_tag, 0.0)
        scale    = expert - baseline if expert != baseline else 1.0

        # Horizontal Expert return reference line — solid gray, drawn before algorithm
        # curves so it sits behind them in case of overlap.
        min_x = min(SIZE_MAPPING.values())
        max_x = max(SIZE_MAPPING.values())
        body_lines.append(
            f"    \\addplot[{EXPERT_LINE_STYLE}, forget plot] coordinates {{({min_x}, 1) ({max_x}, 1)}};"
        )

        for alg_idx, (alg, mt, L) in enumerate(ALG_CONFIGS):
            xs_raw, means, stds = get_size_stats(data, alg, mt, L, FIXED_N, use_sem=use_sem)
            if not xs_raw: continue

            ms_norm = [(m - baseline) / scale for m in means]
            ss_norm = [s / scale for s in stds]
            upper   = [m + s for m, s in zip(ms_norm, ss_norm)]
            lower   = [m - s for m, s in zip(ms_norm, ss_norm)]

            lbl    = ALG_LABELS[(alg, mt, L)]
            clr    = f"algcol{alg_idx}"
            opa    = LINE_OPACITIES[alg_idx]
            lw     = LINE_WIDTHS[alg_idx]
            ls     = LINE_STYLES[alg_idx]
            mrk    = MARKERS[alg_idx]
            msize  = MARK_SIZES[alg_idx]
            is_ovi = (alg == "alg4")
            uid    = path_id
            path_id += 1

            fill_part = "fill=white, " if mrk == "o" else ""
            mo_str = f", mark options={{{fill_part}solid}}"

            coords_upper = " ".join([f"({x},{y:.4f})" for x, y in zip(xs_raw, upper)])
            coords_lower = " ".join([f"({x},{y:.4f})" for x, y in zip(xs_raw, lower)])
            coords_mean  = " ".join([f"({x},{y:.4f})" for x, y in zip(xs_raw, ms_norm)])

            body_lines += [
                f"    \\addplot[name path=pu{uid}, draw=none, forget plot] coordinates {{ {coords_upper} }};",
                f"    \\addplot[name path=pl{uid}, draw=none, forget plot] coordinates {{ {coords_lower} }};",
                f"    \\addplot[fill={clr}, fill opacity={BAND_OPACITY}, draw=none, forget plot] fill between[of=pu{uid} and pl{uid}];",
            ]

            fp = ", forget plot" if is_ovi else ""
            body_lines.append(
                f"    \\addplot[{clr}, {ls}, mark={mrk}, mark size={msize}pt, line width={lw}pt, opacity={opa}{mo_str}{fp}] coordinates {{ {coords_mean} }};"
            )
            if not is_ovi and not legend_placed:
                body_lines.append(f"    \\addlegendentry{{{lbl}}}")

        legend_placed = True

    footer = (
        r"  \end{groupplot}" "\n"
        r"  % Centered single X-label" "\n"
        r"  \node[below=0.6cm, font=\normalsize\bfseries] at ($(group c2r1.south east)!0.5!(group c3r1.south west)$) {Learner/Expert hidden layer size ratio};" "\n"
        r"  % Single legend spanning the full combined plot width" "\n"
        r"  \newlength{\groupplotwidth}" "\n"
        r"  \pgfextractx{\groupplotwidth}{\pgfpointdiff{\pgfpointanchor{group c1r1}{west}}{\pgfpointanchor{group c4r1}{east}}}" "\n"
        r"  \node[below=1.2cm, anchor=north] at ($(group c1r1.south west)!0.5!(group c4r1.south east)$) {%" "\n"
        r"    \resizebox{\groupplotwidth}{!}{\pgfplotslegendfromname{sharedlegend}}%" "\n"
        r"  };" "\n"
        r"  \end{tikzpicture}" "\n"
        r"\end{document}"
    )

    return preamble + "\n".join(body_lines) + "\n" + footer

def main():
    parser = argparse.ArgumentParser(description="Generate Capacity Scaling LaTeX Figure (v8)")
    parser.add_argument("--divide", action="store_true", help="Use SEM instead of SD")
    args = parser.parse_args()

    all_data = {(env, freq): load_data(env, freq) for env, freq in EXPERIMENTS}
    tex_content = build_tex(all_data, use_sem=args.divide)

    with open(OUTPUT_TEX, "w") as f:
        f.write(tex_content)

    print(f"Written → {OUTPUT_TEX}")

if __name__ == "__main__":
    main()
