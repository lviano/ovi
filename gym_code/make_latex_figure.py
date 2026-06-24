#!/usr/bin/env python3
"""
make_latex_figure.py
-----------------------------
Generate a compilable LaTeX/pgfplots file that reproduces the IL comparison
figure as a 2 x 4 grid.

Usage:
  python make_latex_figure.py
  pdflatex results/combined_figure.tex

Output: results/combined_figure.tex
"""

import os
import json
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = "results"
OUTPUT_TEX  = os.path.join(RESULTS_DIR, "combined_figure.tex")

N_TRAJS = [1, 2, 3, 5, 7, 10, 30, 50, 100]

EXCLUDE_N = {}

N_TRAJS_OVERRIDE = {
    "Pendulum_v1_11bins": [1, 3, 5, 7, 10, 15],
    "LunarLander_v2":     [2, 3, 5, 10, 15],
}

EXPERIMENTS = [
    ("Acrobot_v1",         "_sub15"),
    ("CartPole_v1",        "_sub25"),
    ("Pendulum_v1_11bins", "_sub5"),
    ("LunarLander_v2",     "_sub5"),
]

ENV_TITLES = {
    "Acrobot_v1":         "Acrobot-v1",
    "CartPole_v1":        "CartPole-v1",
    "Pendulum_v1_11bins": "Pendulum-v1",
    "LunarLander_v2":     "LunarLander-v2",
}

# Plotting order: baselines first, OVI last (drawn on top).
ALG_CONFIGS = [
    ("alg5", "sm", None),   # SPOIL  – index 0
    ("alg7", None, None),   # DAgger – index 1
    ("alg6", None, None),   # BC     – index 2
    ("alg4", "sm", 50),     # OVI    – index 3 (plotted on top)
]

ALG_CONFIGS_OVERRIDE = {}

# Algorithm names colored with their respective line color via \textcolor{algcol<i>}
ALG_LABELS = {
    ("alg4", "sm", 50):   r"\shortstack[c]{{\large\textcolor{algcol3}{\textbf{\textsf{OVI (Ours)}}}}\\(online, value-based)}",
    ("alg5", "sm", None): r"\shortstack[c]{{\large\textcolor{algcol0}{\textbf{\textsf{SPOIL}}}}\\(offline, value-based)}",
    ("alg6", None, None): r"\shortstack[c]{{\large\textcolor{algcol2}{\textbf{\textsf{BC}}}}\\(offline, policy-based)}",
    ("alg7", None, None): r"\shortstack[c]{{\large\textcolor{algcol1}{\textbf{\textsf{DAgger}}}}\\(online, policy-based)}",
}

# SPOIL=blue, DAgger=purple, BC=green, OVI=red
COLORS_HEX     = ["3B7BC8", "7B6EBF", "389C56", "E63946"]
LINE_OPACITIES = [1.0,      1.0,      1.0,      1.0     ]
LINE_WIDTHS    = [1.0,      1.0,      1.0,      1.5     ]
LINE_STYLES    = ["dashed", "solid",  "dashed", "solid" ]
MARKERS        = ["*",      "o",      "o",      "*"     ]
MARK_SIZES     = [2.0,      2.0,      2.0,      2.0     ]

XTICK_OVERRIDE = {
    "Pendulum_v1_11bins": [1, 3, 5, 7, 10, 15],
    "LunarLander_v2":     [2, 3, 5, 10, 15],
}

SIZE_LABELS = {
    "large":         r"{\small\textcolor{blue}{$\pi_{\textsf{E}} \in \Pi$}, Learner width = 64}",
    "small":         r"{\small Learner width = 16}",
    "verysmall":     r"{\small Learner width = 8}",
    "veryverysmall": r"{\small Learner width = 4}",
    "tiny":          r"{\small Learner width = 2}",
}

# Expert return line: solid gray, same weight as a reference line
EXPERT_LINE_STYLE = "gray, solid, line width=1.5pt"

UNIFORM_RETURN = {
    "Acrobot_v1":         -500.0,
    "CartPole_v1":           8.0,
    "Pendulum_v1_11bins": -1650.0,
    "LunarLander_v2":      -600.0,
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _make_full_key(alg, mt, L, size):
    if alg == "alg4":   return f"alg4_{mt}_L{L}_{size}"
    elif alg == "alg5": return f"alg5_{mt}_{size}"
    else:               return f"{alg}_{size}"


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
        if sa == "expert":
            continue
        parts = sa.split("_")
        try:
            if   parts[0] == "alg4":            alg, mt, L = "alg4", parts[1], int(parts[2][1:])
            elif parts[0] == "alg5":            alg, mt, L = "alg5", parts[1], None
            elif parts[0] in ("alg6", "alg7"):  alg, mt, L = parts[0], None, None
            else:                               continue
        except Exception:
            continue

        with open(os.path.join(RESULTS_DIR, fname)) as f:
            fd = json.load(f)

        for k, v in fd.items():
            if k == "expert":
                results.setdefault("expert", v)
            elif isinstance(v, dict):
                results[_make_full_key(alg, mt, L, k)] = v
    return results


def _get_stats(results, alg, mt, L, size, ns, use_sem=False, max_seeds=None):
    key = _make_full_key(alg, mt, L, size)
    d = results.get(key, {})
    xs, ms, ss = [], [], []
    for n in ns:
        vals = d.get(str(n), [])
        if max_seeds is not None:
            vals = vals[:max_seeds]
        if vals:
            xs.append(n)
            ms.append(float(np.mean(vals)))
            s = float(np.std(vals))
            ss.append(s / np.sqrt(len(vals)) if use_sem else s)
    return xs, ms, ss


def _compute_norm(results, size, env_tag):
    expert_val = results.get("expert", 1.0)
    uniform = UNIFORM_RETURN.get(env_tag)
    if uniform is not None:
        baseline_val = uniform
    else:
        means_at_1 = []
        for alg, mt, L in ALG_CONFIGS_OVERRIDE.get(env_tag, ALG_CONFIGS):
            key = _make_full_key(alg, mt, L, size)
            vals = results.get(key, {}).get("1", [])
            if vals:
                means_at_1.append(float(np.mean(vals)))
        baseline_val = min(means_at_1) if means_at_1 else 0.0
    return baseline_val, expert_val


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def _coords(xs, ys):
    return " ".join(f"({x},{y:.5f})" for x, y in zip(xs, ys))


def _make_subplot(results, env_tag, size, ns, path_id,
                  ylabel_tex, title_tex, is_legend_source,
                  is_first_col=True, norm_min=0.0, norm_max=1.0, use_sem=False, max_seeds=None):
    expert = results.get("expert")
    scale = norm_max - norm_min if norm_max != norm_min else 1.0

    def _norm(v):
        return (v - norm_min) / scale

    alg_configs = ALG_CONFIGS_OVERRIDE.get(env_tag, ALG_CONFIGS)
    avail = set()
    for alg, mt, L in alg_configs:
        for n in results.get(_make_full_key(alg, mt, L, size), {}):
            avail.add(int(n))
    exclude = EXCLUDE_N.get(env_tag, set())
    plot_ns = sorted(n for n in ns if n in avail and n not in exclude) or sorted(avail)

    ax_opts = ["ymin=-0.1", "ymax=1.1"]
    if env_tag in XTICK_OVERRIDE:
        ticks = ",".join(str(n) for n in XTICK_OVERRIDE[env_tag])
        ax_opts.append(f"xtick={{{ticks}}}")
    if not is_first_col:
        ax_opts.append("yticklabels={}")
    if title_tex:
        ax_opts.append(f"title={{{title_tex}}}")
    if ylabel_tex:
        ax_opts.append(f"ylabel={{{ylabel_tex}}}")
    if is_legend_source:
        ax_opts += [
            "legend to name=sharedlegend",
            "legend columns=4",
            (r"legend style={"
             r"draw=none,"
             r"font=\small,"
             r"/tikz/every even column/.append style={column sep=0.5cm}"
             r"}"),
        ]

    if ax_opts:
        opt_block = ",\n    ".join(ax_opts)
        lines = [f"  \\nextgroupplot[\n    {opt_block}\n  ]"]
    else:
        lines = [r"  \nextgroupplot"]

    if is_legend_source:
        ovi_idx = next(i for i, (a, _, __) in enumerate(alg_configs) if a == "alg4")
        ovi_clr = f"algcol{ovi_idx}"
        ovi_lw  = LINE_WIDTHS[ovi_idx]
        ovi_ls  = LINE_STYLES[ovi_idx]
        ovi_ms  = MARK_SIZES[ovi_idx]
        ovi_mrk = MARKERS[ovi_idx]
        ovi_lbl = ALG_LABELS[("alg4", "sm", 50)]
        lines.append(
            f"  \\addlegendimage{{{ovi_clr}, {ovi_ls}, mark={ovi_mrk}, mark size={ovi_ms}pt, line width={ovi_lw}pt}}"
        )
        lines.append(f"  \\addlegendentry{{{ovi_lbl}}}")

    # Horizontal Expert return reference line — solid gray, drawn before algorithm
    # curves so it sits behind them in case of overlap.
    if expert is not None and plot_ns:
        lines.append(
            f"  \\addplot[{EXPERT_LINE_STYLE}, forget plot]"
            f" coordinates {{ ({plot_ns[0]},1.00000) ({plot_ns[-1]},1.00000) }};"
        )

    for i, (alg, mt, L) in enumerate(alg_configs):
        xs, ms, ss = _get_stats(results, alg, mt, L, size, plot_ns, use_sem=use_sem, max_seeds=max_seeds)
        if not xs:
            continue
        ms_norm = [_norm(m) for m in ms]
        ss_norm = [s / scale for s in ss]
        upper   = [m + s for m, s in zip(ms_norm, ss_norm)]
        lower   = [m - s for m, s in zip(ms_norm, ss_norm)]

        clr    = f"algcol{i}"
        opa    = LINE_OPACITIES[i]
        lw     = LINE_WIDTHS[i]
        ls     = LINE_STYLES[i]
        mrk    = MARKERS[i]
        msize  = MARK_SIZES[i]
        lbl    = ALG_LABELS[(alg, mt, L)]
        is_ovi = (alg == "alg4")
        uid    = path_id
        path_id += 1

        fill_part = "fill=white, " if mrk == "o" else ""
        mo_str = f", mark options={{{fill_part}solid}}"
        fp = ", forget plot" if is_ovi else ""

        lines += [
            f"  % {lbl}",
            f"  \\addplot[name path=pu{uid}, draw=none, forget plot]"
            f" coordinates {{ {_coords(xs, upper)} }};",
            f"  \\addplot[name path=pl{uid}, draw=none, forget plot]"
            f" coordinates {{ {_coords(xs, lower)} }};",
            f"  \\addplot[fill={clr}, fill opacity=0.28, draw=none, forget plot]"
            f" fill between[of=pu{uid} and pl{uid}];",
            f"  \\addplot[{clr}, {ls}, mark={mrk}, mark size={msize}pt, line width={lw}pt, opacity={opa}{mo_str}{fp}]"
            f" coordinates {{ {_coords(xs, ms_norm)} }};",
        ]
        if not is_ovi and is_legend_source:
            lines.append(f"  \\addlegendentry{{{lbl}}}")

    return "\n".join(lines), path_id


# ---------------------------------------------------------------------------
# Full document builder
# ---------------------------------------------------------------------------

def build_tex(all_data, sizes=None, use_sem=False, max_seeds=None):
    if sizes is None:
        sizes = ["large", "small"]
    xtick_str = ",".join(str(n) for n in N_TRAJS)

    color_defs = "\n".join(
        f"\\definecolor{{algcol{i}}}{{HTML}}{{{COLORS_HEX[i]}}}"
        for i in range(len(COLORS_HEX))
    )

    preamble = (
        r"\documentclass[border=8pt]{standalone}" "\n"
        r"\usepackage{pgfplots}" "\n"
        r"\usepgfplotslibrary{fillbetween, groupplots}" "\n"
        r"\usetikzlibrary{positioning, calc}" "\n"
        r"\pgfplotsset{compat=1.18}" "\n"
        r"\usepackage{amsmath}" "\n"
        r"\usepackage{graphicx}" "\n"
        r"\usepackage{xcolor}" "\n"
        "\n"
        + color_defs + "\n"
        "\n"
        r"\pgfplotsset{" "\n"
        r"  il plot/.style={" "\n"
        r"    width=5.8cm," "\n"
        r"    height=5.8cm," "\n"
        r"    grid=both," "\n"
        r"    grid style={line width=0.35pt, draw=gray!35}," "\n"
        f"    xtick={{{xtick_str}}}," "\n"
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
        r"    il plot," "\n"
        r"    group style={" "\n"
        f"      group size=4 by {len(sizes)}," "\n"
        r"      horizontal sep=0.2cm," "\n"
        r"      vertical sep=0.6cm," "\n"
        r"      xlabels at=edge bottom," "\n"
        r"      ylabels at=edge left," "\n"
        r"    }," "\n"
        r"  ]" "\n"
    )

    body_lines = []
    path_id = 0
    legend_placed = False

    for row, size in enumerate(sizes):
        size_label = SIZE_LABELS.get(size, size)
        for col, (env_tag, freq_tag) in enumerate(EXPERIMENTS):
            data = all_data[(env_tag, freq_tag)]
            is_first_col     = (col == 0)
            is_legend_source = (not legend_placed)

            title_tex = ENV_TITLES[env_tag] if row == 0 else ""

            if is_first_col:
                ylabel_tex = size_label
            else:
                ylabel_tex = ""

            norm_min, norm_max = _compute_norm(data, size, env_tag)

            ns_for_env = N_TRAJS_OVERRIDE.get(env_tag, N_TRAJS)
            code, path_id = _make_subplot(
                data, env_tag, size, ns_for_env, path_id,
                ylabel_tex=ylabel_tex,
                title_tex=title_tex,
                is_legend_source=is_legend_source,
                is_first_col=is_first_col,
                norm_min=norm_min,
                norm_max=norm_max,
                use_sem=use_sem,
                max_seeds=max_seeds,
            )
            body_lines.append(code)

            if is_legend_source:
                legend_placed = True

    last_row = len(sizes)
    footer = (
        r"  \end{groupplot}" "\n"
        r"  % Single shared Y-label spanning all rows on the left" "\n"
        f"  \\node[rotate=90, font=\\normalsize\\bfseries, anchor=center, yshift=1.5cm] at ($(group c1r1.north west)!0.5!(group c1r{last_row}.south west)$) {{Normalized return}};" "\n"
        r"  % Centered single X-label below all panels" "\n"
        f"  \\node[below=0.5cm, font=\\normalsize\\bfseries] at ($(group c2r{last_row}.south east)!0.5!(group c3r{last_row}.south west)$) {{Number of trajectories}};" "\n"
        r"  % Shared legend spanning the full combined width of the 4 plots" "\n"
        r"  \newlength{\groupplotwidth}" "\n"
        r"  \pgfextractx{\groupplotwidth}{\pgfpointdiff{\pgfpointanchor{group c1r1}{west}}{\pgfpointanchor{group c4r1}{east}}}" "\n"
        f"  \\node[below=1.2cm, anchor=north] at ($(group c1r{last_row}.south west)!0.5!(group c4r{last_row}.south east)$) {{%" "\n"
        r"    \resizebox{\groupplotwidth}{!}{\pgfplotslegendfromname{sharedlegend}}%" "\n"
        r"  };" "\n"
        r"  \end{tikzpicture}" "\n"
        "\n"
        r"\end{document}" "\n"
    )

    return preamble + "\n".join(body_lines) + "\n" + footer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate LaTeX/pgfplots IL comparison figure (v5 style)")
    parser.add_argument("--divide", action="store_true",
                        help="Use standard error of the mean instead of std for shaded bands")
    parser.add_argument("--max_seeds", type=int, default=None,
                        help="Maximum number of seeds to use per (N, config) entry (default: all)")
    parser.add_argument("--rows", nargs="+", default=["large", "small"],
                        metavar="SIZE",
                        help="Size keys to include as rows, in order (default: large small)")
    args = parser.parse_args()

    all_data = {
        (env_tag, freq_tag): load_data(env_tag, freq_tag)
        for env_tag, freq_tag in EXPERIMENTS
    }

    for (env_tag, freq_tag), data in all_data.items():
        n_keys = len([k for k in data if k != "expert"])
        print(f"  {env_tag}{freq_tag}: {n_keys} config keys, "
              f"expert={data.get('expert', 'N/A')}")

    tex = build_tex(all_data, sizes=args.rows, use_sem=args.divide, max_seeds=args.max_seeds)
    with open(OUTPUT_TEX, "w") as f:
        f.write(tex)

    print(f"\nWritten → {OUTPUT_TEX}")
    print("Compile with:")
    print(f"  pdflatex results/combined_figure.tex")


if __name__ == "__main__":
    main()
