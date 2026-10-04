"""Interactive reporting and diagnostic visualizations for importance_sampling.

Generates polished, production-grade interactive reports and dashboards for inspecting:
1. Model Fit & Convergence (Total Evidence, BIC, and subject-level evidence trajectories).
2. Model Parameters Summary Table.
3. Hyperparameter Evolution Grid (3 plots per row, solid mean line, shaded +/- 1 SD area).
4. Individual Subject Posterior Means (separate scatter plot per parameter: Y = subject #, X = parameter mean).
5. Multinormal Parameter Correlation / Covariance Matrix Heatmap.
"""

from __future__ import annotations

import math
import os
import tempfile
import webbrowser
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from importance_sampling.sampler import Sampler


def _is_notebook() -> bool:
    """Detect whether execution is occurring inside an interactive notebook environment."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        shell = ip.__class__.__name__
        # Terminal IPython shell cannot display rich interactive HTML
        if "Terminal" in shell:
            return False
        # Jupyter Notebook, JupyterLab, VS Code, or Google Colab
        if "ZMQ" in shell or "Colab" in shell or hasattr(ip, "kernel"):
            return True
        return False
    except Exception:
        return False


class ReportDashboard:
    """A self-contained, interactive HTML dashboard for model diagnostics."""

    def __init__(self, html_content: str, filename: Optional[str] = None, figure: Any = None):
        self.html_content = html_content
        self.filename = filename
        self.figure = figure

    def _repr_html_(self) -> str:
        """Render rich interactive HTML automatically in Jupyter and Google Colab."""
        import html
        escaped = html.escape(self.html_content, quote=True)
        return (
            f'<iframe srcdoc="{escaped}" '
            f'style="width: 100%; height: 860px; border: 1px solid #d1d9e0; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);" '
            f'frameborder="0"></iframe>'
        )

    def __repr__(self) -> str:
        dest = f" (saved to '{self.filename}')" if self.filename else ""
        return f"<ReportDashboard{dest}>"

    def save(self, filename: str) -> str:
        """Save the dashboard to an HTML file."""
        abs_path = os.path.abspath(filename)
        dir_name = os.path.dirname(abs_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(self.html_content)
        self.filename = abs_path
        return abs_path

    def show(self) -> None:
        """Open the dashboard in the default web browser."""
        if self.filename and os.path.exists(self.filename):
            target_path = self.filename
        else:
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html", encoding="utf-8") as f:
                f.write(self.html_content)
                target_path = f.name
            self.filename = target_path

        url = f"file://{os.path.abspath(target_path)}"
        print(f"Opening report in web browser: {url}")
        webbrowser.open(url)


def highlight_python_code_html(code: str) -> str:
    """Highlights Python source code with IDE-style syntax colors for HTML output."""
    if not code:
        return ""

    import html as py_html
    import re

    token_regex = re.compile(
        r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\''
        r'|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
        r'|#[^\n]*'
        r'|\bdef\s+([a-zA-Z_]\w*)'
        r'|\b(?:def|return|for|in|if|else|elif|import|from|as|and|or|not|while|yield|pass|break|continue|lambda|try|except|finally|raise|with|class)\b'
        r'|\b(?:True|False|None)\b'
        r'|\b(?:self)\b'
        r'|\b(?:np|zeros|ones|array|exp|log|max|min|sum|len|range|zip|enumerate|float|int|str|dict|list|set|bool|expit|softplus|sigmoid|clip|print|abs|round)\b'
        r'|\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b'
        r'|(==|!=|<=|>=|\+=|-=|\*=|/=|[-+*/=<>%]))'
    )

    keywords = {
        "def", "return", "for", "in", "if", "else", "elif", "import", "from",
        "as", "and", "or", "not", "while", "yield", "pass", "break", "continue",
        "lambda", "try", "except", "finally", "raise", "with", "class"
    }
    builtins = {
        "np", "zeros", "ones", "array", "exp", "log", "max", "min", "sum",
        "len", "range", "zip", "enumerate", "float", "int", "str", "dict",
        "list", "set", "bool", "expit", "softplus", "sigmoid", "clip", "print", "abs", "round"
    }

    result = []
    last_idx = 0
    for match in token_regex.finditer(code):
        start, end = match.span()
        if start > last_idx:
            result.append(py_html.escape(code[last_idx:start]))

        token = match.group(0)

        if token.startswith(('"""', "'''")):
            result.append(f'<span style="color: #7ee787; font-style: italic;">{py_html.escape(token)}</span>')
        elif token.startswith(('"', "'")):
            result.append(f'<span style="color: #a5d6ff;">{py_html.escape(token)}</span>')
        elif token.startswith('#'):
            result.append(f'<span style="color: #8b949e; font-style: italic;">{py_html.escape(token)}</span>')
        elif token.startswith("def "):
            func_name = match.group(2) or token[4:].strip()
            result.append(f'<span style="color: #ff7b72; font-weight: 600;">def</span> <span style="color: #d2a8ff; font-weight: 700;">{py_html.escape(func_name)}</span>')
        elif token in keywords:
            result.append(f'<span style="color: #ff7b72; font-weight: 600;">{py_html.escape(token)}</span>')
        elif token in ("True", "False", "None"):
            result.append(f'<span style="color: #79c0ff; font-weight: 600;">{py_html.escape(token)}</span>')
        elif token == "self":
            result.append(f'<span style="color: #ffa657; font-style: italic;">self</span>')
        elif token in builtins:
            result.append(f'<span style="color: #79c0ff;">{py_html.escape(token)}</span>')
        elif re.match(r'^\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$', token):
            result.append(f'<span style="color: #79c0ff;">{py_html.escape(token)}</span>')
        elif re.match(r'^(==|!=|<=|>=|\+=|-=|\*=|/=|[-+*/=<>%])$', token):
            result.append(f'<span style="color: #ff7b72;">{py_html.escape(token)}</span>')
        else:
            result.append(py_html.escape(token))

        last_idx = end

    if last_idx < len(code):
        result.append(py_html.escape(code[last_idx:]))

    return "".join(result)


def create_report(
    sampler: "Sampler",
    filename: Optional[str] = None,
    show: bool = True,
    transformed: bool = True,
    params_to_plot: Optional[Sequence[str]] = None,
    random_ll: Optional[float] = None,
    renderer: Optional[str] = None,
) -> ReportDashboard:
    """Generate a clean, beautiful, interactive HTML diagnostic report for a fitted model.

    Fixes common Plotly issues by generating a modular, card-based dashboard layout:
    - No overlapping titles or pooled, cluttered legends.
    - Each plot has its own dedicated card and axes.
    - Responsive 3-column grids for Hyperparameter Evolution and Subject Posterior Means.
    - Renders natively in Jupyter/Colab via `_repr_html_`, opens in browser via `.show()`,
      or exports to a standalone HTML file via `filename="report.html"`.

    Parameters
    ----------
    sampler : Sampler
        A fitted Sampler instance.
    filename : Optional[str], default=None
        Path to save the self-contained HTML report (e.g., 'model_report.html').
    show : bool, default=True
        Whether to open the report in the default browser or notebook environment.
    transformed : bool, default=True
        If True, displays parameters transformed into their valid domain bounds.
        If False, displays parameters in latent standard normal space.
    params_to_plot : Optional[Sequence[str]], default=None
        Subset of parameter keys to visualize in evolution and subject plots. If None, plots all parameters.
    random_ll : Optional[float], default=None
        User-provided chance baseline log-likelihood (0 parameters). Used to compute random BIC
        (-2 * random_ll) and display benchmark comparison metrics across the report.
    renderer : Optional[str], default=None
        Plotly renderer option if displaying a figure directly.

    Returns
    -------
    ReportDashboard
        A dashboard object supporting `.save()`, `.show()`, `.html_content`, and notebook rendering.
    """
    try:
        import plotly.graph_objects as go
        from plotly.io import to_html
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for interactive reports. Install it via: pip install plotly"
        ) from exc

    all_params = sampler.params
    active_params = [p for p in params_to_plot if p in all_params] if params_to_plot else all_params
    iters = list(range(len(sampler.hyper_params_list)))
    fit_iters = list(range(len(sampler.evidence)))

    eff_random_ll = float(random_ll) if random_ll is not None else getattr(sampler, "random_ll", None)
    eff_random_bic = (-2.0 * eff_random_ll) if eff_random_ll is not None else None

    palette = [
        "#0969da",
        "#1a7f37",
        "#8250df",
        "#cf222e",
        "#bf8700",
        "#0550ae",
        "#116329",
        "#5a32a3",
        "#82071e",
        "#7d4e00",
    ]

    plotly_config = {
        "responsive": True,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "displaylogo": False,
    }

    # -------------------------------------------------------------------------
    # 1. PLOT: Total Evidence & BIC
    # -------------------------------------------------------------------------
    fig_fit = go.Figure()
    fig_fit.add_trace(
        go.Scatter(
            x=fit_iters,
            y=sampler.evidence,
            mode="lines+markers",
            line=dict(color="#0969da", width=2.5),
            marker=dict(size=5.5),
            name="Total Evidence (LL)",
            hovertemplate="Iter %{x}<br><b>Evidence</b>: %{y:.2f}<extra></extra>",
        )
    )

    if eff_random_ll is not None and fit_iters:
        fig_fit.add_trace(
            go.Scatter(
                x=[fit_iters[0], fit_iters[-1]],
                y=[eff_random_ll, eff_random_ll],
                mode="lines",
                line=dict(color="#8c959f", width=1.8, dash="dot"),
                name=f"Random LL ({eff_random_ll:.1f})",
                hovertemplate=f"<b>Random Benchmark (k=0)</b><br>LL: {eff_random_ll:.2f}<extra></extra>",
            )
        )

    if sampler.BIC:
        fig_fit.add_trace(
            go.Scatter(
                x=fit_iters,
                y=sampler.BIC,
                mode="lines+markers",
                line=dict(color="#cf222e", width=2, dash="dash"),
                marker=dict(size=5),
                name="BIC",
                yaxis="y2",
                hovertemplate="Iter %{x}<br><b>BIC</b>: %{y:.2f}<extra></extra>",
            )
        )

    if eff_random_bic is not None and fit_iters and sampler.BIC:
        fig_fit.add_trace(
            go.Scatter(
                x=[fit_iters[0], fit_iters[-1]],
                y=[eff_random_bic, eff_random_bic],
                mode="lines",
                line=dict(color="#d97706", width=1.8, dash="dot"),
                name=f"Random BIC ({eff_random_bic:.1f})",
                yaxis="y2",
                hovertemplate=f"<b>Random Benchmark (k=0)</b><br>BIC: {eff_random_bic:.2f}<extra></extra>",
            )
        )

    fig_fit.update_layout(
        template="plotly_white",
        height=280,
        margin=dict(l=55, r=55, t=20, b=40),
        xaxis=dict(title="Iteration", gridcolor="#eaeef2"),
        yaxis=dict(title=dict(text="Total Evidence", font=dict(color="#0969da")), gridcolor="#eaeef2"),
        yaxis2=dict(
            title=dict(text="BIC", font=dict(color="#cf222e")),
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
        ),
        legend=dict(
            orientation="h",
            x=0.02,
            y=1.05,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(size=11),
        ),
        hovermode="x unified",
    )
    div_fit = to_html(fig_fit, include_plotlyjs=False, full_html=False, config=plotly_config)

    # -------------------------------------------------------------------------
    # 2. PLOT: Subject Evidence Spaghetti Plot
    # -------------------------------------------------------------------------
    fig_spaghetti = go.Figure()
    if sampler.subj_evidence:
        subj_mat = np.array(sampler.subj_evidence)
        for s in range(sampler.n_subjects):
            fig_spaghetti.add_trace(
                go.Scatter(
                    x=fit_iters,
                    y=subj_mat[:, s],
                    mode="lines",
                    line=dict(width=1, color="rgba(89, 99, 110, 0.35)"),
                    showlegend=False,
                    name=f"Subj {s}",
                    hovertemplate=f"Subj {s}<br>Iter %{{x}}: %{{y:.2f}}<extra></extra>",
                )
            )

        mean_ev = np.mean(subj_mat, axis=1)
        fig_spaghetti.add_trace(
            go.Scatter(
                x=fit_iters,
                y=mean_ev,
                mode="lines+markers",
                line=dict(color="#1f2328", width=2.5),
                marker=dict(size=5),
                name="Mean Subj Ev",
                hovertemplate="Iter %{x}<br><b>Mean Subj Ev</b>: %{y:.2f}<extra></extra>",
            )
        )

    fig_spaghetti.update_layout(
        template="plotly_white",
        height=280,
        margin=dict(l=55, r=20, t=20, b=40),
        xaxis=dict(title="Iteration", gridcolor="#eaeef2"),
        yaxis=dict(title="Subject Log Likelihood", gridcolor="#eaeef2"),
        legend=dict(
            orientation="h",
            x=0.02,
            y=1.05,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(size=11),
        ),
        hovermode="closest",
    )
    div_spaghetti = to_html(fig_spaghetti, include_plotlyjs=False, full_html=False, config=plotly_config)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # 3. PLOTS: Hyperparameter Evolution (One Clean Plot per Parameter)
    # -------------------------------------------------------------------------
    evolution_divs: List[Dict[str, Any]] = []
    space_label = "Transformed Space" if transformed else "Latent Normal Space"

    # Unified color scheme for parameter evolution:
    # A single consistent color for all parameters (which is also used for the grand mean),
    # and two distinct additional colors for groups (Group 1: Blue, Group 2: Crimson).
    grand_mean_color = "#1f2328"
    group_colors = ["#0969da", "#cf222e", "#8250df", "#1a7f37"]

    for idx, param in enumerate(active_params):
        t_func = sampler.transformations[param] if transformed else (lambda x: x)

        means_raw = np.array([hp[param]["mean"] for hp in sampler.hyper_params_list])
        sds_raw = np.array([hp[param]["sd"] for hp in sampler.hyper_params_list])

        means_plot = t_func(means_raw)
        upper_plot = t_func(means_raw + sds_raw)
        lower_plot = t_func(means_raw - sds_raw)

        fig_p = go.Figure()

        group_dict = getattr(sampler, "group_diff", None) or getattr(sampler, "params_with_group_diff", {})
        has_group_diff = param in group_dict

        # Ribbon upper bound
        fig_p.add_trace(
            go.Scatter(
                x=iters,
                y=upper_plot,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Ribbon lower bound with fill
        fig_p.add_trace(
            go.Scatter(
                x=iters,
                y=lower_plot,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(31, 35, 40, 0.12)",
                showlegend=False,
                name="±1 SD",
                hovertemplate="Iter %{x}<br><b>-1 SD</b>: %{y:.4f}<extra></extra>",
            )
        )

        # Grand Mean line (uses single unified color across all parameters)
        fig_p.add_trace(
            go.Scatter(
                x=iters,
                y=means_plot,
                mode="lines+markers",
                line=dict(color=grand_mean_color, width=2.5),
                marker=dict(size=4, color=grand_mean_color),
                showlegend=has_group_diff,
                name="Grand Mean" if has_group_diff else f"{param} Mean",
                hovertemplate=f"Iter %{{x}}<br><b>{'Grand Mean' if has_group_diff else param + ' Mean'}</b>: %{{y:.4f}}<extra></extra>",
            )
        )

        # Plot individual group means if group differences enabled using dedicated group colors
        if has_group_diff:
            for g_i, g in enumerate(sampler.group_names[param]):
                g_col = group_colors[g_i % len(group_colors)]
                g_means_hist = [
                    hp[param].get("group_means", {}).get(g, hp[param]["mean"])
                    for hp in sampler.hyper_params_list
                ]
                fig_p.add_trace(
                    go.Scatter(
                        x=iters,
                        y=[t_func(m) for m in g_means_hist],
                        mode="lines+markers",
                        line=dict(color=g_col, width=2.2, dash="dash" if g != sampler.ref_group[param] else "solid"),
                        marker=dict(size=4, color=g_col),
                        name=f"Group: {g}",
                        hovertemplate=f"Iter %{{x}}<br><b>Group {g}</b>: %{{y:.4f}}<extra></extra>",
                    )
                )

        fig_p.update_layout(
            template="plotly_white",
            height=220,
            margin=dict(l=45, r=15, t=15, b=35),
            xaxis=dict(title="Iteration", gridcolor="#eaeef2"),
            yaxis=dict(title=param, gridcolor="#eaeef2"),
            hovermode="closest",
            legend=dict(orientation="h", y=1.12, x=0.02, font=dict(size=10)),
        )

        div_p = to_html(fig_p, include_plotlyjs=False, full_html=False, config=plotly_config)
        evolution_divs.append({
            "param": param,
            "final_mean": means_plot[-1],
            "div": div_p,
        })

    # -------------------------------------------------------------------------
    # 4. PLOTS: Individual Subject Posterior Means (Y = Subj #, X = Value)
    # -------------------------------------------------------------------------
    subject_divs: List[Dict[str, Any]] = []
    subj_indices = list(range(sampler.n_subjects))

    if sampler.mean_params:
        for idx, param in enumerate(active_params):
            color = palette[idx % len(palette)]
            t_func = sampler.transformations[param] if transformed else (lambda x: x)
            vals = [t_func(v) for v in sampler.mean_params[param]]
            group_dict = getattr(sampler, "group_diff", None) or getattr(sampler, "params_with_group_diff", {})
            has_group_diff = param in group_dict

            fig_s = go.Figure()

            if has_group_diff:
                g_colors = ["#0969da", "#8250df", "#cf222e", "#bf8700"]
                for g_i, g in enumerate(sampler.group_names[param]):
                    g_col = g_colors[g_i % len(g_colors)]
                    sub_idx = [s for s in range(sampler.n_subjects) if sampler.subject_groups[param][s] == g]
                    if sub_idx:
                        fig_s.add_trace(
                            go.Scatter(
                                x=[vals[s] for s in sub_idx],
                                y=sub_idx,
                                mode="markers",
                                marker=dict(size=8, color=g_col, line=dict(color="#ffffff", width=1), opacity=0.85),
                                name=f"{g}",
                                hovertemplate=f"<b>Subj %{{y}} ({g})</b><br>{param}: %{{x:.4f}}<extra></extra>",
                            )
                        )
            else:
                fig_s.add_trace(
                    go.Scatter(
                        x=vals,
                        y=subj_indices,
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=color,
                            line=dict(color="#ffffff", width=1),
                            opacity=0.85,
                        ),
                        showlegend=False,
                        hovertemplate=f"<b>Subject %{{y}}</b><br>{param}: %{{x:.4f}}<extra></extra>",
                    )
                )

            fig_s.update_layout(
                template="plotly_white",
                height=220,
                margin=dict(l=45, r=15, t=15, b=35),
                xaxis=dict(title=f"{param} Mean", gridcolor="#eaeef2"),
                yaxis=dict(
                    title="Subject #",
                    gridcolor="#eaeef2",
                    dtick=max(1, sampler.n_subjects // 5),
                ),
                hovermode="closest",
                legend=dict(orientation="h", y=1.12, x=0.02, font=dict(size=10)),
            )

            div_s = to_html(fig_s, include_plotlyjs=False, full_html=False, config=plotly_config)
            subject_divs.append({
                "param": param,
                "min": min(vals),
                "max": max(vals),
                "div": div_s,
            })

    # -------------------------------------------------------------------------
    # 5. PLOT: Multinormal Correlation Matrix Heatmap
    # -------------------------------------------------------------------------
    cor_matrix = sampler.cor_matrix
    text_annotations = [[f"{val:.2f}" for val in row] for row in cor_matrix]

    fig_corr = go.Figure(
        data=go.Heatmap(
            z=cor_matrix,
            x=all_params,
            y=all_params,
            colorscale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            text=text_annotations,
            texttemplate="%{text}",
            textfont=dict(size=12, color="#1f2328"),
            colorbar=dict(title="r", len=0.8, thickness=16),
            hovertemplate="X: %{x}<br>Y: %{y}<br><b>Correlation</b>: %{z:.3f}<extra></extra>",
            showscale=True,
        )
    )

    fig_corr.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=60, r=20, t=20, b=50),
        xaxis=dict(title="Parameter", gridcolor="#eaeef2"),
        yaxis=dict(title="Parameter", gridcolor="#eaeef2"),
    )
    div_corr = to_html(fig_corr, include_plotlyjs=False, full_html=False, config=plotly_config)

    # -------------------------------------------------------------------------
    # HTML Table of Model Parameters
    # -------------------------------------------------------------------------
    table_rows_html = ""
    for p in all_params:
        t_func = sampler.transformations[p] if transformed else (lambda x: x)
        raw_m = sampler.hyper_params[p]["mean"]
        raw_sd = sampler.hyper_params[p]["sd"]
        fitted_mean = t_func(raw_m)
        fitted_sd = abs(t_func(raw_m + raw_sd) - t_func(raw_m - raw_sd)) / 2.0
        lower_bound = t_func(raw_m - raw_sd)
        upper_bound = t_func(raw_m + raw_sd)

        # Check for group differences on this parameter
        group_dict = getattr(sampler, "group_diff", None) or getattr(sampler, "params_with_group_diff", {})
        has_group_diff = p in group_dict
        group_badge = ""
        group_details_html = ""
        if has_group_diff:
            col_name = group_dict[p]
            g_means = sampler.group_means.get(p, {})
            g_diffs = sampler.group_diffs.get(p, {})
            ref_g = sampler.ref_group.get(p, "")
            group_badge = f'<span style="background: rgba(130, 80, 223, 0.12); color: #8250df; padding: 2px 6px; border-radius: 4px; font-size: 10px; margin-left: 6px;">Group Diff: {col_name}</span>'

            # Format group means in domain space
            g_means_str = ", ".join([f"<strong>{g}</strong>: {t_func(m):.4f}" for g, m in g_means.items()])
            diffs_str = ", ".join([f"<strong>Δ({pair})</strong> = {t_func(g_means[pair.split(' - ')[0]]) - t_func(g_means[ref_g]):+.4f} (latent: {d:+.4f})" for pair, d in g_diffs.items()])
            group_details_html = f"""
            <div style="font-size: 11px; margin-top: 4px; color: #59636e; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
                <span>Group Means: {g_means_str}</span><br>
                <span style="color: #8250df; font-weight: 600;">{diffs_str}</span>
            </div>
            """

        table_rows_html += f"""
        <tr style="border-bottom: 1px solid #eaeef2;">
            <td style="padding: 10px 14px;">
                <span style="font-weight: 600; font-family: monospace;">{p}</span>{group_badge}
                {group_details_html}
            </td>
            <td style="padding: 10px 14px; font-family: monospace; color: #0969da; vertical-align: top;">{fitted_mean:.4f}</td>
            <td style="padding: 10px 14px; font-family: monospace; color: #59636e; vertical-align: top;">{fitted_sd:.4f}</td>
            <td style="padding: 10px 14px; font-family: monospace; color: #59636e; vertical-align: top;">[{lower_bound:.3f}, {upper_bound:.3f}]</td>
        </tr>
        """

    # -------------------------------------------------------------------------
    # Assemble Clean Multi-Page Standalone HTML Document
    # -------------------------------------------------------------------------
    import html as py_html
    import json
    from importance_sampling.utils import time_to_text

    final_evidence_str = f"{sampler.evidence[-1]:.2f}" if sampler.evidence else "N/A"
    final_bic_str = f"{sampler.BIC[-1]:.2f}" if sampler.BIC else "N/A"

    # Extract or generate comprehensive metadata dictionary
    if hasattr(sampler, "get_metadata") and callable(getattr(sampler, "get_metadata")):
        metadata = sampler.get_metadata()
    elif hasattr(sampler, "metadata") and isinstance(sampler.metadata, dict) and sampler.metadata:
        metadata = dict(sampler.metadata)
    else:
        metadata = {
            "model_name": sampler.model_name,
            "created_at": getattr(sampler, "creation_time", "N/A"),
            "last_fit_at": getattr(sampler, "last_fit_time", None) or getattr(sampler, "creation_time", "N/A"),
            "total_fit_time_seconds": getattr(sampler, "total_fit_time", 0.0),
            "total_fit_time_formatted": time_to_text(getattr(sampler, "total_fit_time", 0.0)),
            "iterations": getattr(sampler, "iterations", 0),
            "n_subjects": getattr(sampler, "n_subjects", 0),
            "n_params": getattr(sampler, "n_params", 0),
            "params": getattr(sampler, "params", []),
            "final_evidence": float(sampler.evidence[-1]) if sampler.evidence else None,
            "final_bic": float(sampler.BIC[-1]) if sampler.BIC else None,
            "model_code": getattr(sampler, "model_code", ""),
        }

    run_timestamp = metadata.get("last_fit_at") or metadata.get("timestamp") or getattr(sampler, "creation_time", "N/A")
    created_timestamp = metadata.get("created_at") or getattr(sampler, "creation_time", "N/A")
    fit_duration_text = metadata.get("total_fit_time_formatted") or time_to_text(getattr(sampler, "total_fit_time", 0.0))
    fit_duration_secs = f"{getattr(sampler, 'total_fit_time', 0.0):.2f}s"
    iterations_run = getattr(sampler, "iterations", 0)
    model_desc_str = getattr(sampler, "description", "") or metadata.get("description", "")

    # Benchmark comparison metrics vs Random Chance Baseline (k=0 free parameters)
    d_ll_rand_badge = ""
    d_bic_rand_badge = ""

    if eff_random_ll is not None and sampler.evidence:
        final_ev_val = sampler.evidence[-1]
        final_bic_val = sampler.BIC[-1] if sampler.BIC else None
        d_ll_val = final_ev_val - eff_random_ll
        d_bic_val = (final_bic_val - eff_random_bic) if (final_bic_val is not None and eff_random_bic is not None) else None

        d_ll_rand_badge = f'<div style="font-size: 11px; color: {"#1a7f37" if d_ll_val > 0 else "#cf222e"}; font-weight: 600; margin-top: 2px;">{d_ll_val:+.2f} vs Random</div>'
        if d_bic_val is not None:
            d_bic_rand_badge = f'<div style="font-size: 11px; color: {"#1a7f37" if d_bic_val < 0 else "#cf222e"}; font-weight: 600; margin-top: 2px;">{d_bic_val:+.2f} vs Random</div>'

    # Extract user model Python source code
    model_code_str = getattr(sampler, "model_code", "") or metadata.get("model_code", "")
    if not model_code_str and hasattr(sampler, "model"):
        try:
            import inspect
            model_code_str = inspect.getsource(sampler.model)
        except Exception:
            model_code_str = getattr(sampler.model, "__doc__", "") or str(sampler.model)

    highlighted_code = highlight_python_code_html(model_code_str.strip() or "# Model source code could not be inspected")
    code_lines = len(model_code_str.strip().splitlines()) if model_code_str.strip() else 0

    # Format companion metadata JSON file string
    raw_metadata_json = json.dumps(metadata, indent=2, default=str)
    escaped_json = py_html.escape(raw_metadata_json)

    evolution_cards_html = "".join([
        f"""
        <div style="background: #ffffff; border: 1px solid #eaeef2; border-radius: 8px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);">
            <div style="display: flex; justify-content: space-between; font-size: 12px; font-family: monospace; margin-bottom: 4px;">
                <span style="font-weight: 700; color: #1f2328;">{c['param']}</span>
                <span style="color: #0969da; font-weight: 600;">mean = {c['final_mean']:.4f}</span>
            </div>
            {c['div']}
        </div>
        """
        for c in evolution_divs
    ])

    subject_cards_html = "".join([
        f"""
        <div style="background: #ffffff; border: 1px solid #eaeef2; border-radius: 8px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);">
            <div style="display: flex; justify-content: space-between; font-size: 12px; font-family: monospace; margin-bottom: 4px;">
                <span style="font-weight: 700; color: #1f2328;">{c['param']}</span>
                <span style="color: #1a7f37; font-weight: 600;">range: [{c['min']:.2f}, {c['max']:.2f}]</span>
            </div>
            {c['div']}
        </div>
        """
        for c in subject_divs
    ])

    # Group Differences Dedicated Card
    group_diff_card_html = ""
    group_dict_all = getattr(sampler, "group_diff", None) or getattr(sampler, "params_with_group_diff", {})
    if group_dict_all:
        group_rows_html = ""
        for p, col_name in group_dict_all.items():
            t_func = sampler.transformations[p] if transformed else (lambda x: x)
            ref_g = sampler.ref_group.get(p, "")
            g_means = sampler.group_means.get(p, {})
            g_diffs = sampler.group_diffs.get(p, {})

            for pair, diff_val in g_diffs.items():
                target_g = pair.split(" - ")[0]
                m_ref = g_means.get(ref_g, 0.0)
                m_tar = g_means.get(target_g, 0.0)
                t_ref = t_func(m_ref)
                t_tar = t_func(m_tar)
                t_diff = t_tar - t_ref

                col_display = ".".join(str(c) for c in col_name) if isinstance(col_name, (list, tuple)) else str(col_name)
                group_rows_html += f"""
                <tr style="border-bottom: 1px solid #eaeef2;">
                    <td style="padding: 10px 14px; font-weight: 700; font-family: monospace;">{p}</td>
                    <td style="padding: 10px 14px; font-family: monospace; color: #59636e;">{col_display}</td>
                    <td style="padding: 10px 14px; font-family: monospace;">{ref_g}: <strong>{t_ref:.4f}</strong> (latent: {m_ref:.4f})</td>
                    <td style="padding: 10px 14px; font-family: monospace;">{target_g}: <strong>{t_tar:.4f}</strong> (latent: {m_tar:.4f})</td>
                    <td style="padding: 10px 14px; font-family: monospace; font-weight: 700; color: {'#0969da' if t_diff > 0 else '#cf222e'};">
                        {t_diff:+.4f} (latent: {diff_val:+.4f})
                    </td>
                </tr>
                """

        group_diff_card_html = f"""
        <!-- GROUP DIFFERENCES SUMMARY CARD -->
        <div class="card">
            <div class="card-header">
                <div>
                    <div class="card-title" style="display: flex; align-items: center; gap: 8px;">
                        <span>Group Differences Analysis</span>
                        <span style="background: rgba(130, 80, 223, 0.12); color: #8250df; padding: 2px 8px; border-radius: 999px; font-size: 11px;">
                            Active
                        </span>
                    </div>
                    <div class="card-subtitle">
                        Estimated separate group means and between-group shifts (2 + N-1 hyperparameters per parameter).
                    </div>
                </div>
            </div>
            <div style="overflow-x: auto; border: 1px solid #eaeef2; border-radius: 8px;">
                <table>
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Group Column</th>
                            <th>Reference Group Mean</th>
                            <th>Target Group Mean</th>
                            <th>Estimated Difference (Δ)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {group_rows_html}
                    </tbody>
                </table>
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Report: {sampler.model_name}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f6f8fa;
            color: #1f2328;
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1160px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .card {{
            background: #ffffff;
            border: 1px solid #d1d9e0;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(31,35,40,0.04);
        }}
        .card-header {{
            padding-bottom: 12px;
            margin-bottom: 16px;
            border-bottom: 1px solid #eaeef2;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .card-title {{
            font-size: 15px;
            font-weight: 700;
            color: #1f2328;
        }}
        .card-subtitle {{
            font-size: 12px;
            color: #59636e;
            margin-top: 2px;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .grid-3 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 16px;
        }}
        .grid-4 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
        }}
        @media (max-width: 768px) {{
            .grid-2 {{ grid-template-columns: 1fr; }}
            .grid-3 {{ grid-template-columns: 1fr; }}
            .grid-4 {{ grid-template-columns: 1fr; }}
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 600;
            background: rgba(9, 105, 218, 0.1);
            color: #0969da;
            border: 1px solid rgba(9, 105, 218, 0.2);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 12px;
        }}
        th {{
            background-color: #f6f8fa;
            color: #59636e;
            font-weight: 600;
            padding: 10px 14px;
            border-bottom: 1px solid #d1d9e0;
        }}

        /* MULTI-PAGE NAVIGATION BAR */
        .report-navbar {{
            background: #ffffff;
            border: 1px solid #d1d9e0;
            border-radius: 10px;
            padding: 8px 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            position: sticky;
            top: 16px;
            z-index: 100;
        }}
        .nav-tabs {{
            display: flex;
            gap: 8px;
        }}
        .nav-tab {{
            background: transparent;
            border: 1px solid transparent;
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 600;
            color: #59636e;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.15s ease;
        }}
        .nav-tab:hover {{
            background: #f6f8fa;
            color: #1f2328;
        }}
        .nav-tab.active {{
            background: rgba(9, 105, 218, 0.08);
            border-color: rgba(9, 105, 218, 0.25);
            color: #0969da;
        }}
        .nav-tab .tab-badge {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 700;
            background: #eaeef2;
            color: #59636e;
        }}
        .nav-tab.active .tab-badge {{
            background: #0969da;
            color: #ffffff;
        }}
        .nav-actions {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .action-btn {{
            background: #f6f8fa;
            border: 1px solid #d1d9e0;
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 12px;
            font-weight: 600;
            color: #1f2328;
            cursor: pointer;
            transition: background 0.15s;
        }}
        .action-btn:hover {{
            background: #eaeef2;
        }}

        /* PAGE VISIBILITY */
        .report-page {{
            display: none;
            flex-direction: column;
            gap: 20px;
            animation: fadeIn 0.15s ease-in-out;
        }}
        .report-page.active {{
            display: flex;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(3px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* PRINT OPTIMIZATION (SEVERAL PAGES) */
        @media print {{
            body {{
                background: #ffffff !important;
                padding: 0 !important;
            }}
            .no-print {{
                display: none !important;
            }}
            .report-page {{
                display: flex !important;
                page-break-after: always;
                break-after: page;
                margin-bottom: 24px;
            }}
            .report-page:last-child {{
                page-break-after: avoid;
                break-after: avoid;
            }}
            .card {{
                box-shadow: none !important;
                border: 1px solid #d1d9e0 !important;
            }}
        }}

        /* CODE BOX CONTAINER */
        .code-container {{
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 8px;
            overflow: hidden;
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
        }}
        .code-header {{
            background: #161b22;
            border-bottom: 1px solid #30363d;
            padding: 8px 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: #8b949e;
            font-size: 12px;
        }}
        .code-dots {{
            display: flex;
            gap: 6px;
        }}
        .dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }}
        .dot.red {{ background: #ff5f56; }}
        .dot.yellow {{ background: #ffbd2e; }}
        .dot.green {{ background: #27c93f; }}
        .copy-btn {{
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 4px;
            color: #c9d1d9;
            font-size: 11px;
            padding: 4px 8px;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .copy-btn:hover {{
            background: rgba(255, 255, 255, 0.15);
            color: #ffffff;
        }}
        .code-body {{
            padding: 14px;
            font-size: 12px;
            line-height: 1.6;
            color: #e6edf3;
            overflow-x: auto;
            max-height: 480px;
            white-space: pre;
        }}

        /* TOAST ALERT */
        #toast {{
            position: fixed;
            bottom: 24px;
            right: 24px;
            background: #1f2328;
            color: #ffffff;
            padding: 10px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            display: none;
            z-index: 1000;
        }}

        /* PAGINATION BUTTONS */
        .page-footer {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-top: 12px;
        }}
        .pager-btn {{
            background: #ffffff;
            border: 1px solid #d1d9e0;
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 600;
            color: #0969da;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.15s;
        }}
        .pager-btn:hover {{
            background: #f6f8fa;
            border-color: #0969da;
        }}
    </style>
</head>
<body>
    <div id="toast"></div>

    <div class="container">
        <!-- TOP HEADER HERO CARD -->
        <div class="card" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
            <div>
                <div style="margin-bottom: 6px;">
                    <span class="badge">Model Diagnostics Report</span>
                </div>
                <h1 style="font-size: 22px; font-weight: 800; color: #1f2328;">{sampler.model_name}</h1>
                {f'<p style="font-size: 13px; font-weight: 500; color: #0969da; margin-top: 2px;">{py_html.escape(model_desc_str)}</p>' if model_desc_str else ''}
                <p style="font-size: 13px; color: #59636e; margin-top: 2px;">
                    <strong>{sampler.n_subjects}</strong> Subjects • 
                    <strong>{len(all_params)}</strong> Parameters • 
                    <strong>{iterations_run}</strong> Iterations • 
                    Fit Time: <strong>{fit_duration_text}</strong> ({fit_duration_secs}) • 
                    Run: <strong>{run_timestamp}</strong>
                </p>
            </div>
            <div style="display: flex; gap: 12px; font-family: monospace;">
                <div style="background: #f6f8fa; border: 1px solid #d1d9e0; border-radius: 8px; padding: 10px 16px; text-align: center;">
                    <div style="font-size: 10px; text-transform: uppercase; color: #59636e; font-weight: 600;">Total Evidence</div>
                    <div style="font-size: 18px; font-weight: 700; color: #0969da;">{final_evidence_str}</div>
                    {d_ll_rand_badge}
                </div>
                <div style="background: #f6f8fa; border: 1px solid #d1d9e0; border-radius: 8px; padding: 10px 16px; text-align: center;">
                    <div style="font-size: 10px; text-transform: uppercase; color: #59636e; font-weight: 600;">Final BIC</div>
                    <div style="font-size: 18px; font-weight: 700; color: #1a7f37;">{final_bic_str}</div>
                    {d_bic_rand_badge}
                </div>
            </div>
        </div>

        <!-- MULTI-PAGE TAB NAVIGATION BAR -->
        <nav class="report-navbar no-print">
            <div class="nav-tabs">
                <button class="nav-tab active" data-page="page-general-fit" onclick="switchPage('page-general-fit')">
                    <span class="tab-badge">1</span>
                    <span>General Fit</span>
                </button>
                <button class="nav-tab" data-page="page-parameters" onclick="switchPage('page-parameters')">
                    <span class="tab-badge">2</span>
                    <span>Parameters</span>
                </button>
                <button class="nav-tab" data-page="page-model-metadata" onclick="switchPage('page-model-metadata')">
                    <span class="tab-badge">3</span>
                    <span>Model &amp; Metadata</span>
                </button>
            </div>
            <div class="nav-actions">
                <button class="action-btn" onclick="window.print()">
                    🖨️ Print / Save as PDF
                </button>
            </div>
        </nav>

        <!-- ================================================================= -->
        <!-- PAGE 1: GENERAL FIT                                               -->
        <!-- ================================================================= -->
        <div id="page-general-fit" class="report-page active">
            <div class="card" style="padding: 12px 16px; background: #ffffff; border-left: 4px solid #0969da;">
                <div style="font-size: 13px; font-weight: 700; color: #1f2328;">Page 1: General Model Fit &amp; Convergence Diagnostics</div>
                <div style="font-size: 12px; color: #59636e;">Overall population evidence trajectory, BIC minimization path, and subject-level convergence stability.</div>
            </div>

            <div class="grid-2">
                <div class="card">
                    <div class="card-header">
                        <div>
                            <div class="card-title">Total Model Evidence &amp; BIC</div>
                            <div class="card-subtitle">Dual-axis evolution across estimation iterations</div>
                        </div>
                    </div>
                    {div_fit}
                </div>

                <div class="card">
                    <div class="card-header">
                        <div>
                            <div class="card-title">Subject-Level Evidence Trajectories</div>
                            <div class="card-subtitle">Per-subject log marginal likelihood convergence paths</div>
                        </div>
                    </div>
                    {div_spaghetti}
                </div>
            </div>

            <!-- FIT SUMMARY & CONVERGENCE DIAGNOSTICS CARD -->
            <div class="card">
                <div class="card-header">
                    <div>
                        <div class="card-title">Fit &amp; Convergence Summary</div>
                        <div class="card-subtitle">Numerical diagnostic metrics at the final estimation iteration</div>
                    </div>
                </div>
                <div class="grid-4" style="font-family: monospace;">
                    <div style="background: #f6f8fa; border: 1px solid #eaeef2; border-radius: 8px; padding: 12px;">
                        <div style="font-size: 11px; color: #59636e; text-transform: uppercase;">Total Evidence (LL)</div>
                        <div style="font-size: 16px; font-weight: 700; color: #0969da; margin-top: 4px;">{final_evidence_str}</div>
                        <div style="font-size: 10px; color: #8c959f; font-family: sans-serif; margin-top: 2px;">Sum of log-marginal likelihoods</div>
                    </div>
                    <div style="background: #f6f8fa; border: 1px solid #eaeef2; border-radius: 8px; padding: 12px;">
                        <div style="font-size: 11px; color: #59636e; text-transform: uppercase;">Bayesian Information Crit.</div>
                        <div style="font-size: 16px; font-weight: 700; color: #1a7f37; margin-top: 4px;">{final_bic_str}</div>
                        <div style="font-size: 10px; color: #8c959f; font-family: sans-serif; margin-top: 2px;">Penalized for {len(all_params)} free parameters</div>
                    </div>
                    <div style="background: #f6f8fa; border: 1px solid #eaeef2; border-radius: 8px; padding: 12px;">
                        <div style="font-size: 11px; color: #59636e; text-transform: uppercase;">Iterations Run</div>
                        <div style="font-size: 16px; font-weight: 700; color: #1f2328; margin-top: 4px;">{iterations_run} iters</div>
                        <div style="font-size: 10px; color: #8c959f; font-family: sans-serif; margin-top: 2px;">Duration: {fit_duration_text}</div>
                    </div>
                    <div style="background: #f6f8fa; border: 1px solid #eaeef2; border-radius: 8px; padding: 12px;">
                        <div style="font-size: 11px; color: #59636e; text-transform: uppercase;">Final ΔEvidence (window)</div>
                        <div style="font-size: 16px; font-weight: 700; color: #8250df; margin-top: 4px;">{sampler.evidence_change[-1]:+.4f}</div>
                        <div style="font-size: 10px; color: #8c959f; font-family: sans-serif; margin-top: 2px;">Average change per iteration</div>
                    </div>
                </div>
            </div>

            <!-- PAGE 1 FOOTER NAVIGATION -->
            <div class="page-footer no-print">
                <div></div>
                <button class="pager-btn" onclick="switchPage('page-parameters')">
                    Next Page: Parameters &rarr;
                </button>
            </div>
        </div>

        <!-- ================================================================= -->
        <!-- PAGE 2: PARAMETERS                                                -->
        <!-- ================================================================= -->
        <div id="page-parameters" class="report-page">
            <div class="card" style="padding: 12px 16px; background: #ffffff; border-left: 4px solid #1a7f37;">
                <div style="font-size: 13px; font-weight: 700; color: #1f2328;">Page 2: Model Parameters &amp; Population Distributions</div>
                <div style="font-size: 12px; color: #59636e;">Summary table, between-group differences, hyperparameter trajectories, and subject-level posterior estimates.</div>
            </div>

            <!-- PARAMETERS SUMMARY TABLE -->
            <div class="card">
                <div class="card-header">
                    <div>
                        <div class="card-title">Model Parameters Summary</div>
                        <div class="card-subtitle">Fitted population mean and credible ±1 SD intervals ({space_label})</div>
                    </div>
                </div>
                <div style="overflow-x: auto; border: 1px solid #eaeef2; border-radius: 8px;">
                    <table>
                        <thead>
                            <tr>
                                <th>Parameter Key</th>
                                <th>Fitted Mean (μ)</th>
                                <th>Fitted SD (σ)</th>
                                <th>±1 SD Credible Bound</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows_html}
                        </tbody>
                    </table>
                </div>
            </div>

            {group_diff_card_html}

            <!-- HYPERPARAMETER EVOLUTION GRID (3 PER ROW) -->
            <div class="card">
                <div class="card-header">
                    <div>
                        <div class="card-title">Hyperparameter Evolution Grid</div>
                        <div class="card-subtitle">Dedicated subplots (3 per row). Solid line = population mean; shaded ribbon = ±1 SD</div>
                    </div>
                </div>
                <div class="grid-3">
                    {evolution_cards_html}
                </div>
            </div>

            <!-- INDIVIDUAL SUBJECT POSTERIOR MEANS (3 PER ROW) -->
            <div class="card">
                <div class="card-header">
                    <div>
                        <div class="card-title">Individual Subject Posterior Means</div>
                        <div class="card-subtitle">Scatter plots: Y-axis = Subject Number, X-axis = Parameter Posterior Mean</div>
                    </div>
                </div>
                <div class="grid-3">
                    {subject_cards_html}
                </div>
            </div>

            <!-- MULTINORMAL CORRELATION MATRIX HEATMAP -->
            <div class="card">
                <div class="card-header">
                    <div>
                        <div class="card-title">Multinormal Parameter Correlation Matrix</div>
                        <div class="card-subtitle">Correlation / covariance between all parameters in latent space</div>
                    </div>
                </div>
                <div style="display: flex; justify-content: center;">
                    <div style="width: 100%; max-width: 600px;">
                        {div_corr}
                    </div>
                </div>
            </div>

            <!-- PAGE 2 FOOTER NAVIGATION -->
            <div class="page-footer no-print">
                <button class="pager-btn" onclick="switchPage('page-general-fit')">
                    &larr; Previous: General Fit
                </button>
                <button class="pager-btn" onclick="switchPage('page-model-metadata')">
                    Next Page: Model &amp; Metadata &rarr;
                </button>
            </div>
        </div>

        <!-- ================================================================= -->
        <!-- PAGE 3: MODEL & METADATA                                          -->
        <!-- ================================================================= -->
        <div id="page-model-metadata" class="report-page">
            <div class="card" style="padding: 12px 16px; background: #ffffff; border-left: 4px solid #8250df;">
                <div style="font-size: 13px; font-weight: 700; color: #1f2328;">Page 3: Model Specification &amp; Run Metadata</div>
                <div style="font-size: 12px; color: #59636e;">Source code of the computational model, execution timestamps, duration, and companion metadata file contents.</div>
            </div>

            <!-- MODEL DESCRIPTION CARD (IF AVAILABLE) -->
            {f'''<div class="card" style="padding: 14px 18px; background: rgba(9, 105, 218, 0.03); border-left: 4px solid #0969da;">
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #0969da; margin-bottom: 3px;">Model Description</div>
                <div style="font-size: 13px; color: #1f2328; font-weight: 500;">{py_html.escape(model_desc_str)}</div>
            </div>''' if model_desc_str else ''}

            <!-- METADATA CARDS GRID -->
            <div class="grid-4">
                <div class="card" style="padding: 16px;">
                    <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #59636e; margin-bottom: 8px;">
                        🕒 Timestamps
                    </div>
                    <div style="font-size: 12px; line-height: 1.7; font-family: monospace;">
                        <div><span style="color: #8c959f;">Run At:</span> <strong>{run_timestamp}</strong></div>
                        <div><span style="color: #8c959f;">Created:</span> {created_timestamp}</div>
                        <div><span style="color: #8c959f;">Saved:</span> {metadata.get("saved_at", run_timestamp)}</div>
                    </div>
                </div>

                <div class="card" style="padding: 16px;">
                    <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #59636e; margin-bottom: 8px;">
                        ⏱️ Execution Time
                    </div>
                    <div style="font-size: 12px; line-height: 1.7; font-family: monospace;">
                        <div><span style="color: #8c959f;">Total Duration:</span> <strong style="color: #0969da;">{fit_duration_text}</strong></div>
                        <div><span style="color: #8c959f;">Exact Seconds:</span> {fit_duration_secs}</div>
                        <div><span style="color: #8c959f;">Iterations:</span> <strong>{iterations_run}</strong> completed</div>
                    </div>
                </div>

                <div class="card" style="padding: 16px;">
                    <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #59636e; margin-bottom: 8px;">
                        👥 Model Dimensions
                    </div>
                    <div style="font-size: 12px; line-height: 1.7; font-family: monospace;">
                        <div><span style="color: #8c959f;">Subjects:</span> <strong>{sampler.n_subjects}</strong> subjects</div>
                        <div><span style="color: #8c959f;">Parameters:</span> <strong>{len(all_params)}</strong> ({", ".join(all_params)})</div>
                        <div><span style="color: #8c959f;">Multinormal:</span> {str(sampler.multinormal)}</div>
                    </div>
                </div>

                <div class="card" style="padding: 16px;">
                    <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #59636e; margin-bottom: 8px;">
                        ⚙️ System &amp; Env
                    </div>
                    <div style="font-size: 12px; line-height: 1.7; font-family: monospace;">
                        <div><span style="color: #8c959f;">Python:</span> {metadata.get('system_info', {}).get('python_version', '3.x')}</div>
                        <div><span style="color: #8c959f;">Platform:</span> {metadata.get('system_info', {}).get('platform', 'Linux/macOS/Windows').split('-')[0]}</div>
                        <div><span style="color: #8c959f;">Choices:</span> {sampler.n_choices} per decision</div>
                    </div>
                </div>
            </div>

            <!-- MODEL CODE PRESENTATION CARD -->
            <div class="card">
                <div class="card-header">
                    <div>
                        <div class="card-title">Model Source Code</div>
                        <div class="card-subtitle">Exact Python callable invoked during log-likelihood evaluation and simulation ({code_lines} lines) with standard IDE syntax coloring.</div>
                    </div>
                    <div style="display: flex; gap: 8px; flex-wrap: wrap;" class="no-print">
                        <button class="action-btn" id="single-code-expand-btn" onclick="toggleSingleCodeExpand()">⤢ Show All Code</button>
                        <button class="action-btn" onclick="copyModelCode()">📋 Copy Python Code</button>
                    </div>
                </div>
                <div class="code-container">
                    <div class="code-header">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <div class="code-dots">
                                <span class="dot red"></span>
                                <span class="dot yellow"></span>
                                <span class="dot green"></span>
                            </div>
                            <span style="color: #c9d1d9; font-weight: 600;">{getattr(sampler.model, '__name__', 'model')}.py</span>
                            <span style="font-size: 10px; color: #7ee787; background: rgba(126, 231, 135, 0.1); border: 1px solid rgba(126, 231, 135, 0.2); padding: 1px 6px; border-radius: 4px;">IDE Colors</span>
                        </div>
                        <span style="font-size: 11px; color: #8b949e;">Python • {code_lines} lines</span>
                    </div>
                    <pre class="code-body" id="single-model-code-pre"><code id="model-code-block">{highlighted_code}</code></pre>
                </div>
            </div>

            <!-- COMPANION METADATA DOWNLOAD ACTION (CODE PRESENTATION REMOVED AS REQUESTED) -->
            <script id="metadata-json-raw" type="application/json">{escaped_json}</script>
            <div class="card" style="padding: 14px 18px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 18px;">📁</span>
                    <div>
                        <div style="font-size: 13px; font-weight: 700; color: #1f2328;">Companion Run Metadata File</div>
                        <div style="font-size: 11px; color: #59636e;">All execution timestamps, dimensions, and hyperparameters shown in this report are serialized in <code>{sampler.model_name}_metadata.json</code>.</div>
                    </div>
                </div>
                <div style="display: flex; gap: 8px;" class="no-print">
                    <button class="action-btn" onclick="copyMetadataJson()">📋 Copy JSON</button>
                    <button class="action-btn" onclick="downloadMetadataJson('{sampler.model_name}_metadata.json')">💾 Download JSON</button>
                </div>
            </div>

            <!-- PAGE 3 FOOTER NAVIGATION -->
            <div class="page-footer no-print">
                <button class="pager-btn" onclick="switchPage('page-parameters')">
                    &larr; Previous Page: Parameters
                </button>
                <div></div>
            </div>
        </div>
    </div>

    <script>
        function switchPage(pageId) {{
            document.querySelectorAll('.report-page').forEach(function(el) {{
                el.classList.remove('active');
            }});
            document.querySelectorAll('.nav-tab').forEach(function(el) {{
                el.classList.remove('active');
            }});
            var target = document.getElementById(pageId);
            if (target) {{
                target.classList.add('active');
            }}
            var tab = document.querySelector('[data-page="' + pageId + '"]');
            if (tab) {{
                tab.classList.add('active');
            }}
            window.location.hash = pageId;
            setTimeout(function() {{
                window.dispatchEvent(new Event('resize'));
            }}, 60);
            window.scrollTo({{ top: 0, behavior: 'smooth' }});
        }}

        var isSingleCodeExpanded = false;
        function toggleSingleCodeExpand() {{
            isSingleCodeExpanded = !isSingleCodeExpanded;
            var pre = document.getElementById('single-model-code-pre');
            var btn = document.getElementById('single-code-expand-btn');
            if (pre) {{
                if (isSingleCodeExpanded) {{
                    pre.style.maxHeight = 'none';
                    pre.style.overflowY = 'visible';
                }} else {{
                    pre.style.maxHeight = '480px';
                    pre.style.overflowY = 'auto';
                }}
            }}
            if (btn) {{
                btn.innerText = isSingleCodeExpanded ? '⤡ Compact Window' : '⤢ Show All Code';
            }}
        }}

        function copyModelCode() {{
            var el = document.getElementById('model-code-block');
            if (!el) return;
            navigator.clipboard.writeText(el.innerText).then(function() {{
                showToast('✅ Model source code copied to clipboard!');
            }}).catch(function() {{
                showToast('Failed to copy to clipboard.');
            }});
        }}

        function copyMetadataJson() {{
            var el = document.getElementById('metadata-json-raw');
            if (!el) return;
            navigator.clipboard.writeText(el.textContent || el.innerText).then(function() {{
                showToast('✅ Metadata JSON copied to clipboard!');
            }}).catch(function() {{
                showToast('Failed to copy to clipboard.');
            }});
        }}

        function downloadMetadataJson(filename) {{
            var el = document.getElementById('metadata-json-raw');
            if (!el) return;
            var blob = new Blob([el.textContent || el.innerText], {{ type: 'application/json' }});
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            showToast('💾 Saved ' + filename);
        }}

        function showToast(msg) {{
            var toast = document.getElementById('toast');
            if (!toast) return;
            toast.innerText = msg;
            toast.style.display = 'block';
            setTimeout(function() {{
                toast.style.display = 'none';
            }}, 2400);
        }}

        window.addEventListener('DOMContentLoaded', function() {{
            var hash = window.location.hash ? window.location.hash.substring(1) : '';
            if (hash && document.getElementById(hash)) {{
                switchPage(hash);
            }}
        }});
    </script>
</body>
</html>
"""

    dashboard = ReportDashboard(html_content, filename=filename, figure=fig_fit)

    if filename:
        dashboard.save(filename)

    if show:
        if _is_notebook():
            try:
                import html
                from IPython.display import HTML, display
                escaped = html.escape(html_content, quote=True)
                iframe_html = (
                    f'<iframe srcdoc="{escaped}" '
                    f'style="width: 100%; height: 860px; border: 1px solid #d1d9e0; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);" '
                    f'frameborder="0"></iframe>'
                )
                display(HTML(iframe_html))
            except Exception:
                dashboard.show()
        else:
            dashboard.show()

    return dashboard


# -----------------------------------------------------------------------------
# Multi-Model Comparison Report
# -----------------------------------------------------------------------------


def compare_models(
    samplers: Union[Sequence["Sampler"], Dict[str, "Sampler"]],
    filename: Optional[str] = None,
    show: bool = True,
    compare_params: Optional[Sequence[str]] = None,
    transformed: bool = True,
    random_ll: Optional[float] = None,
    renderer: Optional[str] = None,
) -> ReportDashboard:
    """Compare multiple computational models across Likelihood/BICs, Parameter Inclusion, and Parameter Evolutions.

    Multi-Page Interactive Dashboard:
    - Page 1: Fit & BICs (Total evidence and BIC evolution, final ranking table, and % of participants best explained).
    - Page 2: Parameter Matrix & Code (Parameters on y-axis, models on x-axis, and collapsible model source code viewer).
    - Page 3: Parameter Evolution Comparison (Compares all parameters by default, plotting shared parameters together on the same subplots).

    Parameters
    ----------
    samplers : Union[Sequence[Sampler], Dict[str, Sampler]]
        Two or more fitted Sampler instances.
    filename : Optional[str], default=None
        If provided, saves as a self-contained HTML file.
    show : bool, default=True
        Whether to display the interactive figure or open in browser.
    compare_params : Optional[Sequence[str]], default=None
        List of parameter keys to compare. Defaults to ALL unique parameters across all compared models.
    transformed : bool, default=True
        If True, displays parameters in their valid domain space. If False, displays latent normal values.
    random_ll : Optional[float], default=None
        User-provided chance baseline log-likelihood (0 parameters). Used to compute random BIC
        (-2 * random_ll) and display benchmark comparison metrics and reference lines across all models.
    renderer : Optional[str], default=None
        Plotly renderer option.

    Returns
    -------
    ReportDashboard
        A dashboard object with `.save()`, `.show()`, and notebook rendering.
    """
    try:
        import plotly.graph_objects as go
        from plotly.io import to_html
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for model comparison reports. Install it via: pip install plotly"
        ) from exc

    import html as py_html
    import json
    import inspect

    if isinstance(samplers, dict):
        model_dict = samplers
    else:
        model_dict = {}
        for s in samplers:
            name = getattr(s, "model_name", None) or f"Model_{len(model_dict) + 1}"
            model_dict[name] = s

    if len(model_dict) < 2:
        raise ValueError("compare_models requires at least two fitted Sampler instances.")

    model_names = list(model_dict.keys())
    palette = [
        "#0969da",
        "#1a7f37",
        "#8250df",
        "#cf222e",
        "#bf8700",
        "#0550ae",
        "#116329",
        "#5a32a3",
        "#82071e",
        "#7d4e00",
    ]
    model_colors = {name: palette[i % len(palette)] for i, name in enumerate(model_names)}

    plotly_config = {
        "responsive": True,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "displaylogo": False,
    }

    # Collect all unique parameters across all models
    all_unique_params: List[str] = []
    for name in model_names:
        for p in model_dict[name].params:
            if p not in all_unique_params:
                all_unique_params.append(p)

    # Extract short model descriptions
    model_descriptions: Dict[str, str] = {}
    for name in model_names:
        s = model_dict[name]
        desc = getattr(s, "description", "")
        if not desc and hasattr(s, "metadata") and isinstance(s.metadata, dict):
            desc = s.metadata.get("description", "")
        model_descriptions[name] = desc or f"Model {name}"

    active_params = [p for p in compare_params if p in all_unique_params] if compare_params else all_unique_params

    # Model evaluation metrics
    final_evidence: List[float] = []
    final_bic: List[float] = []
    n_params_list: List[int] = []

    for name in model_names:
        m = model_dict[name]
        ev = m.evidence[-1] if m.evidence else np.nan
        bic = m.BIC[-1] if m.BIC else np.nan
        final_evidence.append(ev)
        final_bic.append(bic)
        n_params_list.append(len(m.params))

    valid_bics = [b for b in final_bic if not np.isnan(b)]
    best_bic = min(valid_bics) if valid_bics else 0.0
    delta_bic = [b - best_bic if not np.isnan(b) else np.nan for b in final_bic]

    sorted_indices = sorted(
        range(len(model_names)),
        key=lambda i: final_bic[i] if not np.isnan(final_bic[i]) else float("inf"),
    )
    ranks = [0] * len(model_names)
    for rank, idx in enumerate(sorted_indices, start=1):
        ranks[idx] = rank

    # -------------------------------------------------------------------------
    # Participant-level Best-fit Breakdown (% Explained by each model)
    # -------------------------------------------------------------------------
    first_sampler = next(iter(model_dict.values()))
    n_subjects = getattr(first_sampler, "n_subjects", 0)
    subj_wins: Dict[str, int] = {m: 0 for m in model_names}
    has_subj_evidence = n_subjects > 0

    if has_subj_evidence:
        for s in range(n_subjects):
            best_m = None
            best_s_ev = -float("inf")
            for m_name in model_names:
                m_obj = model_dict[m_name]
                if hasattr(m_obj, "subj_evidence") and m_obj.subj_evidence and len(m_obj.subj_evidence[-1]) > s:
                    val = m_obj.subj_evidence[-1][s]
                    if val > best_s_ev:
                        best_s_ev = val
                        best_m = m_name
            if best_m is not None:
                subj_wins[best_m] += 1
            else:
                has_subj_evidence = False
                break

    subj_win_pct = {
        m: (subj_wins[m] / n_subjects * 100.0) if n_subjects > 0 else 0.0
        for m in model_names
    }

    # -------------------------------------------------------------------------
    # Optional Chance Baseline Benchmark (k=0 free parameters)
    # -------------------------------------------------------------------------
    eff_random_ll = (
        float(random_ll)
        if random_ll is not None
        else next(
            (getattr(s, "random_ll", None) for s in model_dict.values() if getattr(s, "random_ll", None) is not None),
            None,
        )
    )
    eff_random_bic = (-2.0 * eff_random_ll) if eff_random_ll is not None else None

    # -------------------------------------------------------------------------
    # PAGE 1: Evidence & BIC Evolution + Participant Breakdown Plots
    # -------------------------------------------------------------------------
    fig_ev = go.Figure()
    for name in model_names:
        m = model_dict[name]
        fig_ev.add_trace(
            go.Scatter(
                x=list(range(len(m.evidence))),
                y=m.evidence,
                mode="lines+markers",
                name=name,
                line=dict(color=model_colors[name], width=2.5),
                marker=dict(size=4),
                hovertemplate=f"<b>{name}</b><br>Iter %{{x}}: Ev = %{{y:.2f}}<extra></extra>",
            )
        )

    if eff_random_ll is not None:
        max_ev_iters = max((len(m.evidence) for m in model_dict.values()), default=1)
        if max_ev_iters > 0:
            fig_ev.add_trace(
                go.Scatter(
                    x=[0, max_ev_iters - 1],
                    y=[eff_random_ll, eff_random_ll],
                    mode="lines",
                    line=dict(color="#8c959f", width=1.8, dash="dot"),
                    name=f"Random LL ({eff_random_ll:.1f})",
                    hovertemplate=f"<b>Random Benchmark (k=0)</b><br>LL: {eff_random_ll:.2f}<extra></extra>",
                )
            )

    fig_ev.update_layout(
        template="plotly_white",
        height=280,
        margin=dict(l=50, r=20, t=20, b=40),
        xaxis=dict(title="Iteration", gridcolor="#eaeef2"),
        yaxis=dict(title="Total Evidence (LL)", gridcolor="#eaeef2"),
        legend=dict(orientation="h", y=1.12, x=0.01),
        hovermode="x unified",
    )
    div_ev = to_html(fig_ev, include_plotlyjs=False, full_html=False, config=plotly_config)

    fig_bic = go.Figure()
    for name in model_names:
        m = model_dict[name]
        if m.BIC:
            fig_bic.add_trace(
                go.Scatter(
                    x=list(range(len(m.BIC))),
                    y=m.BIC,
                    mode="lines+markers",
                    name=name,
                    line=dict(color=model_colors[name], width=2.5, dash="dash"),
                    marker=dict(size=4),
                    hovertemplate=f"<b>{name}</b><br>Iter %{{x}}: BIC = %{{y:.2f}}<extra></extra>",
                )
            )

    if eff_random_bic is not None:
        max_bic_iters = max((len(m.BIC) for m in model_dict.values() if m.BIC), default=1)
        if max_bic_iters > 0:
            fig_bic.add_trace(
                go.Scatter(
                    x=[0, max_bic_iters - 1],
                    y=[eff_random_bic, eff_random_bic],
                    mode="lines",
                    line=dict(color="#8c959f", width=1.8, dash="dot"),
                    name=f"Random BIC ({eff_random_bic:.1f})",
                    hovertemplate=f"<b>Random Benchmark (k=0)</b><br>BIC: {eff_random_bic:.2f}<extra></extra>",
                )
            )

    fig_bic.update_layout(
        template="plotly_white",
        height=280,
        margin=dict(l=50, r=20, t=20, b=40),
        xaxis=dict(title="Iteration", gridcolor="#eaeef2"),
        yaxis=dict(title="BIC (Lower is Better)", gridcolor="#eaeef2"),
        legend=dict(orientation="h", y=1.12, x=0.01),
        hovermode="x unified",
    )
    div_bic = to_html(fig_bic, include_plotlyjs=False, full_html=False, config=plotly_config)

    # Best Explained Participants Horizontal Bar Chart
    fig_subj_bar = go.Figure()
    sorted_by_pct = sorted(model_names, key=lambda m: subj_win_pct[m], reverse=True)
    fig_subj_bar.add_trace(
        go.Bar(
            y=sorted_by_pct,
            x=[subj_win_pct[m] for m in sorted_by_pct],
            orientation="h",
            marker=dict(color=[model_colors[m] for m in sorted_by_pct]),
            text=[f"{subj_win_pct[m]:.1f}% ({subj_wins[m]}/{n_subjects} subj)" for m in sorted_by_pct],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Best explained: %{x:.1f}% of participants<extra></extra>",
        )
    )
    fig_subj_bar.update_layout(
        template="plotly_white",
        height=280,
        margin=dict(l=90, r=20, t=20, b=40),
        xaxis=dict(title="% of Participants Best Explained", range=[0, 105], gridcolor="#eaeef2"),
        yaxis=dict(title="", autorange="reversed", gridcolor="#eaeef2"),
    )
    div_subj_bar = to_html(fig_subj_bar, include_plotlyjs=False, full_html=False, config=plotly_config)

    # Comparison Ranking Table Rows (with Model Description and optional Random Benchmark columns)
    comp_rows_html = ""
    for idx in sorted_indices:
        m_name = model_names[idx]
        is_winner = ranks[idx] == 1
        d_bic = delta_bic[idx]
        d_bic_str = "0.0 (Best)" if is_winner else f"+{d_bic:.2f}"
        badge_style = "background: #1a7f37; color: white; padding: 2px 6px; border-radius: 4px; font-weight: 700;" if is_winner else "color: #59636e;"
        pct_display = f"{subj_win_pct[m_name]:.1f}% ({subj_wins[m_name]}/{n_subjects})" if has_subj_evidence else "N/A"
        m_desc = model_descriptions.get(m_name, "")

        rand_cols_html = ""
        if eff_random_ll is not None:
            d_ll_r = final_evidence[idx] - eff_random_ll
            d_bic_r = final_bic[idx] - eff_random_bic
            rand_cols_html = f"""
            <td style="padding: 10px 14px; font-family: monospace; color: {'#1a7f37' if d_ll_r > 0 else '#cf222e'}; font-weight: 600;">{d_ll_r:+.2f}</td>
            <td style="padding: 10px 14px; font-family: monospace; color: {'#1a7f37' if d_bic_r < 0 else '#cf222e'}; font-weight: 600;">{d_bic_r:+.2f}</td>
            """

        comp_rows_html += f"""
        <tr style="border-bottom: 1px solid #eaeef2; {'background: rgba(26, 127, 55, 0.04);' if is_winner else ''}">
            <td style="padding: 10px 14px;"><span style="{badge_style}">#{ranks[idx]}</span></td>
            <td style="padding: 10px 14px; font-weight: 700; font-family: monospace;">
                <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: {model_colors[m_name]}; margin-right: 6px;"></span>
                {m_name}
            </td>
            <td style="padding: 10px 14px; color: #59636e; font-size: 11px; max-width: 200px; line-height: 1.3;">
                {py_html.escape(m_desc)}
            </td>
            <td style="padding: 10px 14px; font-family: monospace; color: #59636e;">{n_params_list[idx]}</td>
            <td style="padding: 10px 14px; font-family: monospace; color: #0969da;">{final_evidence[idx]:.2f}</td>
            <td style="padding: 10px 14px; font-family: monospace; font-weight: 700;">{final_bic[idx]:.2f}</td>
            <td style="padding: 10px 14px; font-family: monospace; color: {'#1a7f37' if is_winner else '#cf222e'}; font-weight: 600;">{d_bic_str}</td>
            {rand_cols_html}
            <td style="padding: 10px 14px; font-family: monospace; font-weight: 600; color: #8250df;">{pct_display}</td>
        </tr>
        """

    # Add Random Baseline row to ranking table if benchmark provided
    if eff_random_ll is not None:
        d_bic_rand_vs_best = eff_random_bic - best_bic
        comp_rows_html += f"""
        <tr style="border-bottom: 1px solid #eaeef2; background: #f6f8fa;">
            <td style="padding: 10px 14px;"><span style="background: #8c959f; color: white; padding: 2px 6px; border-radius: 4px; font-weight: 700; font-size: 10px;">Benchmark</span></td>
            <td style="padding: 10px 14px; font-weight: 700; font-family: monospace; color: #59636e;">
                <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #8c959f; margin-right: 6px;"></span>
                Random Baseline
            </td>
            <td style="padding: 10px 14px; color: #59636e; font-size: 11px; max-width: 200px; line-height: 1.3;">
                Chance benchmark (k=0)
            </td>
            <td style="padding: 10px 14px; font-family: monospace; color: #59636e;">0</td>
            <td style="padding: 10px 14px; font-family: monospace; color: #59636e;">{eff_random_ll:.2f}</td>
            <td style="padding: 10px 14px; font-family: monospace; font-weight: 700; color: #59636e;">{eff_random_bic:.2f}</td>
            <td style="padding: 10px 14px; font-family: monospace; color: #cf222e; font-weight: 600;">+{d_bic_rand_vs_best:.2f}</td>
            <td style="padding: 10px 14px; font-family: monospace; color: #59636e;">0.0</td>
            <td style="padding: 10px 14px; font-family: monospace; color: #59636e;">0.0</td>
            <td style="padding: 10px 14px; font-family: monospace; font-weight: 600; color: #8c959f;">—</td>
        </tr>
        """

    random_table_headers_html = """
        <th>ΔLL (vs Random)</th>
        <th>ΔBIC (vs Random)</th>
    """ if eff_random_ll is not None else ""

    random_header_pill_html = ""
    if eff_random_ll is not None:
        random_header_pill_html = f"""
        <div style="background: #f6f8fa; border: 1px solid #d1d9e0; border-radius: 8px; padding: 8px 14px; text-align: right; font-family: monospace;">
            <div style="font-size: 10px; text-transform: uppercase; color: #59636e; font-weight: 600;">Random (k=0)</div>
            <div style="font-size: 14px; font-weight: 700; color: #59636e; margin-top: 2px;">
                LL: {eff_random_ll:.2f} &bull; BIC: {eff_random_bic:.2f}
            </div>
        </div>
        """

    # -------------------------------------------------------------------------
    # PAGE 2: Parameter Inclusion Matrix Table & Collapsible Model Functions Code
    # -------------------------------------------------------------------------
    # Model description summary cards for Page 2
    model_desc_cards_html = "".join([
        f"""
        <div style="background: #f6f8fa; border: 1px solid #d1d9e0; border-radius: 8px; padding: 10px 14px; flex: 1 1 200px; min-width: 190px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
                <span style="font-weight: 700; font-family: monospace; font-size: 12px; color: #1f2328; display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: {model_colors[name]};"></span>
                    {name}
                </span>
                <span style="font-size: 10px; color: #59636e; font-family: monospace; font-weight: 600;">k={len(model_dict[name].params)}</span>
            </div>
            <div style="font-size: 11px; color: #59636e; line-height: 1.35;">
                {py_html.escape(model_descriptions.get(name, ""))}
            </div>
        </div>
        """
        for name in model_names
    ])

    matrix_header_html = "".join([
        f"""<th style="text-align: center; padding: 10px 14px;">
            <div style="display: flex; flex-direction: column; align-items: center; gap: 2px;">
                <span style="color: {model_colors[name]}; font-family: monospace; font-weight: 700;">{name}</span>
                <span style="font-size: 10px; color: #59636e; font-weight: normal; max-width: 130px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{py_html.escape(model_descriptions.get(name, ''))}">{py_html.escape(model_descriptions.get(name, ''))}</span>
            </div>
        </th>"""
        for name in model_names
    ])

    # Simplistic matrix table: filled squares representing the parameters, with center dot for group differences
    matrix_rows_html = ""
    for p in all_unique_params:
        cells_html = ""
        for name in model_names:
            m = model_dict[name]
            group_diff_data = getattr(m, "group_diff", None) or getattr(m, "params_with_group_diff", None)
            if group_diff_data is None and hasattr(m, "metadata") and isinstance(m.metadata, dict):
                group_diff_data = m.metadata.get("group_diff", None)

            has_group_diff = False
            if isinstance(group_diff_data, dict):
                has_group_diff = p in group_diff_data
            elif isinstance(group_diff_data, (list, tuple, set)):
                has_group_diff = p in group_diff_data

            if p in m.params:
                if has_group_diff:
                    cells_html += f"""
                    <td style="padding: 10px 14px; text-align: center; vertical-align: middle;">
                        <span style="display: inline-flex; align-items: center; justify-content: center; width: 22px; height: 22px; border-radius: 4px; background: {model_colors[name]}; box-shadow: 0 1px 2px rgba(0,0,0,0.12);" title="{name} includes {p} (with group differences)">
                            <span style="width: 7px; height: 7px; border-radius: 50%; background: #ffffff; display: block; box-shadow: 0 0.5px 1px rgba(0,0,0,0.3);"></span>
                        </span>
                    </td>
                    """
                else:
                    cells_html += f"""
                    <td style="padding: 10px 14px; text-align: center; vertical-align: middle;">
                        <span style="display: inline-block; width: 22px; height: 22px; border-radius: 4px; background: {model_colors[name]}; box-shadow: 0 1px 2px rgba(0,0,0,0.12);" title="{name} includes {p}"></span>
                    </td>
                    """
            else:
                cells_html += f"""
                <td style="padding: 10px 14px; text-align: center; vertical-align: middle;">
                    <span style="display: inline-block; width: 22px; height: 22px; border-radius: 4px; border: 1.5px dashed #d1d9e0; background: rgba(246, 248, 250, 0.6);" title="{name} does not include {p}"></span>
                </td>
                """

        # Row highlighting if shared vs specific
        models_with_p = [name for name in model_names if p in model_dict[name].params]
        shared_badge = (
            f'<span style="background: rgba(130, 80, 223, 0.1); color: #8250df; padding: 2px 6px; border-radius: 4px; font-size: 10px; margin-left: 6px;">Shared ({len(models_with_p)})</span>'
            if len(models_with_p) > 1
            else f'<span style="background: #eaeef2; color: #59636e; padding: 2px 6px; border-radius: 4px; font-size: 10px; margin-left: 6px;">Unique</span>'
        )

        matrix_rows_html += f"""
        <tr style="border-bottom: 1px solid #eaeef2;">
            <td style="padding: 10px 14px; font-weight: 700; font-family: monospace;">
                {p} {shared_badge}
            </td>
            {cells_html}
        </tr>
        """

    # Footer row for parameter counts
    matrix_footer_html = "".join([
        f'<td style="padding: 10px 14px; text-align: center; font-weight: 700; font-family: monospace; color: {model_colors[name]};">k = {len(model_dict[name].params)}</td>'
        for name in model_names
    ])

    # Model source codes for collapsible viewer
    model_source_codes: Dict[str, str] = {}
    for name in model_names:
        m = model_dict[name]
        code_str = getattr(m, "model_code", "")
        if not code_str and hasattr(m, "model"):
            try:
                code_str = inspect.getsource(m.model)
            except Exception:
                code_str = getattr(m.model, "__doc__", "") or str(m.model)
        model_source_codes[name] = code_str.strip() or f"# No source code available for model {name}"

    model_code_tabs_html = "".join([
        f"""
        <button class="code-tab {'active' if idx == 0 else ''}" data-code-model="{name}" onclick="showModelCode('{name}')" style="border-bottom: 2px solid {'#0969da' if idx == 0 else 'transparent'};">
            <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: {model_colors[name]}; margin-right: 6px;"></span>
            {name}
        </button>
        """
        for idx, name in enumerate(model_names)
    ])

    model_code_blocks_html = "".join([
        f"""
        <div id="code-panel-{name}" class="code-panel {'active' if idx == 0 else ''}" style="display: {'block' if idx == 0 else 'none'};">
            <div class="code-header" style="background: #161b22; border-bottom: 1px solid #30363d; padding: 8px 14px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="display: flex; gap: 6px;">
                        <span style="width: 10px; height: 10px; border-radius: 50%; background: #ff5f56; display: inline-block;"></span>
                        <span style="width: 10px; height: 10px; border-radius: 50%; background: #ffbd2e; display: inline-block;"></span>
                        <span style="width: 10px; height: 10px; border-radius: 50%; background: #27c93f; display: inline-block;"></span>
                    </div>
                    <span style="color: #c9d1d9; font-weight: 600; font-size: 12px; font-family: monospace;">{name}.py</span>
                    <span style="font-size: 10px; color: #7ee787; background: rgba(126, 231, 135, 0.1); border: 1px solid rgba(126, 231, 135, 0.2); padding: 1px 6px; border-radius: 4px;">IDE Colors</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <button class="action-btn expand-code-btn" onclick="toggleCodeExpand()" style="font-size: 11px; padding: 4px 8px;">⤢ Show All Code</button>
                    <button class="action-btn" onclick="copySpecificCode('{name}')" style="font-size: 11px; padding: 4px 8px;">📋 Copy Code</button>
                </div>
            </div>
            <pre class="code-body" id="code-pre-{name}" style="padding: 14px; font-size: 12px; line-height: 1.6; color: #e6edf3; overflow-x: auto; max-height: 380px; margin: 0; background: #0d1117;"><code id="code-content-{name}">{highlight_python_code_html(model_source_codes[name])}</code></pre>
        </div>
        """
        for idx, name in enumerate(model_names)
    ])

    # -------------------------------------------------------------------------
    # PAGE 3: Parameter Evolution Comparison Plots
    # Shared parameters are plotted together on the same plot!
    # Unique parameters are plotted on their own plot.
    # -------------------------------------------------------------------------
    evolution_comparison_cards = ""
    for p in active_params:
        p_models = [name for name in model_names if p in model_dict[name].params]
        is_shared = len(p_models) > 1

        fig_p = go.Figure()
        for m_name in p_models:
            m = model_dict[m_name]
            p_history = m.hyper_params_list
            t_func = m.transformations.get(p, lambda x: x) if transformed else (lambda x: x)

            p_iters = list(range(len(p_history)))
            p_raw_means = [h[p]["mean"] for h in p_history]
            p_sds = [h[p]["sd"] for h in p_history]

            try:
                p_trans_means = [float(t_func(np.array([mu]))[0]) for mu in p_raw_means]
                p_upper = [float(t_func(np.array([mu + sd]))[0]) for mu, sd in zip(p_raw_means, p_sds)]
                p_lower = [float(t_func(np.array([mu - sd]))[0]) for mu, sd in zip(p_raw_means, p_sds)]
            except Exception:
                p_trans_means = p_raw_means
                p_upper = [mu + sd for mu, sd in zip(p_raw_means, p_sds)]
                p_lower = [mu - sd for mu, sd in zip(p_raw_means, p_sds)]

            c_hex = model_colors[m_name]
            # Shaded ribbon
            fig_p.add_trace(
                go.Scatter(
                    x=p_iters + p_iters[::-1],
                    y=p_upper + p_lower[::-1],
                    fill="toself",
                    fillcolor=f"rgba({int(c_hex[1:3], 16)}, {int(c_hex[3:5], 16)}, {int(c_hex[5:7], 16)}, 0.12)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{m_name} ±1 SD",
                )
            )
            # Mean line
            fig_p.add_trace(
                go.Scatter(
                    x=p_iters,
                    y=p_trans_means,
                    mode="lines+markers",
                    name=m_name,
                    line=dict(color=c_hex, width=2.5),
                    marker=dict(size=4),
                    hovertemplate=f"<b>{m_name}</b><br>Iter %{{x}}: mean = %{{y:.4f}}<extra></extra>",
                )
            )

        space_str = "Transformed Domain" if transformed else "Latent Space"
        fig_p.update_layout(
            template="plotly_white",
            height=260,
            margin=dict(l=45, r=15, t=15, b=35),
            xaxis=dict(title="Iteration", gridcolor="#eaeef2"),
            yaxis=dict(title=f"{p} Value", gridcolor="#eaeef2"),
            legend=dict(orientation="h", y=1.14, x=0.01),
            hovermode="x unified",
        )
        div_p = to_html(fig_p, include_plotlyjs=False, full_html=False, config=plotly_config)

        badge_html = (
            f'<span style="background: rgba(130, 80, 223, 0.1); color: #8250df; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700;">Shared ({len(p_models)} models)</span>'
            if is_shared
            else f'<span style="background: #eaeef2; color: #59636e; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600;">Specific to {p_models[0]}</span>'
        )

        evolution_comparison_cards += f"""
        <div style="background: #ffffff; border: 1px solid #eaeef2; border-radius: 8px; padding: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div style="font-weight: 700; font-family: monospace; font-size: 14px; color: #1f2328;">
                    {p}
                </div>
                {badge_html}
            </div>
            <div style="font-size: 11px; color: #59636e; margin-bottom: 4px;">
                {'Compared across: ' + ', '.join(p_models) if is_shared else 'Unique parameter in ' + p_models[0]} ({space_str})
            </div>
            {div_p}
        </div>
        """

    # -------------------------------------------------------------------------
    # Multi-Page HTML Document Assembly
    # -------------------------------------------------------------------------
    comp_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison Report: {len(model_names)} Models</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f6f8fa;
            color: #1f2328;
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1160px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .card {{
            background: #ffffff;
            border: 1px solid #d1d9e0;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(31,35,40,0.04);
        }}
        .card-header {{
            padding-bottom: 12px;
            margin-bottom: 16px;
            border-bottom: 1px solid #eaeef2;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .card-title {{
            font-size: 15px;
            font-weight: 700;
            color: #1f2328;
        }}
        .card-subtitle {{
            font-size: 12px;
            color: #59636e;
            margin-top: 2px;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .grid-3 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 16px;
        }}
        @media (max-width: 768px) {{
            .grid-2 {{ grid-template-columns: 1fr; }}
            .grid-3 {{ grid-template-columns: 1fr; }}
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 12px;
        }}
        th {{
            background-color: #f6f8fa;
            color: #59636e;
            font-weight: 600;
            padding: 10px 14px;
            border-bottom: 1px solid #d1d9e0;
        }}

        /* MULTI-PAGE NAVIGATION BAR */
        .report-navbar {{
            background: #ffffff;
            border: 1px solid #d1d9e0;
            border-radius: 10px;
            padding: 8px 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            position: sticky;
            top: 16px;
            z-index: 100;
        }}
        .nav-tabs {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .nav-tab {{
            background: transparent;
            border: 1px solid transparent;
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 600;
            color: #59636e;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.15s ease;
        }}
        .nav-tab:hover {{
            background: #f6f8fa;
            color: #1f2328;
        }}
        .nav-tab.active {{
            background: rgba(9, 105, 218, 0.08);
            border-color: rgba(9, 105, 218, 0.25);
            color: #0969da;
        }}
        .nav-tab .tab-badge {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 700;
            background: #eaeef2;
            color: #59636e;
        }}
        .nav-tab.active .tab-badge {{
            background: #0969da;
            color: #ffffff;
        }}
        .action-btn {{
            background: #f6f8fa;
            border: 1px solid #d1d9e0;
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 12px;
            font-weight: 600;
            color: #1f2328;
            cursor: pointer;
            transition: background 0.15s;
        }}
        .action-btn:hover {{
            background: #eaeef2;
        }}

        /* PAGE VISIBILITY */
        .report-page {{
            display: none;
            flex-direction: column;
            gap: 20px;
            animation: fadeIn 0.15s ease-in-out;
        }}
        .report-page.active {{
            display: flex;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(3px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* CODE VIEWER TABS */
        .code-tab {{
            background: transparent;
            border: none;
            padding: 8px 14px;
            font-size: 12px;
            font-weight: 600;
            font-family: monospace;
            color: #59636e;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .code-tab:hover {{
            color: #1f2328;
            background: #f6f8fa;
        }}
        .code-tab.active {{
            color: #0969da;
            font-weight: 700;
        }}

        /* PRINT OPTIMIZATION */
        @media print {{
            body {{ background: #ffffff !important; padding: 0 !important; }}
            .no-print {{ display: none !important; }}
            .report-page {{ display: flex !important; page-break-after: always; break-after: page; margin-bottom: 24px; }}
            .report-page:last-child {{ page-break-after: avoid; break-after: avoid; }}
            .card {{ box-shadow: none !important; border: 1px solid #d1d9e0 !important; }}
        }}
    </style>
</head>
<body>
    <div id="toast" style="position: fixed; bottom: 24px; right: 24px; background: #1f2328; color: #ffffff; padding: 10px 16px; border-radius: 8px; font-size: 13px; font-weight: 500; display: none; z-index: 1000; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"></div>

    <div class="container">
        <!-- TOP HEADER CARD -->
        <div class="card" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
            <div>
                <span style="background: rgba(130, 80, 223, 0.1); color: #8250df; padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; border: 1px solid rgba(130, 80, 223, 0.2);">
                    Multi-Model Comparison Report
                </span>
                <h1 style="font-size: 22px; font-weight: 800; color: #1f2328; margin-top: 6px;">
                    Model Comparison &amp; Parameter Evaluation
                </h1>
                <p style="font-size: 13px; color: #59636e; margin-top: 2px;">
                    Comparing <strong>{len(model_names)}</strong> models: {", ".join(model_names)} • 
                    <strong>{len(all_unique_params)}</strong> unique parameters across models • 
                    <strong>{n_subjects}</strong> subjects evaluated
                </p>
            </div>
            <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
                {random_header_pill_html}
                <div style="background: #f6f8fa; border: 1px solid #d1d9e0; border-radius: 8px; padding: 10px 16px; text-align: right;">
                    <div style="font-size: 10px; text-transform: uppercase; color: #59636e; font-weight: 600;">Best Model (Lowest BIC)</div>
                    <div style="font-size: 16px; font-weight: 700; color: #1a7f37; font-family: monospace;">
                        {model_names[sorted_indices[0]]} (BIC: {final_bic[sorted_indices[0]]:.1f})
                    </div>
                </div>
            </div>
        </div>

        <!-- MULTI-PAGE TAB NAVIGATION BAR -->
        <nav class="report-navbar no-print">
            <div class="nav-tabs">
                <button class="nav-tab active" data-comp-page="comp-page-fit" onclick="switchCompPage('comp-page-fit')">
                    <span class="tab-badge">1</span>
                    <span>Fit &amp; BICs</span>
                </button>
                <button class="nav-tab" data-comp-page="comp-page-matrix" onclick="switchCompPage('comp-page-matrix')">
                    <span class="tab-badge">2</span>
                    <span>Parameter Matrix &amp; Code</span>
                </button>
                <button class="nav-tab" data-comp-page="comp-page-evolution" onclick="switchCompPage('comp-page-evolution')">
                    <span class="tab-badge">3</span>
                    <span>Parameter Evolution Comparison</span>
                </button>
            </div>
            <div class="nav-actions">
                <button class="action-btn" onclick="window.print()">
                    🖨️ Print / Save as PDF
                </button>
            </div>
        </nav>

        <!-- ================================================================= -->
        <!-- PAGE 1: FIT & BICS + PARTICIPANT BEST-EXPLAINED BREAKDOWN         -->
        <!-- ================================================================= -->
        <div id="comp-page-fit" class="report-page active">
            <div class="card" style="padding: 12px 16px; background: #ffffff; border-left: 4px solid #0969da;">
                <div style="font-size: 13px; font-weight: 700; color: #1f2328;">Page 1: Likelihood, BIC Trajectories &amp; Subject-Level Selection</div>
                <div style="font-size: 12px; color: #59636e;">Overall population evidence trajectories, BIC minimization curves, model ranking table, and % of participants best explained by each model.</div>
            </div>

            <!-- Evolution Curves -->
            <div class="grid-2">
                <div class="card">
                    <div class="card-header">
                        <div>
                            <div class="card-title">Total Evidence Evolution Across Iterations</div>
                            <div class="card-subtitle">Log marginal likelihood summed across participants (higher is better)</div>
                        </div>
                    </div>
                    {div_ev}
                </div>

                <div class="card">
                    <div class="card-header">
                        <div>
                            <div class="card-title">BIC Evolution Across Iterations</div>
                            <div class="card-subtitle">Bayesian Information Criterion penalizing parameter count (lower is better)</div>
                        </div>
                    </div>
                    {div_bic}
                </div>
            </div>

            <!-- Final Comparison: Ranking Table & Participants Best Explained Bar -->
            <div class="grid-2">
                <div class="card">
                    <div class="card-header">
                        <div>
                            <div class="card-title">Final Model Ranking &amp; Diagnostic Statistics</div>
                            <div class="card-subtitle">Ranked by BIC; ΔBIC &gt; 10 indicates decisive evidence difference</div>
                        </div>
                    </div>
                    <div style="overflow-x: auto; border: 1px solid #eaeef2; border-radius: 8px;">
                        <table>
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>Model</th>
                                    <th>Description</th>
                                    <th>k</th>
                                    <th>Evidence</th>
                                    <th>BIC</th>
                                    <th>ΔBIC</th>
                                    {random_table_headers_html}
                                    <th>Best Explained %</th>
                                </tr>
                            </thead>
                            <tbody>
                                {comp_rows_html}
                            </tbody>
                        </table>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <div>
                            <div class="card-title">Percentage of Participants Best Explained</div>
                            <div class="card-subtitle">Determined by individual subject log-marginal likelihoods at final iteration</div>
                        </div>
                    </div>
                    {div_subj_bar}
                </div>
            </div>

            <!-- Page 1 Footer Navigation -->
            <div style="display: flex; justify-content: space-between; align-items: center; padding-top: 10px;" class="no-print">
                <div></div>
                <button class="action-btn" onclick="switchCompPage('comp-page-matrix')" style="color: #0969da; font-weight: 700; padding: 8px 16px;">
                    Next Page: Parameter Matrix &amp; Code &rarr;
                </button>
            </div>
        </div>

        <!-- ================================================================= -->
        <!-- PAGE 2: PARAMETER MATRIX & COLLAPSIBLE MODEL SOURCE CODE          -->
        <!-- ================================================================= -->
        <div id="comp-page-matrix" class="report-page">
            <div class="card" style="padding: 12px 16px; background: #ffffff; border-left: 4px solid #1a7f37;">
                <div style="font-size: 13px; font-weight: 700; color: #1f2328;">Page 2: Parameter Inclusion Matrix &amp; Model Function Source Codes</div>
                <div style="font-size: 12px; color: #59636e;">Overview table of which parameters are included in which models, with collapsible source code inspection for each model.</div>
            </div>

            <!-- MODEL SHORT DESCRIPTIONS SUMMARY -->
            <div style="display: flex; flex-wrap: wrap; gap: 12px;">
                {model_desc_cards_html}
            </div>

            <!-- PARAMETERS INCLUSION MATRIX TABLE (Y-axis = Parameters, X-axis = Models) -->
            <div class="card">
                <div class="card-header">
                    <div>
                        <div class="card-title">Parameter Inclusion Overview Matrix</div>
                        <div class="card-subtitle">Parameters on Y-axis, Models on X-axis. Filled squares represent parameters included in each model.</div>
                    </div>
                </div>
                <div style="overflow-x: auto; border: 1px solid #eaeef2; border-radius: 8px;">
                    <table>
                        <thead>
                            <tr>
                                <th style="width: 220px;">Parameter (Y-Axis)</th>
                                {matrix_header_html}
                            </tr>
                        </thead>
                        <tbody>
                            {matrix_rows_html}
                        </tbody>
                        <tfoot>
                            <tr style="background: #f6f8fa; border-top: 2px solid #d1d9e0;">
                                <td style="padding: 10px 14px; font-weight: 700;">Total Parameters (k)</td>
                                {matrix_footer_html}
                            </tr>
                        </tfoot>
                    </table>
                    <div style="padding: 8px 14px; font-size: 11px; color: #59636e; background: #f6f8fa; border-top: 1px solid #eaeef2; display: flex; align-items: center; gap: 16px; flex-wrap: wrap;">
                        <span style="display: inline-flex; align-items: center; gap: 6px;"><span style="display: inline-block; width: 14px; height: 14px; border-radius: 3px; background: #0969da; vertical-align: middle;"></span> Filled square = Included in model</span>
                        <span style="display: inline-flex; align-items: center; gap: 6px;"><span style="display: inline-flex; align-items: center; justify-content: center; width: 14px; height: 14px; border-radius: 3px; background: #0969da; vertical-align: middle;"><span style="width: 5px; height: 5px; border-radius: 50%; background: #ffffff;"></span></span> Filled square with dot = Group differences for parameter</span>
                        <span style="display: inline-flex; align-items: center; gap: 6px;"><span style="display: inline-block; width: 14px; height: 14px; border-radius: 3px; border: 1.5px dashed #d1d9e0; background: transparent; vertical-align: middle;"></span> Dashed outline = Excluded</span>
                    </div>
                </div>
            </div>

            <!-- COLLAPSIBLE & EXPANDABLE MODEL FUNCTIONS SOURCE CODE -->
            <div class="card">
                <div class="card-header">
                    <div>
                        <div class="card-title">Model Function Source Code Inspection</div>
                        <div class="card-subtitle">Select a model to view its exact Python implementation with standard IDE syntax coloring. Expand to show full code without inner scrolling, or collapse to save space.</div>
                    </div>
                    <div style="display: flex; gap: 8px; flex-wrap: wrap;" class="no-print">
                        <button class="action-btn" id="code-expand-toggle-btn" onclick="toggleCodeExpand()" style="color: #0969da; font-weight: 600;">
                            ⤢ Show All Code
                        </button>
                        <button class="action-btn" id="code-collapse-toggle-btn" onclick="toggleCodeSection()">
                            📂 Hide Code
                        </button>
                    </div>
                </div>

                <div id="code-collapsible-wrapper">
                    <!-- Model Selection Tabs -->
                    <div style="display: flex; border-bottom: 1px solid #eaeef2; background: #f6f8fa; border-top-left-radius: 8px; border-top-right-radius: 8px; overflow-x: auto;">
                        {model_code_tabs_html}
                    </div>

                    <!-- Code Display Blocks -->
                    <div style="background: #0d1117; border-bottom-left-radius: 8px; border-bottom-right-radius: 8px; overflow: hidden; border: 1px solid #30363d; border-top: none;">
                        {model_code_blocks_html}
                    </div>
                </div>
            </div>

            <!-- Page 2 Footer Navigation -->
            <div style="display: flex; justify-content: space-between; align-items: center; padding-top: 10px;" class="no-print">
                <button class="action-btn" onclick="switchCompPage('comp-page-fit')" style="padding: 8px 16px;">
                    &larr; Previous Page: Fit &amp; BICs
                </button>
                <button class="action-btn" onclick="switchCompPage('comp-page-evolution')" style="color: #8250df; font-weight: 700; padding: 8px 16px;">
                    Next Page: Parameter Evolution Comparison &rarr;
                </button>
            </div>
        </div>

        <!-- ================================================================= -->
        <!-- PAGE 3: PARAMETER EVOLUTION COMPARISON (SHARED ON SAME PLOTS)     -->
        <!-- ================================================================= -->
        <div id="comp-page-evolution" class="report-page">
            <div class="card" style="padding: 12px 16px; background: #ffffff; border-left: 4px solid #8250df;">
                <div style="font-size: 13px; font-weight: 700; color: #1f2328;">Page 3: Parameter Evolution Comparison</div>
                <div style="font-size: 12px; color: #59636e;">
                    Comparing all {len(active_params)} parameters. Shared parameters are plotted together on the same plot for direct comparison across models; unique parameters are shown for their respective model.
                </div>
            </div>

            <!-- Evolution Subplots Grid -->
            <div class="grid-3">
                {evolution_comparison_cards}
            </div>

            <!-- Page 3 Footer Navigation -->
            <div style="display: flex; justify-content: space-between; align-items: center; padding-top: 10px;" class="no-print">
                <button class="action-btn" onclick="switchCompPage('comp-page-matrix')" style="padding: 8px 16px;">
                    &larr; Previous Page: Parameter Matrix &amp; Code
                </button>
                <div></div>
            </div>
        </div>
    </div>

    <script>
        function switchCompPage(pageId) {{
            document.querySelectorAll('.report-page').forEach(function(el) {{
                el.classList.remove('active');
            }});
            document.querySelectorAll('.nav-tab').forEach(function(el) {{
                el.classList.remove('active');
            }});
            var target = document.getElementById(pageId);
            if (target) {{
                target.classList.add('active');
            }}
            var tab = document.querySelector('[data-comp-page="' + pageId + '"]');
            if (tab) {{
                tab.classList.add('active');
            }}
            window.location.hash = pageId;
            setTimeout(function() {{
                window.dispatchEvent(new Event('resize'));
            }}, 60);
            window.scrollTo({{ top: 0, behavior: 'smooth' }});
        }}

        function showModelCode(modelName) {{
            document.querySelectorAll('.code-panel').forEach(function(el) {{
                el.style.display = 'none';
                el.classList.remove('active');
            }});
            document.querySelectorAll('.code-tab').forEach(function(el) {{
                el.classList.remove('active');
                el.style.borderBottomColor = 'transparent';
            }});
            var panel = document.getElementById('code-panel-' + modelName);
            if (panel) {{
                panel.style.display = 'block';
                panel.classList.add('active');
            }}
            var tab = document.querySelector('[data-code-model="' + modelName + '"]');
            if (tab) {{
                tab.classList.add('active');
                tab.style.borderBottomColor = '#0969da';
            }}
        }}

        function copySpecificCode(modelName) {{
            var el = document.getElementById('code-content-' + modelName);
            if (!el) return;
            navigator.clipboard.writeText(el.innerText).then(function() {{
                showToast('✅ Copied ' + modelName + ' source code!');
            }}).catch(function() {{
                showToast('Failed to copy to clipboard.');
            }});
        }}

        var isCodeCollapsed = false;
        var isCodeExpanded = false;
        function toggleCodeExpand() {{
            isCodeExpanded = !isCodeExpanded;
            var pres = document.querySelectorAll('#code-collapsible-wrapper pre.code-body');
            var expandBtns = document.querySelectorAll('#code-expand-toggle-btn, .expand-code-btn');
            var wrapper = document.getElementById('code-collapsible-wrapper');
            var collapseBtn = document.getElementById('code-collapse-toggle-btn');
            if (isCodeCollapsed && isCodeExpanded) {{
                isCodeCollapsed = false;
                if (wrapper) wrapper.style.display = 'block';
                if (collapseBtn) collapseBtn.innerText = '📂 Hide Code';
            }}
            for (var i = 0; i < pres.length; i++) {{
                if (isCodeExpanded) {{
                    pres[i].style.maxHeight = 'none';
                    pres[i].style.overflowY = 'visible';
                }} else {{
                    pres[i].style.maxHeight = '380px';
                    pres[i].style.overflowY = 'auto';
                }}
            }}
            var btnText = isCodeExpanded ? '⤡ Compact Window' : '⤢ Show All Code';
            var mainBtn = document.getElementById('code-expand-toggle-btn');
            if (mainBtn) mainBtn.innerText = btnText;
            for (var b = 0; b < expandBtns.length; b++) {{
                expandBtns[b].innerText = btnText;
            }}
        }}

        function toggleCodeSection() {{
            var wrapper = document.getElementById('code-collapsible-wrapper');
            var btn = document.getElementById('code-collapse-toggle-btn');
            if (!wrapper) return;
            isCodeCollapsed = !isCodeCollapsed;
            if (isCodeCollapsed) {{
                wrapper.style.display = 'none';
                btn.innerText = '📁 Show Code';
            }} else {{
                wrapper.style.display = 'block';
                btn.innerText = '📂 Hide Code';
            }}
        }}

        function showToast(msg) {{
            var t = document.getElementById('toast');
            if (!t) return;
            t.innerText = msg;
            t.style.display = 'block';
            setTimeout(function() {{
                t.style.display = 'none';
            }}, 2500);
        }}

        window.addEventListener('DOMContentLoaded', function() {{
            var hash = window.location.hash.replace('#', '');
            if (hash && document.getElementById(hash)) {{
                switchCompPage(hash);
            }}
        }});
    </script>
</body>
</html>
"""

    dashboard = ReportDashboard(comp_html, filename=filename, figure=fig_ev)

    if filename:
        dashboard.save(filename)

    if show:
        if _is_notebook():
            try:
                import html
                from IPython.display import HTML, display
                escaped = html.escape(comp_html, quote=True)
                iframe_html = (
                    f'<iframe srcdoc="{escaped}" '
                    f'style="width: 100%; height: 860px; border: 1px solid #d1d9e0; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);" '
                    f'frameborder="0"></iframe>'
                )
                display(HTML(iframe_html))
            except Exception:
                dashboard.show()
        else:
            dashboard.show()

    return dashboard


def compare_parameters(
    samplers: Union[Sequence["Sampler"], Dict[str, "Sampler"]],
    params: Optional[Sequence[str]] = None,
    filename: Optional[str] = None,
    show: bool = True,
    renderer: Optional[str] = None,
) -> ReportDashboard:
    """Generate a dedicated parameter comparison report across multiple models."""
    if isinstance(samplers, dict):
        model_dict = samplers
    else:
        model_dict = {
            getattr(s, "model_name", None) or f"Model_{i + 1}": s
            for i, s in enumerate(samplers)
        }

    if params is None:
        param_counts: Dict[str, int] = {}
        for s in model_dict.values():
            for p in s.params:
                param_counts[p] = param_counts.get(p, 0) + 1
        params = [p for p, count in param_counts.items() if count >= 2]
        if not params:
            params = list(param_counts.keys())

    return compare_models(
        model_dict,
        filename=filename,
        show=show,
        compare_params=params,
        renderer=renderer,
    )
