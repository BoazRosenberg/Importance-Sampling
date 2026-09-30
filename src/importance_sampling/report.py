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


def create_report(
    sampler: "Sampler",
    filename: Optional[str] = None,
    show: bool = True,
    transformed: bool = True,
    params_to_plot: Optional[Sequence[str]] = None,
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
    # 3. PLOTS: Hyperparameter Evolution (One Clean Plot per Parameter)
    # -------------------------------------------------------------------------
    evolution_divs: List[Dict[str, Any]] = []
    space_label = "Transformed Space" if transformed else "Latent Normal Space"

    for idx, param in enumerate(active_params):
        color = palette[idx % len(palette)]
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
                fillcolor="rgba(9, 105, 218, 0.15)",
                showlegend=False,
                name="±1 SD",
                hovertemplate="Iter %{x}<br><b>-1 SD</b>: %{y:.4f}<extra></extra>",
            )
        )

        # Grand Mean line
        fig_p.add_trace(
            go.Scatter(
                x=iters,
                y=means_plot,
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=4, color=color),
                showlegend=has_group_diff,
                name="Grand Mean" if has_group_diff else f"{param} Mean",
                hovertemplate=f"Iter %{{x}}<br><b>Grand Mean</b>: %{{y:.4f}}<extra></extra>",
            )
        )

        # Plot individual group means if group differences enabled
        if has_group_diff:
            g_colors = ["#0969da", "#8250df", "#cf222e", "#bf8700"]
            for g_i, g in enumerate(sampler.group_names[param]):
                g_col = g_colors[g_i % len(g_colors)]
                g_means_hist = [
                    hp[param].get("group_means", {}).get(g, hp[param]["mean"])
                    for hp in sampler.hyper_params_list
                ]
                fig_p.add_trace(
                    go.Scatter(
                        x=iters,
                        y=[t_func(m) for m in g_means_hist],
                        mode="lines+markers",
                        line=dict(color=g_col, width=2, dash="dash" if g != sampler.ref_group[param] else "solid"),
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
    # Assemble Clean Standalone HTML Document
    # -------------------------------------------------------------------------
    final_evidence_str = f"{sampler.evidence[-1]:.2f}" if sampler.evidence else "N/A"
    final_bic_str = f"{sampler.BIC[-1]:.2f}" if sampler.BIC else "N/A"

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
            max-width: 1140px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 24px;
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
    </style>
</head>
<body>
    <div class="container">
        <!-- 1. HEADER HERO CARD -->
        <div class="card" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
            <div>
                <div style="margin-bottom: 6px;">
                    <span class="badge">Model Diagnostics Report</span>
                </div>
                <h1 style="font-size: 22px; font-weight: 800; color: #1f2328;">{sampler.model_name}</h1>
                <p style="font-size: 13px; color: #59636e; margin-top: 2px;">
                    <strong>{sampler.n_subjects}</strong> Subjects • 
                    <strong>{len(all_params)}</strong> Parameters • 
                    <strong>{sampler.iterations}</strong> Iterations • 
                    Fit Time: {sampler.total_fit_time:.2f}s
                </p>
            </div>
            <div style="display: flex; gap: 12px; font-family: monospace;">
                <div style="background: #f6f8fa; border: 1px solid #d1d9e0; border-radius: 8px; padding: 10px 16px; text-align: center;">
                    <div style="font-size: 10px; text-transform: uppercase; color: #59636e; font-weight: 600;">Total Evidence</div>
                    <div style="font-size: 18px; font-weight: 700; color: #0969da;">{final_evidence_str}</div>
                </div>
                <div style="background: #f6f8fa; border: 1px solid #d1d9e0; border-radius: 8px; padding: 10px 16px; text-align: center;">
                    <div style="font-size: 10px; text-transform: uppercase; color: #59636e; font-weight: 600;">Final BIC</div>
                    <div style="font-size: 18px; font-weight: 700; color: #1a7f37;">{final_bic_str}</div>
                </div>
            </div>
        </div>

        <!-- 2. MODEL FIT: EVIDENCE & BIC + SUBJECT SPAGHETTI -->
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

        <!-- 3. PARAMETERS SUMMARY TABLE -->
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

        <!-- 4. HYPERPARAMETER EVOLUTION GRID (3 PER ROW) -->
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

        <!-- 5. INDIVIDUAL SUBJECT POSTERIOR MEANS (3 PER ROW) -->
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

        <!-- 6. MULTINORMAL CORRELATION MATRIX HEATMAP -->
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
    </div>
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
    renderer: Optional[str] = None,
) -> ReportDashboard:
    """Compare multiple computational models on likelihood (evidence), BIC, and parameters.

    Visualizes:
    1. Evolution Trajectories: Total Evidence and BIC across iterations for all models.
    2. Final Fit Comparison: Side-by-side grouped bar chart and ranking table with ΔBIC.
    3. Parameter Comparison: Dedicated visual comparison across models.

    Parameters
    ----------
    samplers : Union[Sequence[Sampler], Dict[str, Sampler]]
        Two or more fitted Sampler instances.
    filename : Optional[str], default=None
        If provided, saves as a self-contained HTML file.
    show : bool, default=True
        Whether to display the interactive figure or open in browser.
    compare_params : Optional[Sequence[str]], default=None
        List of parameter keys to compare across models.
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

    plotly_config = {
        "responsive": True,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "displaylogo": False,
    }

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

    # 1. Total Evidence Evolution Plot
    fig_ev = go.Figure()
    for idx, name in enumerate(model_names):
        m = model_dict[name]
        fig_ev.add_trace(
            go.Scatter(
                x=list(range(len(m.evidence))),
                y=m.evidence,
                mode="lines+markers",
                name=name,
                line=dict(color=palette[idx % len(palette)], width=2.5),
                marker=dict(size=4),
                hovertemplate=f"<b>{name}</b><br>Iter %{{x}}: Ev = %{{y:.2f}}<extra></extra>",
            )
        )
    fig_ev.update_layout(
        template="plotly_white",
        height=280,
        margin=dict(l=50, r=20, t=20, b=40),
        xaxis=dict(title="Iteration", gridcolor="#eaeef2"),
        yaxis=dict(title="Total Evidence", gridcolor="#eaeef2"),
        legend=dict(orientation="h", y=1.08, x=0.02),
        hovermode="x unified",
    )
    div_ev = to_html(fig_ev, include_plotlyjs=False, full_html=False, config=plotly_config)

    # 2. BIC Evolution Plot
    fig_bic = go.Figure()
    for idx, name in enumerate(model_names):
        m = model_dict[name]
        if m.BIC:
            fig_bic.add_trace(
                go.Scatter(
                    x=list(range(len(m.BIC))),
                    y=m.BIC,
                    mode="lines+markers",
                    name=name,
                    line=dict(color=palette[idx % len(palette)], width=2.5, dash="dash"),
                    marker=dict(size=4),
                    hovertemplate=f"<b>{name}</b><br>Iter %{{x}}: BIC = %{{y:.2f}}<extra></extra>",
                )
            )
    fig_bic.update_layout(
        template="plotly_white",
        height=280,
        margin=dict(l=50, r=20, t=20, b=40),
        xaxis=dict(title="Iteration", gridcolor="#eaeef2"),
        yaxis=dict(title="BIC (Lower is Better)", gridcolor="#eaeef2"),
        legend=dict(orientation="h", y=1.08, x=0.02),
        hovermode="x unified",
    )
    div_bic = to_html(fig_bic, include_plotlyjs=False, full_html=False, config=plotly_config)

    # 3. Final BIC Bar Chart
    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            x=model_names,
            y=final_bic,
            name="Final BIC",
            marker=dict(color="#cf222e", opacity=0.85),
            hovertemplate="<b>%{x}</b><br>BIC: %{y:.2f}<extra></extra>",
        )
    )
    fig_bar.update_layout(
        template="plotly_white",
        height=280,
        margin=dict(l=50, r=20, t=20, b=40),
        xaxis=dict(title="Model", gridcolor="#eaeef2"),
        yaxis=dict(title="Final BIC", gridcolor="#eaeef2"),
    )
    div_bar = to_html(fig_bar, include_plotlyjs=False, full_html=False, config=plotly_config)

    # Comparison Table HTML
    comp_rows_html = ""
    for idx in sorted_indices:
        m_name = model_names[idx]
        is_winner = ranks[idx] == 1
        d_bic = delta_bic[idx]
        d_bic_str = "0.0 (Best)" if is_winner else f"+{d_bic:.2f}"
        badge_style = "background: #1a7f37; color: white; padding: 2px 6px; border-radius: 4px; font-weight: 700;" if is_winner else "color: #59636e;"

        comp_rows_html += f"""
        <tr style="border-bottom: 1px solid #eaeef2; {'background: rgba(26, 127, 55, 0.04);' if is_winner else ''}">
            <td style="padding: 10px 14px;"><span style="{badge_style}">#{ranks[idx]}</span></td>
            <td style="padding: 10px 14px; font-weight: 700; font-family: monospace;">{m_name}</td>
            <td style="padding: 10px 14px; font-family: monospace; color: #59636e;">{n_params_list[idx]}</td>
            <td style="padding: 10px 14px; font-family: monospace; color: #0969da;">{final_evidence[idx]:.2f}</td>
            <td style="padding: 10px 14px; font-family: monospace; font-weight: 700;">{final_bic[idx]:.2f}</td>
            <td style="padding: 10px 14px; font-family: monospace; color: {'#1a7f37' if is_winner else '#cf222e'}; font-weight: 600;">{d_bic_str}</td>
        </tr>
        """

    # Optional Parameter Comparisons HTML
    param_cards_html = ""
    if compare_params:
        for p in compare_params:
            valid_models = [name for name in model_names if p in model_dict[name].params]
            p_means = [model_dict[name].hyper_params[p]["mean"] for name in valid_models]
            p_sds = [model_dict[name].hyper_params[p]["sd"] for name in valid_models]

            fig_param = go.Figure()
            fig_param.add_trace(
                go.Scatter(
                    x=valid_models,
                    y=p_means,
                    error_y=dict(type="data", array=p_sds, visible=True),
                    mode="markers+text",
                    text=[f"{m:.3f}" for m in p_means],
                    textposition="top center",
                    marker=dict(size=10, color="#8250df"),
                    name="Mean ± 1 SD",
                    hovertemplate="<b>%{x}</b><br>Mean: %{y:.3f}<extra></extra>",
                )
            )
            fig_param.update_layout(
                template="plotly_white",
                height=240,
                margin=dict(l=45, r=15, t=20, b=40),
                xaxis=dict(title="Model", gridcolor="#eaeef2"),
                yaxis=dict(title=f"{p} Value", gridcolor="#eaeef2"),
            )
            div_param = to_html(fig_param, include_plotlyjs=False, full_html=False, config=plotly_config)
            param_cards_html += f"""
            <div style="background: #ffffff; border: 1px solid #eaeef2; border-radius: 8px; padding: 12px;">
                <div style="font-weight: 700; font-family: monospace; margin-bottom: 4px; font-size: 13px;">Parameter: {p}</div>
                {div_param}
            </div>
            """

    comp_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison Report</title>
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
            max-width: 1140px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 24px;
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
        @media (max-width: 768px) {{
            .grid-2 {{ grid-template-columns: 1fr; }}
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
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="card" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
            <div>
                <span style="background: rgba(130, 80, 223, 0.1); color: #8250df; padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; border: 1px solid rgba(130, 80, 223, 0.2);">
                    Multi-Model Comparison
                </span>
                <h1 style="font-size: 22px; font-weight: 800; color: #1f2328; margin-top: 6px;">Comparative Model Selection</h1>
                <p style="font-size: 13px; color: #59636e;">Comparing {len(model_names)} models: {", ".join(model_names)}</p>
            </div>
            <div style="background: #f6f8fa; border: 1px solid #d1d9e0; border-radius: 8px; padding: 10px 16px; text-align: right;">
                <div style="font-size: 10px; text-transform: uppercase; color: #59636e; font-weight: 600;">Best Model (Lowest BIC)</div>
                <div style="font-size: 16px; font-weight: 700; color: #1a7f37; font-family: monospace;">
                    {model_names[sorted_indices[0]]} (BIC: {final_bic[sorted_indices[0]]:.1f})
                </div>
            </div>
        </div>

        <!-- Evolution Curves -->
        <div class="grid-2">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Total Evidence Evolution Across Iterations</div>
                    <div class="card-subtitle">Log marginal likelihood (higher is better)</div>
                </div>
                {div_ev}
            </div>

            <div class="card">
                <div class="card-header">
                    <div class="card-title">BIC Evolution Across Iterations</div>
                    <div class="card-subtitle">Bayesian Information Criterion (lower is better)</div>
                </div>
                {div_bic}
            </div>
        </div>

        <!-- Final Comparison: Bar Plot & Table -->
        <div class="grid-2">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Final BIC Comparison Bar Chart</div>
                    <div class="card-subtitle">Comparing model complexity and goodness-of-fit</div>
                </div>
                {div_bar}
            </div>

            <div class="card">
                <div class="card-header">
                    <div class="card-title">Final Model Ranking &amp; Statistics</div>
                    <div class="card-subtitle">Ranked by BIC; ΔBIC &gt; 10 indicates decisive evidence</div>
                </div>
                <div style="overflow-x: auto; border: 1px solid #eaeef2; border-radius: 8px;">
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Model</th>
                                <th>k</th>
                                <th>Evidence</th>
                                <th>BIC</th>
                                <th>ΔBIC</th>
                            </tr>
                        </thead>
                        <tbody>
                            {comp_rows_html}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        {f'''
        <!-- Parameter Comparisons -->
        <div class="card">
            <div class="card-header">
                <div class="card-title">Parameter Comparison Across Models</div>
                <div class="card-subtitle">Population mean ± 1 SD for shared parameters</div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px;">
                {param_cards_html}
            </div>
        </div>
        ''' if compare_params else ''}
    </div>
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
