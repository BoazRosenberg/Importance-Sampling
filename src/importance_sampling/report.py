"""Interactive reporting and diagnostic visualizations for importance_sampling.

Generates interactive Plotly reports and dashboards for inspecting:
1. Hyperparameter evolution across iterations (solid line for mean, shaded area for +/- 1 SD).
2. Model fit and evidence convergence (total evidence, BIC, subject-level spaghetti curves).
3. Individual subject posterior means (histograms for single parameters, scatter plots for pairs).
4. Correlation / covariance heatmap for multinormal estimation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from importance_sampling.sampler import Sampler


def create_report(
    sampler: "Sampler",
    filename: Optional[str] = None,
    show: bool = True,
    transformed: bool = True,
    renderer: Optional[str] = None,
) -> Any:
    """Generate an interactive report widget for model diagnostics and results.

    Works seamlessly with models containing any number of parameters (e.g., 2 to 10+).

    Visualizations included:
    1. Hyperparameter Evolution:
       - Separate panel for each parameter showing population mean (solid line) and +/- 1 SD (shaded ribbon).
    2. Model Fit & Convergence:
       - Total Evidence (log marginal likelihood across all subjects) and BIC trajectories.
       - Subject-level evidence spaghetti plot across iterations.
    3. Individual Subject Posterior Means:
       - Histograms showing the distribution of subject posterior means across the cohort.
       - 2D Scatter plot for parameter pairs showing individual subject coordinates.
    4. Multinormal Correlation / Covariance Matrix:
       - Interactive heatmap showing parameter correlations in latent normal space.

    Parameters
    ----------
    sampler : Sampler
        A fitted Sampler instance.
    filename : Optional[str], default=None
        If provided (e.g. 'model_report.html'), exports a self-contained, interactive
        HTML report that can be opened in any web browser without a Python runtime.
    show : bool, default=True
        Whether to display the interactive figure in the current environment
        (e.g., Jupyter notebook, Google Colab, or browser).
    transformed : bool, default=True
        If True, displays hyperparameters transformed into their valid domain bounds.
        If False, displays parameters in latent standard normal space.
    renderer : Optional[str], default=None
        Plotly renderer to use when displaying the figure (e.g., 'browser', 'notebook', 'colab').

    Returns
    -------
    plotly.graph_objects.Figure
        The interactive Plotly Figure object containing the multi-panel report.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for interactive reports. Install it via: pip install plotly"
        ) from exc

    params = sampler.params
    n_params = len(params)
    iters = list(range(len(sampler.hyper_params_list)))
    fit_iters = list(range(len(sampler.evidence)))

    # Compute grid for hyperparameter evolution (2 columns)
    hp_cols = 2 if n_params > 1 else 1
    hp_rows = math.ceil(n_params / hp_cols)

    # Layout structure:
    # Rows 1..hp_rows: Hyperparameter evolution grid (each parameter has its own subplot)
    # Next Row: Model Fit (Col 1: Evidence & BIC, Col 2: Subject Spaghetti Curves)
    # Next Row: Individual Subject Means (Col 1: Subject Means Histogram, Col 2: Subject 2D Scatter)
    # Next Row: Covariance / Correlation Matrix Heatmap
    fit_row = hp_rows + 1
    subj_row = hp_rows + 2
    corr_row = hp_rows + 3
    total_rows = corr_row

    subplot_titles = []
    # 1. Hyperparameter evolution titles (using raw parameter key names)
    space_label = "Transformed" if transformed else "Latent Space"
    for r in range(hp_rows):
        for c in range(hp_cols):
            p_idx = r * hp_cols + c
            if p_idx < n_params:
                subplot_titles.append(f"Evolution: {params[p_idx]} ({space_label})")
            else:
                subplot_titles.append("")

    # 2. Model Fit titles
    subplot_titles.append("Model Fit: Total Evidence & BIC")
    subplot_titles.append("Subject-Level Evidence Trajectories")

    # 3. Subject Means titles
    subplot_titles.append(f"Subject Distribution: {params[0]}")
    if n_params >= 2:
        subplot_titles.append(f"Subject Scatter: {params[0]} vs {params[1]}")
    else:
        subplot_titles.append(f"Subject Histogram (Duplicate)")

    # 4. Correlation / Covariance Matrix title
    subplot_titles.append("Multinormal Correlation / Covariance Matrix")
    subplot_titles.append("")

    # Build subplot specifications
    specs = []
    for _ in range(hp_rows):
        if hp_cols == 2:
            specs.append([{}, {}])
        else:
            specs.append([{"colspan": 2}, None])

    specs.append([{"secondary_y": True}, {}])
    specs.append([{}, {}])
    specs.append([{"colspan": 2}, None])

    fig = make_subplots(
        rows=total_rows,
        cols=2,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
    )

    palette = ["#0969da", "#1a7f37", "#8250df", "#cf222e", "#bf8700", "#0550ae", "#116329", "#5a32a3", "#82071e", "#7d4e00"]

    # -------------------------------------------------------------------------
    # 1. Hyperparameter Evolution (Mean line + Shaded +/- 1 SD)
    # -------------------------------------------------------------------------
    for idx, param in enumerate(params):
        r = (idx // hp_cols) + 1
        c = (idx % hp_cols) + 1
        color = palette[idx % len(palette)]
        t_func = sampler.transformations[param] if transformed else (lambda x: x)

        means_raw = np.array([hp[param]["mean"] for hp in sampler.hyper_params_list])
        sds_raw = np.array([hp[param]["sd"] for hp in sampler.hyper_params_list])

        means_plot = t_func(means_raw)
        upper_plot = t_func(means_raw + sds_raw)
        lower_plot = t_func(means_raw - sds_raw)

        # Upper bound (invisible line for fill anchor)
        fig.add_trace(
            go.Scatter(
                x=iters,
                y=upper_plot,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                name=f"{param} +1 SD",
                hoverinfo="skip",
            ),
            row=r,
            col=c,
        )

        # Lower bound with shaded fill to upper bound
        fill_rgba = "rgba(9, 105, 218, 0.16)"
        if idx % 3 == 1:
            fill_rgba = "rgba(26, 127, 55, 0.16)"
        elif idx % 3 == 2:
            fill_rgba = "rgba(130, 80, 223, 0.16)"

        fig.add_trace(
            go.Scatter(
                x=iters,
                y=lower_plot,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=fill_rgba,
                name=f"{param} ±1 SD",
                hovertemplate=f"<b>{param} -1 SD</b>: %{{y:.4f}}<extra></extra>",
            ),
            row=r,
            col=c,
        )

        # Mean line
        fig.add_trace(
            go.Scatter(
                x=iters,
                y=means_plot,
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=4.5, color=color),
                name=f"{param} Mean",
                hovertemplate=f"Iteration %{{x}}<br><b>{param} Mean</b>: %{{y:.4f}}<extra></extra>",
            ),
            row=r,
            col=c,
        )

        fig.update_xaxes(title_text="Iteration", row=r, col=c, gridcolor="#eaeef2")
        fig.update_yaxes(title_text=param, row=r, col=c, gridcolor="#eaeef2")

    # -------------------------------------------------------------------------
    # 2. Model Fit: Total Evidence & BIC
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=fit_iters,
            y=sampler.evidence,
            mode="lines+markers",
            line=dict(color="#0969da", width=2.5),
            marker=dict(size=5.5),
            name="Total Evidence (LL)",
            hovertemplate="Iteration %{x}<br><b>Evidence</b>: %{y:.2f}<extra></extra>",
        ),
        row=fit_row,
        col=1,
        secondary_y=False,
    )

    if sampler.BIC:
        fig.add_trace(
            go.Scatter(
                x=fit_iters,
                y=sampler.BIC,
                mode="lines+markers",
                line=dict(color="#cf222e", width=2, dash="dash"),
                marker=dict(size=5),
                name="BIC (lower is better)",
                hovertemplate="Iteration %{x}<br><b>BIC</b>: %{y:.2f}<extra></extra>",
            ),
            row=fit_row,
            col=1,
            secondary_y=True,
        )

    fig.update_xaxes(title_text="Iteration", row=fit_row, col=1, gridcolor="#eaeef2")
    fig.update_yaxes(title_text="Total Evidence (Log Likelihood)", row=fit_row, col=1, secondary_y=False, gridcolor="#eaeef2")
    fig.update_yaxes(title_text="BIC", row=fit_row, col=1, secondary_y=True, gridcolor="#eaeef2")

    # Subject-level evidence spaghetti plot
    if sampler.subj_evidence:
        subj_matrix = np.array(sampler.subj_evidence)  # (n_iters, n_subjs)
        for s in range(sampler.n_subjects):
            fig.add_trace(
                go.Scatter(
                    x=fit_iters,
                    y=subj_matrix[:, s],
                    mode="lines",
                    line=dict(width=1, color="rgba(89, 99, 110, 0.35)"),
                    showlegend=False,
                    name=f"Subj {s}",
                    hovertemplate=f"Subj {s}<br>Iteration %{{x}}: %{{y:.2f}}<extra></extra>",
                ),
                row=fit_row,
                col=2,
            )

        mean_subj_ev = np.mean(subj_matrix, axis=1)
        fig.add_trace(
            go.Scatter(
                x=fit_iters,
                y=mean_subj_ev,
                mode="lines+markers",
                line=dict(color="#1f2328", width=2.5),
                marker=dict(size=5),
                name="Mean Subj Evidence",
                hovertemplate="Iteration %{x}<br><b>Mean Subj Evidence</b>: %{y:.2f}<extra></extra>",
            ),
            row=fit_row,
            col=2,
        )

    fig.update_xaxes(title_text="Iteration", row=fit_row, col=2, gridcolor="#eaeef2")
    fig.update_yaxes(title_text="Log Evidence per Subject", row=fit_row, col=2, gridcolor="#eaeef2")

    # -------------------------------------------------------------------------
    # 3. Individual Subject Posterior Means: Histogram & Scatter (No Raw Numbers)
    # -------------------------------------------------------------------------
    if sampler.mean_params:
        p0 = params[0]
        t0 = sampler.transformations[p0] if transformed else (lambda x: x)
        vals0 = [t0(v) for v in sampler.mean_params[p0]]

        # Histogram of subject means for parameter 0
        fig.add_trace(
            go.Histogram(
                x=vals0,
                nbinsx=15,
                marker=dict(color="#0969da", opacity=0.75, line=dict(color="#1f2328", width=1)),
                name=f"{p0} Subject Distribution",
                hovertemplate=f"<b>{p0} Bin</b>: %{{x}}<br>Count: %{{y}}<extra></extra>",
            ),
            row=subj_row,
            col=1,
        )
        fig.update_xaxes(title_text=p0, row=subj_row, col=1, gridcolor="#eaeef2")
        fig.update_yaxes(title_text="Subject Count", row=subj_row, col=1, gridcolor="#eaeef2")

        # Scatter plot for first two parameters (if >= 2)
        if n_params >= 2:
            p1 = params[1]
            t1 = sampler.transformations[p1] if transformed else (lambda x: x)
            vals1 = [t1(v) for v in sampler.mean_params[p1]]

            fig.add_trace(
                go.Scatter(
                    x=vals0,
                    y=vals1,
                    mode="markers+text",
                    text=[f"S{s}" for s in range(sampler.n_subjects)],
                    textposition="top center",
                    marker=dict(size=9, color="#8250df", line=dict(color="#1f2328", width=1)),
                    name=f"{p0} vs {p1} Subjects",
                    hovertemplate=f"<b>Subject %{{text}}</b><br>{p0}: %{{x:.4f}}<br>{p1}: %{{y:.4f}}<extra></extra>",
                ),
                row=subj_row,
                col=2,
            )
            fig.update_xaxes(title_text=p0, row=subj_row, col=2, gridcolor="#eaeef2")
            fig.update_yaxes(title_text=p1, row=subj_row, col=2, gridcolor="#eaeef2")

    # -------------------------------------------------------------------------
    # 4. Multinormal Correlation / Covariance Matrix Heatmap
    # -------------------------------------------------------------------------
    cor_matrix = sampler.cor_matrix
    text_annotations = [[f"{val:.2f}" for val in row] for row in cor_matrix]

    fig.add_trace(
        go.Heatmap(
            z=cor_matrix,
            x=params,
            y=params,
            colorscale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            text=text_annotations,
            texttemplate="%{text}",
            textfont=dict(size=12, color="#1f2328"),
            colorbar=dict(
                title="Correlation (r)",
                len=0.25,
                y=0.10,
                yanchor="bottom",
            ),
            hovertemplate="Param X: %{x}<br>Param Y: %{y}<br><b>Correlation</b>: %{z:.3f}<extra></extra>",
            name="Correlation Matrix",
        ),
        row=corr_row,
        col=1,
    )
    fig.update_xaxes(title_text="Parameter", row=corr_row, col=1, gridcolor="#eaeef2")
    fig.update_yaxes(title_text="Parameter", row=corr_row, col=1, gridcolor="#eaeef2")

    # -------------------------------------------------------------------------
    # Overall Layout & Sizing
    # -------------------------------------------------------------------------
    total_height = max(1100, hp_rows * 260 + 320 + 320 + 360 + 100)
    fig.update_layout(
        title=dict(
            text=f"<b>importance_sampling Diagnostic Report</b>: {sampler.model_name}<br>"
            f"<sup>Subjects: {sampler.n_subjects} | Parameters: {n_params} | Iterations: {sampler.iterations} | "
            f"Evidence: {round(sampler.evidence[-1], 2) if sampler.evidence else 'N/A'} | "
            f"BIC: {round(sampler.BIC[-1], 2) if sampler.BIC else 'N/A'}</sup>",
            x=0.03,
            y=0.99,
            xanchor="left",
            yanchor="top",
            font=dict(size=18, color="#1f2328", family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
        ),
        height=total_height,
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.015,
            xanchor="right",
            x=0.98,
            font=dict(size=10.5),
        ),
        margin=dict(l=60, r=40, t=110, b=60),
    )

    if filename:
        fig.write_html(filename, include_plotlyjs="cdn", full_html=True)

    if show:
        if renderer:
            fig.show(renderer=renderer)
        else:
            fig.show()

    return fig
