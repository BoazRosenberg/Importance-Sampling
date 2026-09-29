"""Interactive reporting and diagnostic visualizations for importance_sampling.

Generates interactive Plotly reports and dashboards for inspecting:
1. Model Fit & Convergence (Total Evidence, BIC, and subject-level evidence trajectories).
2. Hyperparameter Evolution across iterations (3-column grid, solid mean line, shaded +/- 1 SD area).
3. Individual Subject Posterior Means (separate scatter plots with X = parameter mean, Y = subject number).
4. Multinormal Correlation / Covariance Matrix Heatmap.
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
    params_to_plot: Optional[Sequence[str]] = None,
    renderer: Optional[str] = None,
) -> Any:
    """Generate an interactive report widget for model diagnostics and results.

    Layout Order:
    1. Model Fit & Convergence: Total Evidence & BIC, plus Subject-Level Spaghetti Plot.
    2. Hyperparameter Evolution Grid: 3 plots per row, each showing mean line and shaded +/- 1 SD.
    3. Individual Subject Posterior Means: Separate scatter plot per parameter (X = parameter value, Y = subject number).
    4. Multinormal Parameter Correlation / Covariance Heatmap.

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
    params_to_plot : Optional[Sequence[str]], default=None
        Subset of parameter keys to visualize in evolution and subject means. If None, plots all parameters.
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

    all_params = sampler.params
    active_params = [p for p in params_to_plot if p in all_params] if params_to_plot else all_params
    n_active = len(active_params)
    n_all = len(all_params)

    iters = list(range(len(sampler.hyper_params_list)))
    fit_iters = list(range(len(sampler.evidence)))

    # Grid calculations (3 columns for parameter plots)
    hp_cols = min(3, max(1, n_active))
    hp_rows = math.ceil(n_active / hp_cols)

    subj_cols = min(3, max(1, n_active))
    subj_rows = math.ceil(n_active / subj_cols)

    # Total rows:
    # Row 1: Model Fit (2 columns: Evidence & BIC, and Subject Spaghetti)
    # Rows 2 .. (1 + hp_rows): Hyperparameter Evolution Grid (3 columns)
    # Rows (2 + hp_rows) .. (1 + hp_rows + subj_rows): Subject Means Grid (3 columns)
    # Final Row: Correlation / Covariance Heatmap (spanning all 3 columns)
    total_rows = 1 + hp_rows + subj_rows + 1

    subplot_titles: List[str] = []

    # Row 1: Model Fit titles
    subplot_titles.extend(["Model Fit: Total Evidence & BIC", "Subject-Level Evidence Trajectories", ""])

    # Evolution titles (3 per row)
    space_label = "Transformed" if transformed else "Latent Space"
    for r in range(hp_rows):
        for c in range(3):
            p_idx = r * hp_cols + c
            if c < hp_cols and p_idx < n_active:
                subplot_titles.append(f"Evolution: {active_params[p_idx]} ({space_label})")
            else:
                subplot_titles.append("")

    # Subject Means titles (3 per row)
    for r in range(subj_rows):
        for c in range(3):
            p_idx = r * subj_cols + c
            if c < subj_cols and p_idx < n_active:
                subplot_titles.append(f"Subject Means: {active_params[p_idx]}")
            else:
                subplot_titles.append("")

    # Correlation Matrix title
    subplot_titles.extend(["Multinormal Correlation Matrix (Latent Space)", "", ""])

    # Build specs for make_subplots (3 columns wide)
    specs: List[List[Dict[str, Any]]] = []

    # Row 1: Evidence & BIC (Col 1-2), Spaghetti (Col 3)
    specs.append([{"colspan": 2, "secondary_y": True}, None, {}])

    # Evolution Rows
    for _ in range(hp_rows):
        specs.append([{}, {}, {}])

    # Subject Means Rows
    for _ in range(subj_rows):
        specs.append([{}, {}, {}])

    # Correlation Heatmap Row (Col 1-3)
    specs.append([{"colspan": 3}, None, None])

    fig = make_subplots(
        rows=total_rows,
        cols=3,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.045,
        horizontal_spacing=0.065,
    )

    palette = ["#0969da", "#1a7f37", "#8250df", "#cf222e", "#bf8700", "#0550ae", "#116329", "#5a32a3", "#82071e", "#7d4e00"]

    # -------------------------------------------------------------------------
    # 1. MODEL FIT (Row 1)
    # -------------------------------------------------------------------------
    # Col 1: Total Evidence & BIC
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
        row=1,
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
                name="BIC",
                hovertemplate="Iteration %{x}<br><b>BIC</b>: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

    fig.update_xaxes(title_text="Iteration", row=1, col=1, gridcolor="#eaeef2")
    fig.update_yaxes(title_text="Total Evidence", row=1, col=1, secondary_y=False, gridcolor="#eaeef2")
    fig.update_yaxes(title_text="BIC", row=1, col=1, secondary_y=True, gridcolor="#eaeef2")

    # Col 3: Subject Spaghetti Curves
    if sampler.subj_evidence:
        subj_matrix = np.array(sampler.subj_evidence)
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
                row=1,
                col=3,
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
            row=1,
            col=3,
        )

    fig.update_xaxes(title_text="Iteration", row=1, col=3, gridcolor="#eaeef2")
    fig.update_yaxes(title_text="Subj Evidence", row=1, col=3, gridcolor="#eaeef2")

    # -------------------------------------------------------------------------
    # 2. HYPERPARAMETER EVOLUTION (Rows 2 .. 1 + hp_rows, 3 columns)
    # -------------------------------------------------------------------------
    for idx, param in enumerate(active_params):
        r = 1 + (idx // hp_cols) + 1
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

        fig.add_trace(
            go.Scatter(
                x=iters,
                y=lower_plot,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(9, 105, 218, 0.16)",
                name=f"{param} ±1 SD",
                showlegend=False,
                hovertemplate=f"<b>{param} -1 SD</b>: %{{y:.4f}}<extra></extra>",
            ),
            row=r,
            col=c,
        )

        fig.add_trace(
            go.Scatter(
                x=iters,
                y=means_plot,
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=4.5, color=color),
                name=f"{param} Mean",
                showlegend=False,
                hovertemplate=f"Iteration %{{x}}<br><b>{param} Mean</b>: %{{y:.4f}}<extra></extra>",
            ),
            row=r,
            col=c,
        )

        fig.update_xaxes(title_text="Iteration", row=r, col=c, gridcolor="#eaeef2")
        fig.update_yaxes(title_text=param, row=r, col=c, gridcolor="#eaeef2")

    # -------------------------------------------------------------------------
    # 3. INDIVIDUAL SUBJECT POSTERIOR MEANS (Rows 2 + hp_rows .. , 3 columns)
    # X-axis = parameter value, Y-axis = Subject index/number
    # -------------------------------------------------------------------------
    subj_start_row = 1 + hp_rows + 1
    subj_indices = list(range(sampler.n_subjects))

    if sampler.mean_params:
        for idx, param in enumerate(active_params):
            r = subj_start_row + (idx // subj_cols)
            c = (idx % subj_cols) + 1
            color = palette[idx % len(palette)]
            t_func = sampler.transformations[param] if transformed else (lambda x: x)
            vals = [t_func(v) for v in sampler.mean_params[param]]

            fig.add_trace(
                go.Scatter(
                    x=vals,
                    y=subj_indices,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=color,
                        line=dict(color="#1f2328", width=1),
                        opacity=0.85,
                    ),
                    name=f"Subj {param}",
                    showlegend=False,
                    hovertemplate=f"<b>Subject %{{y}}</b><br>{param}: %{{x:.4f}}<extra></extra>",
                ),
                row=r,
                col=c,
            )

            fig.update_xaxes(title_text=f"{param} Mean", row=r, col=c, gridcolor="#eaeef2")
            fig.update_yaxes(
                title_text="Subject #",
                row=r,
                col=c,
                gridcolor="#eaeef2",
                tickmode="linear",
                dtick=max(1, sampler.n_subjects // 8),
            )

    # -------------------------------------------------------------------------
    # 4. MULTINORMAL CORRELATION HEATMAP (Final Row, Col 1-3)
    # -------------------------------------------------------------------------
    corr_row = total_rows
    cor_matrix = sampler.cor_matrix
    text_annotations = [[f"{val:.2f}" for val in row] for row in cor_matrix]

    fig.add_trace(
        go.Heatmap(
            z=cor_matrix,
            x=all_params,
            y=all_params,
            colorscale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            text=text_annotations,
            texttemplate="%{text}",
            textfont=dict(size=12, color="#1f2328"),
            colorbar=dict(
                title="Correlation",
                len=0.2,
                y=0.08,
                yanchor="bottom",
            ),
            hovertemplate="X: %{x}<br>Y: %{y}<br><b>r</b>: %{z:.3f}<extra></extra>",
            name="Correlation Matrix",
            showlegend=False,
        ),
        row=corr_row,
        col=1,
    )
    fig.update_xaxes(title_text="Parameter", row=corr_row, col=1, gridcolor="#eaeef2")
    fig.update_yaxes(title_text="Parameter", row=corr_row, col=1, gridcolor="#eaeef2")

    # -------------------------------------------------------------------------
    # Overall Layout & Height
    # -------------------------------------------------------------------------
    total_height = 300 + hp_rows * 240 + subj_rows * 240 + 380
    fig.update_layout(
        title=dict(
            text=f"<b>importance_sampling Diagnostic Report</b>: {sampler.model_name}<br>"
            f"<sup>Subjects: {sampler.n_subjects} | Parameters: {n_all} | Iterations: {sampler.iterations} | "
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


# -----------------------------------------------------------------------------
# Multi-Model Comparison Report
# -----------------------------------------------------------------------------


def compare_models(
    samplers: Union[Sequence["Sampler"], Dict[str, "Sampler"]],
    filename: Optional[str] = None,
    show: bool = True,
    compare_params: Optional[Sequence[str]] = None,
    renderer: Optional[str] = None,
) -> Any:
    """Compare multiple computational models on likelihood (evidence), BIC, and parameters.

    Visualizes:
    1. Evolution Trajectories:
       - Total Evidence (log likelihood) across iterations for all models.
       - BIC across iterations for all models.
    2. Final Fit Comparison:
       - Grouped bar plot comparing Final Total Evidence and Final BIC.
       - Complete comparative table with:
         * Model Name
         * Number of parameters (k)
         * Final Total Evidence
         * Final BIC
         * ΔBIC (difference relative to the best model, lower is better)
         * Best Model Rank
    3. Parameter Comparison (Optional):
       - If `compare_params` is specified (e.g. ['lr', 'beta']), includes subplots
         comparing population mean ± 1 SD and subject posterior distributions
         across the models sharing those parameters.

    Parameters
    ----------
    samplers : Union[Sequence[Sampler], Dict[str, Sampler]]
        Two or more fitted Sampler instances, either as a list or a dict mapping names to samplers.
    filename : Optional[str], default=None
        If provided (e.g. 'model_comparison.html'), saves a self-contained interactive HTML report.
    show : bool, default=True
        Whether to display the interactive figure in the current environment.
    compare_params : Optional[Sequence[str]], default=None
        Optional list of parameter keys to compare across models.
    renderer : Optional[str], default=None
        Plotly renderer option (e.g. 'browser', 'notebook', 'colab').

    Returns
    -------
    plotly.graph_objects.Figure
        The interactive Plotly Figure object containing the comparative report.

    Examples
    --------
    >>> from importance_sampling import compare_models
    >>> compare_models([sampler_standard, sampler_dual_lr, sampler_perseveration])
    >>>
    >>> # Save to standalone HTML file with parameter comparisons
    >>> compare_models(
    ...     [m1, m2, m3],
    ...     filename="comparison_report.html",
    ...     compare_params=["lr", "inv_temp"],
    ... )
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for model comparison reports. Install it via: pip install plotly"
        ) from exc

    # Normalize models dictionary
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

    # Calculate final metrics for each model
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

    # Calculate delta BIC relative to the minimum (best) BIC
    valid_bics = [b for b in final_bic if not np.isnan(b)]
    best_bic = min(valid_bics) if valid_bics else 0.0
    delta_bic = [b - best_bic if not np.isnan(b) else np.nan for b in final_bic]

    # Ranking by BIC (lowest BIC = rank 1)
    sorted_indices = sorted(
        range(len(model_names)),
        key=lambda i: final_bic[i] if not np.isnan(final_bic[i]) else float("inf"),
    )
    ranks = [0] * len(model_names)
    for rank, idx in enumerate(sorted_indices, start=1):
        ranks[idx] = rank

    # Parameter comparison setup
    include_params = compare_params is not None and len(compare_params) > 0
    param_rows = len(compare_params) if include_params and compare_params else 0

    # Layout structure:
    # Row 1: Evolution Trajectories (Col 1: Evidence Evolution, Col 2: BIC Evolution)
    # Row 2: Final Fit Comparison (Col 1: Bar Plot, Col 2: Table Summary)
    # Rows 3..3+param_rows: Parameter Comparison (if compare_params given)
    total_rows = 2 + param_rows

    subplot_titles = [
        "Total Evidence Evolution Across Iterations",
        "BIC Evolution Across Iterations (Lower is Better)",
        "Final Model Evidence & BIC Comparison",
        "Final Model Comparison Table",
    ]

    if include_params and compare_params:
        for p in compare_params:
            subplot_titles.append(f"Parameter Comparison across Models: {p}")
            subplot_titles.append("")

    specs: List[List[Dict[str, Any]]] = [
        [{}, {}],
        [{}, {"type": "table"}],
    ]

    for _ in range(param_rows):
        specs.append([{"colspan": 2}, None])

    fig = make_subplots(
        rows=total_rows,
        cols=2,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    # -------------------------------------------------------------------------
    # 1. EVOLUTION PLOTS (Row 1)
    # -------------------------------------------------------------------------
    for idx, name in enumerate(model_names):
        m = model_dict[name]
        color = palette[idx % len(palette)]
        ev_iters = list(range(len(m.evidence)))

        # Evidence trajectory
        fig.add_trace(
            go.Scatter(
                x=ev_iters,
                y=m.evidence,
                mode="lines+markers",
                name=f"{name} (Ev)",
                line=dict(color=color, width=2.5),
                marker=dict(size=4),
                hovertemplate=f"<b>{name}</b><br>Iter %{{x}}: Evidence = %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # BIC trajectory
        if m.BIC:
            fig.add_trace(
                go.Scatter(
                    x=ev_iters,
                    y=m.BIC,
                    mode="lines+markers",
                    name=f"{name} (BIC)",
                    line=dict(color=color, width=2.2, dash="dash"),
                    marker=dict(size=4),
                    hovertemplate=f"<b>{name}</b><br>Iter %{{x}}: BIC = %{{y:.2f}}<extra></extra>",
                ),
                row=1,
                col=2,
            )

    fig.update_xaxes(title_text="Iteration", row=1, col=1, gridcolor="#eaeef2")
    fig.update_yaxes(title_text="Total Evidence (Log Likelihood)", row=1, col=1, gridcolor="#eaeef2")
    fig.update_xaxes(title_text="Iteration", row=1, col=2, gridcolor="#eaeef2")
    fig.update_yaxes(title_text="BIC (Bayesian Information Criterion)", row=1, col=2, gridcolor="#eaeef2")

    # -------------------------------------------------------------------------
    # 2. FINAL BAR PLOT & TABLE (Row 2)
    # -------------------------------------------------------------------------
    # Col 1: Final BIC & Evidence Bar Chart
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=final_bic,
            name="Final BIC",
            marker=dict(color="#cf222e", opacity=0.85),
            hovertemplate="<b>%{x}</b><br>Final BIC: %{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=model_names,
            y=final_evidence,
            name="Final Evidence",
            marker=dict(color="#0969da", opacity=0.85),
            hovertemplate="<b>%{x}</b><br>Final Evidence: %{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Model", row=2, col=1, gridcolor="#eaeef2")
    fig.update_yaxes(title_text="Score", row=2, col=1, gridcolor="#eaeef2")

    # Col 2: Comparative Summary Table
    table_header = ["Rank", "Model Name", "k (Params)", "Final Evidence", "Final BIC", "ΔBIC"]
    table_rows = [
        [f"#{ranks[i]}" for i in range(len(model_names))],
        model_names,
        [str(k) for k in n_params_list],
        [f"{ev:.2f}" for ev in final_evidence],
        [f"{bic:.2f}" for bic in final_bic],
        [f"+{d:.2f}" if d > 0 else f"{d:.2f}" for d in delta_bic],
    ]

    fig.add_trace(
        go.Table(
            header=dict(
                values=[f"<b>{h}</b>" for h in table_header],
                fill_color="#f6f8fa",
                align="center",
                font=dict(size=12, color="#1f2328"),
                line_color="#d1d9e0",
            ),
            cells=dict(
                values=table_rows,
                fill_color="#ffffff",
                align="center",
                font=dict(size=11, color="#1f2328"),
                line_color="#eaeef2",
                height=26,
            ),
        ),
        row=2,
        col=2,
    )

    # -------------------------------------------------------------------------
    # 3. PARAMETER COMPARISONS (Optional Rows 3.. )
    # -------------------------------------------------------------------------
    if include_params and compare_params:
        for p_idx, p in enumerate(compare_params):
            r = 3 + p_idx
            # Compare population mean +/- SD and subject scatter across models that have parameter p
            valid_models = [name for name in model_names if p in model_dict[name].params]
            p_means = []
            p_sds = []

            for name in valid_models:
                m = model_dict[name]
                p_means.append(m.hyper_params[p]["mean"])
                p_sds.append(m.hyper_params[p]["sd"])

            # Error bar of population mean +/- 1 SD
            fig.add_trace(
                go.Scatter(
                    x=valid_models,
                    y=p_means,
                    error_y=dict(type="data", array=p_sds, visible=True),
                    mode="markers+text",
                    text=[f"{m:.3f}" for m in p_means],
                    textposition="top right",
                    marker=dict(size=10, color="#8250df"),
                    name=f"{p} (Mean ± 1 SD)",
                    hovertemplate=f"<b>%{{x}}</b><br>{p} Population Mean: %{{y:.3f}}<extra></extra>",
                ),
                row=r,
                col=1,
            )

            # Scatter of subject means across models
            for idx_m, name in enumerate(valid_models):
                m = model_dict[name]
                if m.mean_params and p in m.mean_params:
                    s_vals = m.mean_params[p]
                    fig.add_trace(
                        go.Scatter(
                            x=[name] * len(s_vals),
                            y=s_vals,
                            mode="markers",
                            marker=dict(
                                size=6,
                                color=palette[idx_m % len(palette)],
                                opacity=0.7,
                            ),
                            showlegend=False,
                            name=f"{name} subjects",
                            hovertemplate=f"<b>{name}</b><br>Subject {p} Mean: %{{y:.3f}}<extra></extra>",
                        ),
                        row=r,
                        col=1,
                    )

            fig.update_xaxes(title_text="Model", row=r, col=1, gridcolor="#eaeef2")
            fig.update_yaxes(title_text=f"{p} Value", row=r, col=1, gridcolor="#eaeef2")

    # Overall Layout & Height
    total_height = 700 + param_rows * 280
    fig.update_layout(
        title=dict(
            text=f"<b>importance_sampling Comparative Model Report</b><br>"
            f"<sup>Comparing {len(model_names)} Models ({', '.join(model_names)}) | "
            f"Best Model by BIC: <b>{model_names[sorted_indices[0]]}</b></sup>",
            x=0.03,
            y=0.985,
            xanchor="left",
            yanchor="top",
            font=dict(size=18, color="#1f2328", family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
        ),
        height=total_height,
        template="plotly_white",
        barmode="group",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
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


def compare_parameters(
    samplers: Union[Sequence["Sampler"], Dict[str, "Sampler"]],
    params: Optional[Sequence[str]] = None,
    filename: Optional[str] = None,
    show: bool = True,
    renderer: Optional[str] = None,
) -> Any:
    """Generate a dedicated parameter comparison report across multiple models.

    Useful when you want to inspect parameter estimates across models without
    overcrowding the main model fit report.

    Parameters
    ----------
    samplers : Union[Sequence[Sampler], Dict[str, Sampler]]
        Two or more fitted Sampler instances.
    params : Optional[Sequence[str]], default=None
        List of parameter keys to compare. If None, automatically selects all
        parameters shared by at least two models.
    filename : Optional[str], default=None
        If provided, exports a self-contained HTML file.
    show : bool, default=True
        Whether to display the interactive figure in the current environment.
    renderer : Optional[str], default=None
        Plotly renderer option.
    """
    if isinstance(samplers, dict):
        model_dict = samplers
    else:
        model_dict = {
            getattr(s, "model_name", None) or f"Model_{i + 1}": s
            for i, s in enumerate(samplers)
        }

    # Discover shared parameters if not provided
    if params is None:
        param_counts: Dict[str, int] = {}
        for s in model_dict.values():
            for p in s.params:
                param_counts[p] = param_counts.get(p, 0) + 1
        params = [p for p, count in param_counts.items() if count >= 2]
        if not params:
            # Fallback to all parameters
            params = list(param_counts.keys())

    return compare_models(
        model_dict,
        filename=filename,
        show=show,
        compare_params=params,
        renderer=renderer,
    )

