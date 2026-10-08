import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Fourier Approximation", layout="wide")

# Custom minimal clean styling
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .stSlider { margin-top: -15px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Discrete Fourier Series Approximator")

# 1. Preset or sample data
if "data" not in st.session_state:
    x_init = np.linspace(0, 2 * np.pi, 9)[:-1]
    y_init = 2.0 + 3.0 * np.cos(x_init) - 1.5 * np.sin(2 * x_init)
    st.session_state.data = pd.DataFrame({"x": x_init, "f(x)": y_init})

col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("1. Data Points")
    edited_data = st.data_editor(
        st.session_state.data,
        num_rows="dynamic",
        use_container_width=True,
        height=320,
        column_config={
            "x": st.column_config.NumberColumn("x (rad)", format="%.4f"),
            "f(x)": st.column_config.NumberColumn("f(x)", format="%.4f"),
        },
    )

    st.subheader("2. Harmonics (Live)")
    # Reactive Slider: Updates the curve instantly
    m_harmonics = st.slider("Select M (Harmonics Order)", min_value=1, max_value=20, value=3)

# 2. Math Calculations
data = edited_data.dropna().copy()
data["x"] = pd.to_numeric(data["x"], errors="coerce")
data["f(x)"] = pd.to_numeric(data["f(x)"], errors="coerce")
data = data.dropna().sort_values("x")

with col_right:
    if len(data) < 3:
        st.warning("Please provide at least 3 points.")
    else:
        x_pts = data["x"].to_numpy(dtype=float)
        y_pts = data["f(x)"].to_numpy(dtype=float)
        N = len(x_pts)

        # Discrete Fourier formulas
        a0 = np.sum(y_pts) / N
        an = [(2 * np.sum(y_pts * np.cos(n * x_pts))) / N for n in range(1, m_harmonics + 1)]
        bn = [(2 * np.sum(y_pts * np.sin(n * x_pts))) / N for n in range(1, m_harmonics + 1)]

        # Generate smooth function curve
        x_dense = np.linspace(np.min(x_pts), np.max(x_pts), 600)
        y_dense = np.full_like(x_dense, a0)
        for n, (a, b) in enumerate(zip(an, bn), start=1):
            y_dense += a * np.cos(n * x_dense) + b * np.sin(n * x_dense)

        # Plot (Desmos / Modern Scientific Look)
        fig = go.Figure()

        # Fitted curve
        fig.add_trace(
            go.Scatter(
                x=x_dense,
                y=y_dense,
                mode="lines",
                name=f"Fourier Fit (M={m_harmonics})",
                line=dict(color="#0066FF", width=3),
            )
        )

        # Data markers
        fig.add_trace(
            go.Scatter(
                x=x_pts,
                y=y_pts,
                mode="markers",
                name="Input Points",
                marker=dict(color="#FF3366", size=10, line=dict(color="white", width=2)),
            )
        )

        fig.update_layout(
            height=460,
            margin=dict(l=10, r=10, t=20, b=10),
            template="simple_white",
            xaxis=dict(showgrid=True, gridcolor="#E5E7EB", zeroline=True, zerolinecolor="#9CA3AF"),
            yaxis=dict(showgrid=True, gridcolor="#E5E7EB", zeroline=True, zerolinecolor="#9CA3AF"),
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
        )

        st.plotly_chart(fig, use_container_width=True)

        # LaTeX Equation Display
        terms = [f"{a0:.2f}"]
        for n, (a, b) in enumerate(zip(an, bn), start=1):
            if abs(a) > 0.01:
                terms.append(f"{'+' if a > 0 else '-'} {abs(a):.2f}\\cos({n}x)")
            if abs(b) > 0.01:
                terms.append(f"{'+' if b > 0 else '-'} {abs(b):.2f}\\sin({n}x)")

        st.markdown("**Approximation Formula:**")
        st.latex(r"\hat{f}(x) = " + " ".join(terms))

        # Compact Table
        with st.expander("Show Coefficients Table"):
            coeff_df = pd.DataFrame({
                "n": list(range(1, m_harmonics + 1)),
                "an": an,
                "bn": bn
            })
            st.dataframe(coeff_df.style.format({"an": "{:.4f}", "bn": "{:.4f}"}), use_container_width=True, hide_index=True)
