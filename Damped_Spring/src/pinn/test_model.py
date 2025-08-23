import numpy as np
import plotly.graph_objects as go
import tensorflow as tf


def analytical_solution(t, m, k, c, x0, v0):
    omega_n = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(m * k))

    if zeta >= 1:
        raise ValueError("El sistema no está subamortiguado (zeta < 1 requerido).")

    omega_d = omega_n * np.sqrt(1 - zeta**2)

    A = x0
    B = (v0 + zeta * omega_n * x0) / omega_d

    return np.exp(-zeta * omega_n * t) * (
        A * np.cos(omega_d * t) + B * np.sin(omega_d * t)
    )


def run():
    print("🔮 Usando modelos entrenados para predicciones...")

    ia_model = tf.keras.models.load_model("Damped_Spring/models/ia_model/1")
    pinn_model = tf.keras.models.load_model("Damped_Spring/models/pinn_model/1")

    t_test = np.linspace(0, 10, 1000)
    masa_test = 2.0
    k = 20.0
    c = 1.0
    x0_test = -5.0
    v0_test = 10.0

    X_test = np.column_stack(
        [
            t_test,
            np.full_like(t_test, masa_test),
            np.full_like(t_test, x0_test),
            np.full_like(t_test, v0_test),
        ]
    )

    y_pred_pinn = pinn_model.predict(X_test)
    y_pred_ia = ia_model.predict(X_test)

    x_real = analytical_solution(t_test, masa_test, k, c, x0_test, v0_test)

    fig = go.Figure()

    # Solución analítica
    fig.add_trace(
        go.Scatter(
            x=t_test,
            y=x_real,
            mode="lines",
            name="Solución Analítica",
            line=dict(color="deepskyblue", width=4),
            hovertemplate="t: %{x:.2f}<br>x: %{y:.2f}",
        )
    )

    # PINN
    fig.add_trace(
        go.Scatter(
            x=t_test,
            y=y_pred_pinn.squeeze(),
            mode="lines",
            name="Predicción PINN",
            line=dict(color="magenta", dash="dash", width=5),
            hovertemplate="t: %{x:.2f}<br>x: %{y:.2f}",
        )
    )

    # IA pura
    fig.add_trace(
        go.Scatter(
            x=t_test,
            y=y_pred_ia.squeeze(),
            mode="lines",
            name="Predicción IA",
            line=dict(color="limegreen", dash="dot", width=3),
            hovertemplate="t: %{x:.2f}<br>x: %{y:.2f}",
        )
    )

    # Estilo visual
    fig.update_layout(
        title=dict(
            text="🌌 Comparación: Red Neuronal vs Solución Analítica",
            font=dict(size=24, color="white"),
            x=0.5,
        ),
        xaxis_title="Tiempo (s)",
        yaxis_title="Oscilación (x)",
        xaxis=dict(gridcolor="gray"),
        yaxis=dict(gridcolor="gray"),
        legend=dict(
            x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)", font=dict(color="white", size=12)
        ),
        template="plotly_dark",
        width=1000,
        height=600,
    )

    fig.show()
