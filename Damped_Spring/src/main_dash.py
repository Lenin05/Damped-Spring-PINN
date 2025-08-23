import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

from plotly.subplots import make_subplots


# ==============================
# 📌 Solución Analítica
# ==============================
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


# ==============================
# 📌 App principal Streamlit
# ==============================
def main():
    st.set_page_config(page_title="Damped Spring PINN", layout="wide")

    # Crear pestañas
    tab1, tab2 = st.tabs(["🏃 Simulación", "📊 Análisis de Datos"])

    with tab1:
        st.title("🌌 Comparación: PINN vs IA vs Solución Analítica")

        st.markdown(
            """
            Esta aplicación compara:
            - 📘 **Solución Analítica** del oscilador amortiguado  
            - 🤖 **Modelo IA Pura**  
            - 🧠 **PINN (Physics-Informed Neural Network)**  
            """
        )

        # Parámetros ajustables
        st.sidebar.header("⚙️ Parámetros del Sistema")

        masa_test = st.sidebar.slider("Masa (m)", 0.5, 2.0, 2.0, 0.1)
        x0_test = st.sidebar.slider("Posición inicial (x0)", -10.0, 10.0, -5.0, 0.1)
        v0_test = st.sidebar.slider("Velocidad inicial (v0)", -10.0, 10.0, 10.0, 0.5)

        k = 20.0
        c = 1.0

        if st.button("🚀 Start Simulation"):
            # Cargar modelos
            with st.spinner("Cargando modelos..."):
                ia_model = tf.keras.models.load_model("Damped_Spring/models/ia_model/1")
                pinn_model = tf.keras.models.load_model(
                    "Damped_Spring/models/pinn_model/1"
                )

            # Tiempo fijo
            t_test = np.linspace(
                0, 10, 300
            )  # menos puntos para que la animación sea fluida

            X_test = np.column_stack(
                [
                    t_test,
                    np.full_like(t_test, masa_test),
                    np.full_like(t_test, x0_test),
                    np.full_like(t_test, v0_test),
                ]
            )

            # Predicciones
            y_pred_pinn = pinn_model.predict(X_test, verbose=0)
            y_pred_ia = ia_model.predict(X_test, verbose=0)
            x_real = analytical_solution(t_test, masa_test, k, c, x0_test, v0_test)

            # Asegurarnos de que las predicciones sean arrays 1D
            y_pred_pinn = y_pred_pinn.squeeze()
            y_pred_ia = y_pred_ia.squeeze()

            # Gráfico con Plotly
            fig = go.Figure()

            # Líneas estáticas (trazos 0, 1, 2)
            fig.add_trace(
                go.Scatter(
                    x=t_test,
                    y=x_real,
                    mode="lines",
                    name="Analítica",
                    line=dict(color="deepskyblue", width=3),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=t_test,
                    y=y_pred_pinn,
                    mode="lines",
                    name="PINN",
                    line=dict(color="magenta", dash="dash", width=3),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=t_test,
                    y=y_pred_ia,
                    mode="lines",
                    name="IA",
                    line=dict(color="limegreen", dash="dot", width=3),
                )
            )

            # Marcadores iniciales (trazos 3, 4, 5)
            fig.add_trace(
                go.Scatter(
                    x=[t_test[0]],
                    y=[x_real[0]],
                    mode="markers",
                    name="Analítica (punto)",
                    marker=dict(color="deepskyblue", size=12),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[t_test[0]],
                    y=[y_pred_pinn[0]],
                    mode="markers",
                    name="PINN (punto)",
                    marker=dict(color="magenta", size=12),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[t_test[0]],
                    y=[y_pred_ia[0]],
                    mode="markers",
                    name="IA (punto)",
                    marker=dict(color="limegreen", size=12),
                )
            )

            # Frames para animación - SOLO actualizan los puntos (trazos 3, 4 y 5)
            frames = []
            for i in range(len(t_test)):
                frames.append(
                    go.Frame(
                        data=[
                            go.Scatter(
                                x=[t_test[i]], y=[x_real[i]]
                            ),  # Actualiza punto analítico
                            go.Scatter(
                                x=[t_test[i]], y=[y_pred_pinn[i]]
                            ),  # Actualiza punto PINN
                            go.Scatter(
                                x=[t_test[i]], y=[y_pred_ia[i]]
                            ),  # Actualiza punto IA
                        ],
                        traces=[
                            3,
                            4,
                            5,
                        ],  # Índices de los trazos a actualizar (los puntos)
                    )
                )

            fig.frames = frames

            # Botones de control
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        buttons=[
                            dict(
                                label="▶️ Play",
                                method="animate",
                                args=[
                                    None,
                                    {
                                        "frame": {"duration": 25, "redraw": True},
                                        "fromcurrent": True,
                                        "transition": {"duration": 0},
                                    },
                                ],
                            ),
                            dict(
                                label="⏸️ Pause",
                                method="animate",
                                args=[
                                    [None],
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                    },
                                ],
                            ),
                        ],
                        x=0.1,
                        y=-0.2,
                        xanchor="left",
                        yanchor="top",
                    )
                ],
                title="Oscilación de un Sistema Masa Resorte Subamortiguado",
                template="plotly_dark",
                width=1000,
                height=600,
                xaxis_title="Tiempo (s)",
                yaxis_title="Oscilación (x)",
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.title("📊 Análisis de Datos del Sistema")

        try:
            # Cargar datos del CSV
            df = pd.read_csv("Damped_Spring/files/data_experiment.csv")

            # Obtener solo el primer dato de cada experimento
            df_first = df.groupby("experimento").first().reset_index()

            # Mostrar información básica del dataset
            st.subheader(
                "📋 Vista Previa de los Datos (Primer punto de cada experimento)"
            )
            st.dataframe(df_first.head())

            # Mostrar estadísticas descriptivas
            st.subheader("📈 Estadísticas Descriptivas")
            st.dataframe(df_first.describe())

            # Análisis de distribución de parámetros
            st.subheader("📊 Distribución de Parámetros")

            # Crear subplots para las distribuciones
            fig_dist = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Distribución de Masa",
                    "Distribución de Posición Inicial",
                    "Distribución de Velocidad Inicial",
                    "Distribución de Oscilación Inicial",
                ),
            )

            # Histograma de masa
            fig_dist.add_trace(
                go.Histogram(x=df_first["masa"], nbinsx=50, name="Masa"), row=1, col=1
            )

            # Histograma de posición inicial
            fig_dist.add_trace(
                go.Histogram(
                    x=df_first["posicion_inicial"], nbinsx=50, name="Posición Inicial"
                ),
                row=1,
                col=2,
            )

            # Histograma de velocidad inicial
            fig_dist.add_trace(
                go.Histogram(
                    x=df_first["velocidad_inicial"], nbinsx=50, name="Velocidad Inicial"
                ),
                row=2,
                col=1,
            )

            # Histograma de oscilación inicial
            fig_dist.add_trace(
                go.Histogram(
                    x=df_first["oscilacion"], nbinsx=50, name="Oscilación Inicial"
                ),
                row=2,
                col=2,
            )

            fig_dist.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)

            # Gráfica de oscilaciones de experimentos
            st.subheader("📈 Experimentos Realizados")

            # Seleccionar un subconjunto de experimentos para no saturar la gráfica
            unique_experiments = df["experimento"].unique()
            num_experiments = len(unique_experiments)

            # Permitir al usuario elegir cuántos experimentos mostrar
            max_experiments = st.slider(
                "Número máximo de experimentos a mostrar",
                min_value=5,
                max_value=min(25, num_experiments),
                value=min(10, num_experiments),
            )

            # Seleccionar experimentos aleatoriamente
            if num_experiments > max_experiments:
                selected_experiments = np.random.choice(
                    unique_experiments, max_experiments, replace=False
                )
            else:
                selected_experiments = unique_experiments

            # Crear la figura
            fig_oscillations = go.Figure()

            # Añadir cada experimento seleccionado
            for exp_id in selected_experiments:
                data_exp = df[df["experimento"] == exp_id]

                # Obtener parámetros para la leyenda
                m = data_exp["masa"].iloc[0]
                x0 = data_exp["posicion_inicial"].iloc[0]
                v0 = data_exp["velocidad_inicial"].iloc[0]

                fig_oscillations.add_trace(
                    go.Scatter(
                        x=data_exp["tiempo"],
                        y=data_exp["oscilacion"],
                        mode="lines",
                        name=f"Exp {exp_id} (m={m:.2f}, x0={x0:.2f}, v0={v0:.2f})",
                        opacity=0.7,
                    )
                )

            # Actualizar diseño
            fig_oscillations.update_layout(
                title="Oscilaciones Subamortiguadas con Ruido - Experimentos con variabilidad",
                xaxis_title="Tiempo [s]",
                yaxis_title="Amplitud",
                height=600,
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=0, xanchor="right", x=1
                ),
            )

            st.plotly_chart(fig_oscillations, use_container_width=True)

        except FileNotFoundError:
            st.error(
                "❌ No se pudo encontrar el archivo de datos. Asegúrate de que la ruta 'Damped_Spring/files/data_experiment.csv' es correcta."
            )
        except Exception as e:
            st.error(f"❌ Ocurrió un error al cargar los datos: {str(e)}")


if __name__ == "__main__":
    main()
