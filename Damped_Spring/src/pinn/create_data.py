import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run():
    print("📀 Generating training dataset for the model...")

    k = 20.0  # constante del resorte
    c = 1.0  # factor de amortiguamiento

    t = np.linspace(0, 10, 1000)

    experimentos = []
    np.random.seed(42)

    for i in range(1, 100):
        m = np.random.uniform(0.5, 2.0)
        x0 = np.random.uniform(-10.0, 10.0)
        v0 = np.random.uniform(0, 10.0)

        omega_n = np.sqrt(k / m)
        zeta = c / (2 * np.sqrt(m * k))
        omega_d = omega_n * np.sqrt(1 - zeta**2)

        A = x0
        B = (v0 + zeta * omega_n * x0) / omega_d

        def x(t):
            return np.exp(-zeta * omega_n * t) * (
                A * np.cos(omega_d * t) + B * np.sin(omega_d * t)
            )

        ruido = np.random.normal(0, 0.1, size=t.shape)
        x_ruidoso = x(t) + ruido

        df_temp = pd.DataFrame(
            {
                "tiempo": t,
                "experimento": i,
                "masa": m,
                "posicion_inicial": x0,
                "velocidad_inicial": v0,
                "oscilacion": x_ruidoso,
            }
        )

        experimentos.append(df_temp)

    df = pd.concat(experimentos, ignore_index=True)
    print(df.head())
    print(f"\nDataFrame final: {df.shape[0]} filas, {df.shape[1]} columnas")

    plt.figure(figsize=(12, 6))

    for exp_id in df["experimento"].unique():
        data_exp = df[df["experimento"] == exp_id]
        plt.plot(data_exp["tiempo"], data_exp["oscilacion"], alpha=0.6)

    plt.title("Oscilaciones Subamortiguadas con Ruido - Experimentos con variabilidad")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.legend(ncol=3, fontsize=8)
    plt.show()

    df.to_csv("Damped_Spring/files/data_experiment.csv", index=False)
    print("\nArchivo 'data_experiment.csv' guardado correctamente ✅")
