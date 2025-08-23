import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping


def run():
    print("🦾 Training the Neuronal Network Model...")

    df = pd.read_csv("Damped_Spring/files/data_experiment.csv")

    input_fields = ["tiempo", "masa", "posicion_inicial", "velocidad_inicial"]
    output_fields = ["oscilacion"]

    df_input = df[input_fields]
    df_output = df[output_fields]

    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Separación en conjuntos
    X_train = df_input.iloc[:n_train]
    y_train = df_output.iloc[:n_train]

    X_val = df_input.iloc[n_train : n_train + n_val]
    y_val = df_output.iloc[n_train : n_train + n_val]

    X_test = df_input.iloc[n_train + n_val :]
    y_test = df_output.iloc[n_train + n_val :]

    model = models.Sequential(
        [
            layers.Dense(32, activation="relu", input_shape=(4,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse")

    # Definir el callback de Early Stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        verbose=1,
        callbacks=[early_stopping],
    )

    # --- VISUALIZACIÓN DEL HISTORIAL ---
    plt.figure(figsize=(10, 6))

    # Pérdida total
    plt.plot(history.history["loss"], label="Train Total Loss")
    plt.plot(history.history["val_loss"], label="Val Total Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    masa_test = 2
    k = 20.0
    c = 1.0
    x0_test = 5
    v0_test = 5

    t_test = np.linspace(0, 10, 1000)

    X_test = np.column_stack(
        [
            t_test,
            np.full_like(t_test, masa_test),
            np.full_like(t_test, x0_test),
            np.full_like(t_test, v0_test),
        ]
    )

    y_pred = model.predict(X_test)

    # -------------------------
    # Solución analítica del SHO
    # -------------------------
    omega_n = np.sqrt(k / masa_test)
    zeta = c / (2 * np.sqrt(masa_test * k))
    omega_d = omega_n * np.sqrt(1 - zeta**2)

    # Constantes A y B
    A = x0_test
    B = (v0_test + zeta * omega_n * x0_test) / omega_d

    def x(t):
        return np.exp(-zeta * omega_n * t) * (
            A * np.cos(omega_d * t) + B * np.sin(omega_d * t)
        )

    x_real = x(t_test)

    # -------------------------
    # Graficar
    # -------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(t_test, x_real, label="Solución Analítica", color="blue", linewidth=2)
    plt.plot(t_test, y_pred, label="Predicción NN", color="red", linestyle="--")
    plt.xlabel("Tiempo")
    plt.ylabel("Oscilación")
    plt.title("Comparación: Red Neuronal vs Solución Analítica")
    plt.legend()
    plt.grid(True)
    plt.show()

    model.save("Damped_Spring/models/ia_model/1")
