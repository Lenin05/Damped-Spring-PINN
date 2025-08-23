import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from custom_models.pinn_model import ResorteModel
from custom_models.scaler_layers import NormalizerInverseLayer, NormalizerLayer


def set_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def set_global_determinism(seed=42):
    set_seeds(seed)

    # Configurar TensorFlow para usar operaciones deterministas
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    # Configurar paralelismo de threads (puede afectar la reproducibilidad)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


def run():
    print("🦾 Training the Physics-Informed Neural Network...")
    set_global_determinism(seed=42)

    # Cargar datos
    df = pd.read_csv("Damped_Spring/files/data_experiment.csv")

    print(df)

    # Columnas de entrada y salida
    input_fields = ["tiempo", "masa", "posicion_inicial", "velocidad_inicial"]
    output_fields = ["oscilacion"]

    df_input = df[input_fields]
    df_output = df[output_fields]

    # División de dataset
    train_ratio = 0.7
    val_ratio = 0.2

    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Separación en conjuntos
    df_input_train = df_input.iloc[:n_train]
    df_output_train = df_output.iloc[:n_train]
    df_input_val = df_input.iloc[n_train : n_train + n_val]
    df_output_val = df_output.iloc[n_train : n_train + n_val]

    # TensorFlow tensors
    input_train_tensor = tf.convert_to_tensor(df_input_train.values, dtype=tf.float32)
    output_train_tensor = tf.convert_to_tensor(df_output_train.values, dtype=tf.float32)

    # Escaladores personalizados
    scalerX = NormalizerLayer()
    scalerY = NormalizerLayer()
    scalerX.adapt(input_train_tensor)
    scalerY.adapt(output_train_tensor)

    y_train_scale = scalerY(output_train_tensor)
    y_val_scale = scalerY(tf.convert_to_tensor(df_output_val.values, dtype=tf.float32))

    x_train = input_train_tensor
    x_val = tf.convert_to_tensor(df_input_val.values, dtype=tf.float32)

    # --- DEFINICIÓN DEL MODELO ---
    input_shape = (df_input_train.shape[1],)  # e.g. (n_features,)
    output_dim = df_output_train.shape[1]

    def build_net_main(input_shape=input_shape, output_dim=output_dim, scalerX=None):
        model = tf.keras.models.Sequential(
            [
                scalerX,
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Dense(128, activation="swish"),
                tf.keras.layers.Dense(64, activation="swish"),
                tf.keras.layers.Dense(64, activation="swish"),
                tf.keras.layers.Dense(32, activation="swish"),
                tf.keras.layers.Dense(32, activation="swish"),
                tf.keras.layers.Dense(output_dim, dtype="float32"),
            ],
            name="Resorte",
        )

        return model

    net_main = build_net_main(
        input_shape=input_shape, output_dim=output_dim, scalerX=scalerX
    )

    pinn = ResorteModel(net_main=net_main, scalerY=scalerY, k=20.0, c=1.0)

    pinn.compile(optimizer=tf.keras.optimizers.Adam(0.0001))

    history = pinn.fit(
        x_train,
        y_train_scale,
        validation_data=(x_val, y_val_scale),
        batch_size=64,
        epochs=120,
    )

    serving_input = tf.keras.Input(shape=input_shape, name="serving_input")
    predictions = pinn(serving_input, training=False)
    rescaled_output = NormalizerInverseLayer(scalerY)(predictions)

    pinn_final = tf.keras.Model(inputs=serving_input, outputs=rescaled_output)

    # --- VISUALIZACIÓN DEL HISTORIAL ---
    plt.figure(figsize=(10, 6))

    # Pérdida total
    plt.plot(history.history["loss"], label="Train Total Loss")
    plt.plot(history.history["val_loss"], label="Val Total Loss")

    # Pérdida de datos
    plt.plot(history.history["loss_data"], label="Train Data Loss")
    plt.plot(history.history["val_loss_data"], label="Val Data Loss")

    # Pérdida física
    plt.plot(history.history["loss_phys"], label="Train Physics Loss")
    plt.plot(history.history["val_loss_phys"], label="Val Physics Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Vector de tiempo
    t_test = np.linspace(0, 10, 1000)

    masa_test = 2
    k = 20.0  # constante de resorte
    c = 1.0  # amortiguamiento
    x0_test = 5
    v0_test = 5

    X_test = np.column_stack(
        [
            t_test,
            np.full_like(t_test, masa_test),
            np.full_like(t_test, x0_test),
            np.full_like(t_test, v0_test),
        ]
    )

    y_pred = pinn_final.predict(X_test)

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

    plt.figure(figsize=(12, 6))
    plt.plot(t_test, x_real, label="Solución Analítica", color="blue", linewidth=2)
    plt.plot(t_test, y_pred, label="Predicción NN", color="red", linestyle="--")
    plt.xlabel("Tiempo")
    plt.ylabel("Oscilación")
    plt.title("Comparación: Red Neuronal vs Solución Analítica")
    plt.legend()
    plt.grid(True)
    plt.show()

    pinn_final.save("Damped_Spring/models/pinn_model/1", save_format="tf")
