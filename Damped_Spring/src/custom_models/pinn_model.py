import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class ResorteModel(tf.keras.Model):
    def __init__(self, net_main, scalerY, k=20.0, c=1.0):
        super().__init__()
        self.net_main = net_main  # tu red neuronal
        self.scalerY = scalerY  # escalador de salida
        self.k = k
        self.c = c
        self.mse = tf.keras.losses.MeanSquaredError()

    def call(self, X, training=False):
        """
        X: [batch, 4] con columnas [tiempo, masa, pos_inicial, vel_inicial]
        """
        y_pred = self.net_main(X, training=training)
        return y_pred

    def compute_physics(self, X):
        """
        Calcula la solución analítica esperada según las ecuaciones del resorte amortiguado
        """
        t = X[:, 0]
        m = X[:, 1]
        x0 = X[:, 2]
        v0 = X[:, 3]

        # parámetros derivados
        omega_n = tf.sqrt(self.k / m)
        zeta = self.c / (2.0 * tf.sqrt(m * self.k))
        omega_d = omega_n * tf.sqrt(1.0 - zeta**2)

        A = x0
        B = (v0 + zeta * omega_n * x0) / omega_d

        x_exact = tf.exp(-zeta * omega_n * t) * (
            A * tf.cos(omega_d * t) + B * tf.sin(omega_d * t)
        )

        return tf.expand_dims(x_exact, -1)  # shape [batch, 1]

    def compute_losses(self, X, y_true, training=False):
        # predicción de la red
        y_pred = self.call(X, training=training)

        # pérdida de datos
        loss_data = self.mse(y_true, y_pred)

        # solución física
        y_phys = self.compute_physics(X)
        y_phys_scaled = self.scalerY(y_phys)

        # pérdida física
        loss_phys = self.mse(y_phys_scaled, y_pred)

        # combinación
        w_data, w_phys = 0.5, 0.5
        loss_total = w_data * loss_data + w_phys * loss_phys

        return loss_total, loss_data, loss_phys, y_pred

    def train_step(self, data):
        X, y_true = data
        with tf.GradientTape() as tape:
            loss_total, loss_data, loss_phys, y_pred = self.compute_losses(
                X, y_true, training=True
            )

        grads = tape.gradient(loss_total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {
            "loss": loss_total,
            "loss_data": loss_data,
            "loss_phys": loss_phys,
        }

    def test_step(self, data):
        X, y_true = data
        loss_total, loss_data, loss_phys, y_pred = self.compute_losses(
            X, y_true, training=False
        )

        return {
            "loss": loss_total,
            "loss_data": loss_data,
            "loss_phys": loss_phys,
        }
