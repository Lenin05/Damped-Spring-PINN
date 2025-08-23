import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class NormalizerLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.y_mean = None
        self.y_std = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.y_mean = self.add_weight(
            name="mean", shape=(input_dim,), initializer="zeros", trainable=False
        )
        self.y_std = self.add_weight(
            name="standard_deviation",
            shape=(input_dim,),
            initializer="ones",
            trainable=False,
        )
        super().build(input_shape)

    def adapt(self, data):
        data = tf.cast(data, tf.float32)
        if not self.built:
            self.build(data.shape)

        # Calcular estadísticas
        mean = tf.reduce_mean(data, axis=0)
        std = tf.math.reduce_std(data, axis=0)

        # Protección contra división por cero
        std_safe = tf.where(std == 0.0, tf.ones_like(std), std)

        self.y_mean.assign(mean)
        self.y_std.assign(std_safe)

    @tf.function
    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        return tf.divide(tf.subtract(inputs, self.y_mean), self.y_std)

    @tf.function
    def inverse(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        return tf.add(tf.multiply(inputs, self.y_std), self.y_mean)


@tf.keras.utils.register_keras_serializable()
class NormalizerInverseLayer(tf.keras.layers.Layer):
    def __init__(self, scaler_layer, **kwargs):
        super().__init__(**kwargs)
        self.scaler_layer = scaler_layer  # referencia a la capa entrenada

    @tf.function
    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        return self.scaler_layer.inverse(inputs)
