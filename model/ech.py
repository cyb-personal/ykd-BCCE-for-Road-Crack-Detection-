import tensorflow as tf
from keras import layers


def mish(x):
    """Mish激活函数：x * tanh(softplus(x))"""
    return x * tf.keras.backend.tanh(tf.keras.backend.softplus(x))


def EfficientClassifier(input_tensor, num_classes, reduction_dim=128, dropout_rate=0.5):
    """高效分类器：1×1卷积降维 + BN + Mish + Dropout + Softmax"""
    x = layers.Conv2D(reduction_dim, (1, 1), padding="same", activation=None)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(mish)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(num_classes, activation="softmax")(x)
    return output