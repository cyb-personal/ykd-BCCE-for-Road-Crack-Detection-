import tensorflow as tf
from keras import layers

def channel_attention(input_feature, ratio=8):
    """通道注意力：全局平均/最大池化 + 共享MLP + 激活"""
    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    max_pool = layers.GlobalMaxPooling2D()(input_feature)

    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    shared_mlp = layers.Conv2D(channel // ratio, (1, 1), padding="same", activation="relu")
    avg_mlp = shared_mlp(avg_pool)
    max_mlp = shared_mlp(max_pool)

    channel_att = layers.Add()([avg_mlp, max_mlp])
    channel_att = layers.Activation("sigmoid")(channel_att)

    if channel_axis == 1:
        channel_att = layers.Permute((3, 1, 2))(channel_att)
    return layers.Multiply()([input_feature, channel_att])

def spatial_attention(input_feature):
    """空间注意力：通道维度平均/最大池化 + 7×7卷积 + 激活"""
    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1

    avg_pool = layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=channel_axis, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: tf.keras.backend.max(x, axis=channel_axis, keepdims=True))(input_feature)
    concat = layers.Concatenate(axis=channel_axis)([avg_pool, max_pool])

    spatial_att = layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid")(concat)
    return layers.Multiply()([input_feature, spatial_att])

def CBAM(input_feature, ratio=8):
    """CBAM模块：通道注意力 → 空间注意力 串行执行"""
    x = channel_attention(input_feature, ratio)
    x = spatial_attention(x)
    return x