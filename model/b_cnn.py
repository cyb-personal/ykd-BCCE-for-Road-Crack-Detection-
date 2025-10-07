from keras import layers


def bilinear_conv_block(input_tensor, filters, kernel_size, strides=(1, 1), padding="same"):
    """双线性卷积分支：生成两个并行卷积特征图，用于后续双线性池化"""
    branch1 = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, activation="relu")(input_tensor)
    branch1 = layers.BatchNormalization()(branch1)

    branch2 = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, activation="relu")(input_tensor)
    branch2 = layers.BatchNormalization()(branch2)
    return branch1, branch2


def bilinear_pooling(branch1, branch2):
    """双线性池化：计算两个特征图的元素乘积 + 全局平均池化"""
    bilinear_product = layers.Multiply()([branch1, branch2])
    pooled = layers.GlobalAveragePooling2D()(bilinear_product)
    return pooled