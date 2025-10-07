from keras import Input, Model
from keras import layers
from .b_cnn import bilinear_conv_block, bilinear_pooling
from .cbam import CBAM
from .ech import EfficientClassifier


def BCCE_Model(input_shape=(400, 400, 3), num_classes=4):
    """BCCE模型：整合双线性卷积、CBAM注意力、组合池化、高效分类器"""
    # 输入层
    input_layer = Input(shape=input_shape)

    # 基础卷积层（Conv1 → Conv2 → Pool3 → Conv4）
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(input_layer)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # 双线性卷积分支（B-CNN核心）
    branch1, branch2 = bilinear_conv_block(x, filters=128, kernel_size=(3, 3))
    bilinear_feat = bilinear_pooling(branch1, branch2)
    bilinear_feat = layers.Reshape((1, 1, -1))(bilinear_feat)  # 调整为4D张量

    # 上层CBAM注意力
    cbam_out1 = CBAM(x)

    # 后续卷积（Conv5）
    conv5 = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(cbam_out1)
    conv5 = layers.BatchNormalization()(conv5)

    # 下层CBAM注意力
    cbam_out2 = CBAM(conv5)

    # 组合池化：全局最大池化（GMP） + 全局平均池化（GAP）
    gmp = layers.GlobalMaxPooling2D()(cbam_out2)
    gap = layers.GlobalAveragePooling2D()(cbam_out2)
    combined_pool = layers.Concatenate()([gmp, gap, bilinear_feat])  # 拼接双线性特征
    combined_pool = layers.Reshape((1, 1, -1))(combined_pool)  # 适配1×1卷积

    # 高效分类器（降维 + 分类）
    output = EfficientClassifier(combined_pool, num_classes=num_classes)

    # 构建模型
    model = Model(inputs=input_layer, outputs=output, name="BCCE")
    return model


# 测试模型结构（运行脚本时打印网络结构）
if __name__ == "__main__":
    model = BCCE_Model(input_shape=(400, 400, 3), num_classes=4)
    model.summary()