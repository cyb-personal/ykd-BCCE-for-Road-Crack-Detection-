import tensorflow as tf
import argparse
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 导入自定义模块
from models.TSSC import build_tssc_model
from dataset.data_loader import PeaDiseaseDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='使用模型预测道路裂缝情况')
    parser.add_argument('--image_path', type=str, required=True,
                        help='待预测图像路径')
    parser.add_argument('--weight_path', type=str, default='./weights/best_tssc.h5',
                        help='模型权重文件路径')
    parser.add_argument('--data_dir', type=str, default='./pea_disease_dataset',
                        help='数据集根目录（用于获取类别信息）')
    parser.add_argument('--device', type=str, default='GPU',
                        choices=['GPU', 'CPU'], help='预测设备')
    parser.add_argument('--img_size', type=int, nargs=2, default=[400, 400],
                        help='图像尺寸')
    return parser.parse_args()


def preprocess_image(image_path, img_size):
    """预处理输入图像"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 转换为RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 调整大小
    image = cv2.resize(image, img_size)
    # 归一化
    image = image.astype(np.float32) / 255.0
    # 增加批次维度
    image = np.expand_dims(image, axis=0)
    return image


def predict(args):
    # 设置设备
    if args.device == 'GPU' and tf.test.is_gpu_available():
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("使用GPU进行预测")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("使用CPU进行预测")

    # 获取类别信息
    data_loader = PeaDiseaseDataLoader(
        data_dir=args.data_dir,
        img_size=tuple(args.img_size)
    )
    class_names = data_loader.get_class_names()
    num_classes = len(class_names)

    # 加载模型
    print(f"加载模型权重: {args.weight_path}")
    # 先构建模型结构，再加载权重
    model = build_tssc_model(num_classes=num_classes)
    model.load_weights(args.weight_path)

    # 预处理图像
    print(f"预处理图像: {args.image_path}")
    image = preprocess_image(args.image_path, tuple(args.img_size))

    # 进行预测
    print("进行预测...")
    predictions = model.predict(image)
    pred_probs = predictions[0]
    pred_class_idx = np.argmax(pred_probs)
    pred_class = class_names[pred_class_idx]
    pred_confidence = pred_probs[pred_class_idx] * 100

    # 输出结果
    print("\n预测结果:")
    print(f"输入图像路径: {args.image_path}")
    print(f"预测类别: {pred_class}")
    print(f"置信度: {pred_confidence:.2f}%")

    # 输出前3名预测结果
    print("\nTop 3 预测结果:")
    top3_indices = np.argsort(pred_probs)[-3:][::-1]
    for idx in top3_indices:
        print(f"{class_names[idx]}: {pred_probs[idx] * 100:.2f}%")

    return {
        'class': pred_class,
        'confidence': float(pred_confidence),
        'top3': [(class_names[idx], float(pred_probs[idx] * 100)) for idx in top3_indices]
    }


if __name__ == "__main__":
    args = parse_args()
    predict(args)
