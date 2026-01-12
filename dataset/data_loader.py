from cProfile import label
import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split


class PeaDiseaseDataLoader:
    def __init__(self, data_dir, img_size=(400, 400), batch_size=16):
        """
        道路裂缝数据集加载器
        :param data_dir: 数据集根目录
        :param img_size: 图像尺寸，默认(400, 400)
        :param batch_size: 批次大小
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = self._get_class_names()
        self.num_classes = len(self.class_names)
        self.class_indices = {name: i for i, name in enumerate(self.class_names)}

    def _get_class_names(self):
        """获取类别名称（从文件夹名提取）"""
        class_names = [name for name in os.listdir(self.data_dir)
                       if os.path.isdir(os.path.join(self.data_dir, name))]
        return sorted(class_names)

    def _load_image_paths_and_labels(self):
        """加载所有图像路径和对应的标签"""
        image_paths = []
        labels = []

        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(img_path)
                    labels = self.class_indices[class_name]
                    labels.append(label)

        return np.array(image_paths), np.array(labels)

    def _preprocess_image(self, image_path, label):
        """图像预处理函数"""
        # 读取图像
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        # 调整大小
        image = tf.image.resize(image, self.img_size)
        # 归一化到[0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        # 转换标签为独热编码
        label = tf.one_hot(label, depth=self.num_classes)
        return image, label

    def _augment_image(self, image, label):
        """数据增强函数"""
        # 随机水平翻转
        image = tf.image.random_flip_left_right(image)
        # 随机垂直翻转
        image = tf.image.random_flip_up_down(image)
        # 随机亮度调整
        image = tf.image.random_brightness(image, max_delta=0.1)
        # 随机对比度调整
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        # 随机旋转
        image = tf.keras.layers.RandomRotation(0.1)(tf.expand_dims(image, 0))
        image = tf.squeeze(image, 0)
        return image, label

    def get_datasets(self, val_split=0.1, test_split=0.2, shuffle=True):
        """
        获取训练集、验证集和测试集
        :param val_split: 验证集比例
        :param test_split: 测试集比例
        :param shuffle: 是否打乱数据
        :return: 训练集、验证集、测试集
        """
        # 加载所有图像路径和标签
        image_paths, labels = self._load_image_paths_and_labels()

        # 先划分训练集和临时集（包含验证集和测试集）
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels,
            test_size=val_split + test_split,
            shuffle=shuffle,
            stratify=labels
        )

        # 从临时集中划分验证集和测试集
        val_size = val_split / (val_split + test_split)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=1 - val_size,
            shuffle=shuffle,
            stratify=temp_labels
        )

        # 创建数据集
        train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

        # 预处理
        train_ds = train_ds.map(self._preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(self._preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(self._preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

        # 训练集应用数据增强
        train_ds = train_ds.map(self._augment_image, num_parallel_calls=tf.data.AUTOTUNE)

        # 打乱和批处理
        if shuffle:
            train_ds = train_ds.shuffle(buffer_size=len(train_paths))

        train_ds = train_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds, test_ds

    def get_class_names(self):
        """返回类别名称列表"""
        return self.class_names
