import tensorflow as tf
import argparse
import os
import datetime
from tensorflow.keras import optimizers, metrics, callbacks
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.metrics import f1_score, classification_report
import numpy as np

# 导入自定义模块
from models.TSSC import build_tssc_model
from dataset.data_loader import PeaDiseaseDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='训练道路裂缝识别模型')
    parser.add_argument('--data_dir', type=str, default='./pea_disease_dataset',
                        help='数据集根目录')
    parser.add_argument('--epochs', type=int, default=60,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减系数')
    parser.add_argument('--save_dir', type=str, default='./weights',
                        help='模型权重保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志保存目录')
    parser.add_argument('--device', type=str, default='GPU',
                        choices=['GPU', 'CPU'], help='训练设备')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--img_size', type=int, nargs=2, default=[400, 400],
                        help='图像尺寸')
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置设备
    if args.device == 'GPU' and tf.test.is_gpu_available():
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("使用GPU进行训练")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("使用CPU进行训练")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 加载数据
    print("加载数据集...")
    data_loader = PeaDiseaseDataLoader(
        data_dir=args.data_dir,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size
    )
    train_ds, val_ds, test_ds = data_loader.get_datasets(
        val_split=args.val_split,
        test_split=args.test_split
    )
    num_classes = data_loader.num_classes
    print(f"类别数量: {num_classes}, 类别名称: {data_loader.get_class_names()}")

    # 构建模型
    print("构建TSSC模型...")
    model = build_tssc_model(num_classes=num_classes)
    model.summary()

    # 定义优化器
    optimizer = optimizers.Adam(
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )

    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss=CategoricalCrossentropy(from_logits=False),
        metrics=[
            metrics.CategoricalAccuracy(name='accuracy'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall')
        ]
    )

    # 定义回调函数
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = callbacks.TensorBoard(
        log_dir=os.path.join(args.log_dir, current_time),
        histogram_freq=1
    )

    # 模型保存回调（保存验证集性能最好的模型）
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=os.path.join(args.save_dir, 'best_tssc.h5'),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    # 学习率衰减
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    # 早停策略
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    # 训练模型
    print("开始训练...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[
            tensorboard_callback,
            model_checkpoint,
            lr_scheduler,
            early_stopping
        ]
    )

    # 在测试集上评估
    print("在测试集上评估模型...")
    test_loss, test_acc, test_precision, test_recall = model.evaluate(test_ds, verbose=1)

    # 计算F1分数
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    test_f1 = f1_score(y_true, y_pred, average='macro')

    print(f"测试集结果:")
    print(f"损失: {test_loss:.4f}")
    print(f"准确率: {test_acc:.4f}")
    print(f"精确率: {test_precision:.4f}")
    print(f"召回率: {test_recall:.4f}")
    print(f"宏平均F1: {test_f1:.4f}")

    # 打印详细分类报告
    print("\n分类报告:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=data_loader.get_class_names()
    ))

    # 保存最终模型
    final_model_path = os.path.join(args.save_dir, 'final_tssc.h5')
    model.save(final_model_path)
    print(f"最终模型已保存至: {final_model_path}")


if __name__ == "__main__":
    main()
