# BCCE-for-Road-Crack-Detection
> 官方TensorFlow/Keras实现 | 论文题目：《BCCE: A Novel Road Crack Detection model Based on Spectral Analysis and Deep Learning Algorithms》  
> 提出BCCE模型，基于TensorFlow 1.14.0与Keras 2.2.4框架，通过“1D光谱转2D图像+双线性卷积+注意力机制+高效分类”流程，实现道路裂缝的高精度识别，总体分类准确率达98.73%，助力公路基础设施智能化监测与维护。  


## 1. 研究背景与模型定位  

道路裂缝是公路基础设施退化的核心表征之一，若未及时检测与维护，会引发路面进一步破损（如坑槽、剥落），增加养护成本与安全风险。传统人工检测依赖视觉巡检，存在效率低、主观性强、难以规模化覆盖的问题；现有深度学习方法多聚焦**2D道路图像**，但缺乏对**1D光谱数据**（更易通过便携式设备快速采集）的有效处理能力。  


本文提出**BCCE（Bilinear CNN-CBAM-Efficient Classifier）** 模型，通过两阶段创新流程突破技术瓶颈：  

1. **光谱-图像转换**：利用便携式光纤光谱仪采集道路裂缝的1D反射光谱，通过`Reshape`等操作将其转换为2D图像数据，解决1D光谱“难以被2D卷积网络直接处理”的问题；  
2. **2D图像分类**：设计融合“双线性卷积神经网络（B-CNN）、卷积块注意力模块（CBAM）、高效分类器（ECH）”的BCCE模型，对转换后的2D图像进行细粒度裂缝分类，兼顾特征表达能力与分类效率。  


## 2. BCCE核心创新点  

### 2.1 双线性卷积神经网络（B-CNN）：增强细粒度特征表达  

B-CNN通过“双线性池化”操作，捕捉图像中**像素对之间的高阶关联**，解决传统CNN对“裂缝细微形态（如网状裂缝的纹理密度、纵向裂缝的延伸方向）”表征不足的问题：  

- 对400×400的裂缝图像，B-CNN能同时关注“裂缝边缘梯度”与“区域纹理分布”，强化横向/纵向/块状/网状裂缝的形态差异；  
- 相比普通CNN，双线性操作使特征维度更丰富，为后续注意力机制提供更细粒度的输入。  


### 2.2 卷积块注意力模块（CBAM）：聚焦裂缝核心区域  

CBAM通过“通道注意力+空间注意力”的串行机制，动态强化裂缝区域的特征权重，抑制背景（如路面纹理、污渍）干扰：  

- **通道注意力**：筛选与“裂缝语义”强相关的特征通道（如边缘检测、纹理描述通道）；  
- **空间注意力**：在图像空间维度突出裂缝的位置与形状，使模型聚焦裂缝的“断裂面、延伸轨迹”等关键区域。  


### 2.3 高效分类器（ECH）：平衡精度与推理速度  

ECH采用“轻量级全连接层+正则化策略”，替代传统复杂分类头：  

- 减少参数量（相比普通分类器减少约40%参数），同时通过Dropout与BatchNormalization防止过拟合；  
- 在GTX 1080显卡上，单张图像分类推理耗时低至15ms，满足“快速巡检”的工程需求。  


## 3. 实验数据集：四类道路裂缝数据集  

### 3.1 数据集概况  

本研究基于**四类道路裂缝数据集**，涵盖不同形态的道路裂缝与对应1D光谱数据，图像尺寸统一为400×400（RGB三通道）：  


| 裂缝类别                | 英文名称              | 图像数量 | 数据来源                     |  
|-------------------------|-----------------------|----------|------------------------------|  
| 横向裂缝                | Transverse crack      | 742      | 便携式光纤光谱仪+实地采集    |  
| 块状裂缝                | Block crack           | 637      | 便携式光纤光谱仪+实地采集    |  
| 网状裂缝                | Alligator crack       | 819      | 便携式光纤光谱仪+实地采集    |  
| 纵向裂缝                | Longitudinal crack    | 953      | 便携式光纤光谱仪+实地采集    |  


### 3.2 数据集获取与结构  

#### 3.2.1 下载方式  

数据集暂未公开，如需研究使用请联系作者获取；公开后将更新百度网盘/Google Drive链接及提取码。  


#### 3.2.3 文件夹组织（解压后放置于项目根目录，结构如下）  

```
road_crack_dataset/  
├── transverse_crack/    # 横向裂缝图像及对应1D光谱文件  
├── block_crack/         # 块状裂缝图像及对应1D光谱文件  
├── alligator_crack/     # 网状裂缝图像及对应1D光谱文件  
└── longitudinal_crack/  # 纵向裂缝图像及对应1D光谱文件  
```  


## 4. 实验环境配置  

### 4.1 依赖与版本  

本项目依赖特定版本的深度学习框架与硬件加速库，需严格匹配以确保兼容性：  


| 组件         | 版本          | 说明                              |  
|--------------|---------------|-----------------------------------|  
| Python       | 3.10          | 开发语言                          |  
| TensorFlow   | 1.14.0        | 深度学习框架（需与CUDA 9.2兼容）|  
| Keras        | 2.2.4         | 高层API（TensorFlow后端）|  
| CUDA         | 9.2           | GPU加速计算平台                   |  
| CuDNN        | 7.0           | NVIDIA深度神经网络加速库          |  
| Anaconda     | 3 5.2.0       | 环境管理工具                      |  
| Spyder       | 3.3.3         | Python集成开发环境（可选，也可终端运行） |  


### 4.2 硬件环境  

| 组件         | 规格                              | 作用                              |  
|--------------|-----------------------------------|-----------------------------------|  
| CPU          | Intel Core i7-9700                | 模型训练/推理的CPU算力支撑        |  
| 内存         | 32GB DDR4                         | 数据集加载与中间变量存储          |  
| GPU          | NVIDIA GeForce GTX 1080（16GB显存） | 加速模型训练与推理（核心依赖）|  
| 操作系统     | Windows 10 x64                    | 运行环境                          |  


### 4.3 环境搭建步骤  

推荐使用Anaconda创建虚拟环境，确保框架与硬件加速库版本匹配：  


```bash  
# 1. 创建并激活虚拟环境  
conda create -n bcce-tf python=3.10  
conda activate bcce-tf  

# 2. 安装TensorFlow 1.14.0（GPU版本，需提前安装CUDA 9.2+CuDNN 7.0）  
pip install tensorflow-gpu==1.14.0  

# 3. 安装Keras 2.2.4（指定TensorFlow后端）  
pip install keras==2.2.4  

# 4. 安装其他依赖库  
pip install numpy~=1.23.5 matplotlib~=3.7.1 opencv-python~=4.7.0.72  
pip install pandas~=2.0.3 scikit-learn~=1.2.2 h5py~=3.9.0  
pip install tqdm~=4.65.0 pillow~=9.4.0  
```  


## 5. 代码使用说明  

### 5.1 光谱数据转2D图像（第一阶段）  

运行`spectrum_to_image.py`脚本，将1D反射光谱转换为2D图像（需确保光谱文件与数据集目录对应）：  


```bash  
python spectrum_to_image.py \  
  --spectrum_dir ./road_crack_dataset \  # 光谱文件根目录  
  --output_image_dir ./processed_2d_images \  # 2D图像输出目录  
  --image_size 400  # 输出图像尺寸（与模型输入匹配）  
```  


### 5.2 模型训练（第二阶段）  

运行`train_bcce.py`脚本，使用转换后的2D图像训练BCCE模型，示例命令：  


```bash  
python train_bcce.py \  
  --image_dir ./processed_2d_images \  # 2D图像目录（来自第一阶段输出）  
  --epochs 50 \                        # 训练轮数  
  --batch_size 32 \                    # 批次大小（根据GPU显存调整，建议16/32）  
  --lr 1e-4 \                          # 初始学习率  
  --save_dir ./weights \               # 模型权重保存目录（.h5格式）  
  --log_dir ./tensorboard_logs \       # TensorBoard日志目录  
  --device GPU  # 训练设备（GPU/CPU，GPU需配置CUDA）  
```  


#### 关键参数说明  

| 参数名         | 含义                                  | 默认值                          |  
|----------------|---------------------------------------|---------------------------------|  
| `--image_dir`  | 2D图像数据集目录（需包含四类裂缝子文件夹） | `./processed_2d_images`         |  
| `--epochs`     | 训练轮数                              | 50                              |  
| `--batch_size` | 批次大小（GTX 1080 16GB建议≤32）| 32                              |  
| `--lr`         | 初始学习率（采用Adam优化器）| 1e-4                            |  
| `--save_dir`   | 模型权重保存目录（自动生成.h5文件）| `./weights`                     |  
| `--device`     | 训练设备（GPU需确保CUDA/CuDNN配置正常） | `GPU`                           |  


### 5.3 模型预测  

使用训练好的`.h5`权重进行单张2D图像预测，运行`predict_bcce.py`脚本：  


```bash  
python predict_bcce.py \  
  --image_path ./examples/transverse_crack_sample.jpg \  # 待预测2D图像路径  
  --weight_path ./weights/best_bcce_model.h5 \           # 预训练模型权重（.h5格式）  
  --device GPU  # 预测设备（GPU/CPU）  
```  


#### 预测输出示例  

```  
输入图像路径：./examples/transverse_crack_sample.jpg  
预测裂缝类别：横向裂缝（Transverse crack）  
分类置信度：0.9927  
推理耗时：14.8ms（GPU，GTX 1080）/ 85.3ms（CPU，i7-9700）  
```  


## 6. 项目文件结构  

```  
BCCE-for-Road-Crack-Detection/  
├── dataset/       # 原始数据集（含1D光谱与分类标签，需联系作者获取）  
├── examples/      # 预测示例图像  
├── models/        # 模型核心实现  
│   ├── bcce_model.py  # BCCE模型定义（整合B-CNN、CBAM、ECH）  
│   ├── b_cnn.py       # 双线性卷积神经网络模块  
│   ├── cbam.py        # 卷积块注意力模块（通道+空间注意力）  
│   └── ech.py         # 高效分类器模块  
├── train.py       # 模型训练脚本（TensorFlow/Keras版）  
├── predict.py     # 模型预测脚本（TensorFlow/Keras版）  
└── README.md      # 项目说明文档（本文档）  
```  


## 7. 已知问题与注意事项  

1. **框架兼容性**：TensorFlow 1.14.0仅支持Python 3.5-3.7原生兼容，但本项目通过特殊依赖适配Python 3.10，若安装失败需优先确保`setuptools`版本为`41.0.0`左右（可通过`pip install setuptools==41.0.0`调整）；  
2. **CUDA版本匹配**：TensorFlow 1.14.0需严格对应**CUDA 9.2 + CuDNN 7.0**，高版本CUDA（如10.x/11.x）会导致框架初始化错误；  
3. **图像尺寸限制**：模型固定输入为400×400×3，若输入图像尺寸不符，需在`predict_bcce.py`中添加resize逻辑（或预处理时统一尺寸）；  
4. **光谱数据格式**：需确保1D光谱文件为`.csv`或`.txt`格式（每行对应一个像素的光谱反射值），否则需修改`spectrum_to_image.py`的读取逻辑；  
5. **训练显存占用**：若使用`batch_size=32`时显存不足（如GTX 1080 8GB版本），可降低`batch_size`至16或8，以牺牲训练速度换取显存兼容。  


## 8. 引用与联系方式  

### 8.1 引用方式  

若本研究对您的工作有帮助，可引用以下格式（论文正式发表后将更新DOI）：  


```bibtex  
@article{bcce_road_crack,  
  title={BCCE: A Novel Road Crack Detection model Based on Spectral Analysis and Deep Learning Algorithms},  
  author={[作者姓名，待发表时补充]},  
  journal={[期刊名称，待录用后补充]},  
  year={2025},  
  note={Manuscript submitted for publication}  
}  
```  


### 8.2 联系方式  

若需数据集、代码协助或学术交流，可通过以下方式联系：  

- 邮箱：changyibu@huuc.edu.cn  
- GitHub Issue：在本仓库提交Issue，1-3个工作日内回复；  
- 学术交流：发送主题为“BCCE-学术交流”的邮件，附研究方向与问题描述，优先回复。
