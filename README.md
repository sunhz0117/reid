# Reid Experiments



## 数据集与模型

单模态RGB数据集：

* Market1501
* <https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view>

跨模态RGB-IR数据集：

* sysu
* <https://drive.google.com/open?id=181K9PQGnej0K5xNX9DRBDPAf3K9JosYk>

模型

* 百度云：https://pan.baidu.com/s/1nS-JtmPa2ramtlXSghbwkw
* 提取码：am1w



## 评价指标

指标：RANK-N 命中率，CMC曲线，mAP，mINP指标

指标说明：<https://blog.csdn.net/qq_29159273/article/details/104375440>

mINP指标参考论文：<https://arxiv.org/abs/2001.04193>



## 单模态RGB Reid

baseline：MGN

MGN参考论文：<https://arxiv.org/abs/1804.01438>

文件夹：reid/mgn



结合行人属性进行联合训练（目的：sysu数据集没有现成的行人属性，需要训练一个RGB的属性模型inference获得sysu数据集的RGB图像部分的行人属性）

Market1501属性数据集与标签说明：<https://github.com/vana77/Market-1501_Attribute>（git中已保存，无需额外下载）

损失函数：ID（CrossEntropy）+ Triplet + Attribute（CrossEntropy）

evaluate方法：获得多层feature层进行拼接（256 * 8 = 2048）计算L2距离排序（rank / re-rank）

re-rank方法参考论文：<https://arxiv.org/abs/2001.04193>



command：

```RGB
# enter directory
cd reid/mgn

# command
bash ./run.sh ${gpu} ${load_model} ${save_model_directory} ${--test_only if only for test}

# MGN
train: bash ./run.sh 1 '' xxx
test: bash ./run.sh 1 experiment/mgn/model/model_best.pt mgn --test_only
result:
[INFO] mAP: 0.8129 rank(1,3,5,10,20): 0.8596|0.9044|0.9186|0.9365|0.9584 attr: 0.3599 (Best: 0.8129 @epoch 50)

# MGN + Attribute
train: bash ./run.sh 1 '' mgn_attr --loss 1*ID+2*Triplet+1*Attribute
test: bash ./run.sh 1 experiment/mgn_attr/model/model_best.pt mgn_attr --test_only
result:
[INFO] mAP: 0.8115 rank(1,3,5,10,20): 0.8575|0.9008|0.9136|0.9409|0.9567 attr: 0.7010 (Best: 0.8115 @epoch 50)

# Note: ${gpu} means the gpu cuda device id, only use 1 gpu is enough
# GPU Memory: 11G
# If you have less GPU memory, please reduce the parameters ${args.batchid} for train and ${args.batchtest} for evaluate in *option.py*
```



## 多模态RGB-IR Reid

baseline：Resnet50 + 双头网络（embed） + AGW方法

参考论文：<https://arxiv.org/abs/2001.04193>

双头网络参考论文：<https://www.researchgate.net/publication/326203460_Visible_Thermal_Person_Re-Identification_via_Dual-Constrained_Top-Ranking>

文件夹：reid/Cross-Modal-Re-ID-baseline



RGB模型inference得到sysu attribute：**sysu_attribute.txt**

Attention机制Non-Local参考论文：<https://arxiv.org/abs/1711.07971>

损失函数：ID（CrossEntropy）+ Triplet（多个变种） + Attribute（CrossEntropy）

Triplet变种：

1、传统Triplet：base中使用， 采用trihard最远正样本与最近负样本

2、Weighted Triplet：https://arxiv.org/abs/2001.04193，agw中使用

3、Cross-Triplet：仅计算不同模态的Triplet，每张图片计算与自己不同模态的最远正样本与最近负样本，crosstri中使用

4、Rank-loss：排序损失，考虑了同模态的ap，an和不同模态的ap，an，对负样本对采用加权，loss取距离均值而不是最难样本

Attribute结合行人属性进行联合训练：

1、attr_39：len(fc) = 39，带颜色

2、attr_22：len(fc) = 22，不带颜色



Evaluate方法：1次evaluation每个行人每个相机下随机采样一张图片（因为一个行人在gallery中有多个图像，为multi-shot方式，但每个行人的图片数量基本维持相等），构成gallery集，evaluate采样10次取平均值



command：

```RGB-IR
# enter directory
cd reid/Cross-Modal-Re-ID-baseline

# command
bash ./run.sh ${gpu} ${method} ${load_model} ${save_model_directory} ${mode:train/eval} ${attr_num:39(with color) / 22(with color) / 0(do not train attribute)}

# base method
train: bash ./run.sh 2 base '' checkpoint/xxx train 0
test: bash ./run.sh 2 base sysu_base_p4_n8_lr_0.1_seed_0_best.t checkpoint/base eval 0
result：
IR --> RGB
POOL:   Rank-1: 50.88% | Rank-5: 77.95% | Rank-10: 86.98%| Rank-20: 93.99%| mAP: 49.81%| mINP: 36.82%
FC:   Rank-1: 48.08% | Rank-5: 76.64% | Rank-10: 87.32%| Rank-20: 95.36%| mAP: 48.29%| mINP: 34.92%
RGB --> IR
POOL:   Rank-1: 21.86% | Rank-5: 48.75% | Rank-10: 63.65%| Rank-20: 78.83%| mAP: 28.59%| mINP: 25.11%
FC:   Rank-1: 19.55% | Rank-5: 45.47% | Rank-10: 60.70%| Rank-20: 77.50%| mAP: 26.36%| mINP: 21.27%

# agw non—local method
train: bash ./run.sh 2 agw '' checkpoint/xxx train 0
test: bash ./run.sh 2 agw sysu_agw_p4_n8_lr_0.1_seed_0_best.t checkpoint/non_local eval 0
result：
IR --> RGB
POOL:   Rank-1: 48.48% | Rank-5: 75.93% | Rank-10: 85.62%| Rank-20: 93.22%| mAP: 47.35%| mINP: 34.65%
FC:   Rank-1: 44.95% | Rank-5: 74.99% | Rank-10: 85.70%| Rank-20: 93.72%| mAP: 45.52%| mINP: 34.04%
RGB --> IR
POOL:   Rank-1: 21.17% | Rank-5: 47.36% | Rank-10: 61.87%| Rank-20: 76.59%| mAP: 27.02%| mINP: 21.83%
FC:   Rank-1: 19.33% | Rank-5: 46.32% | Rank-10: 61.71%| Rank-20: 77.75%| mAP: 26.04%| mINP: 21.05%

# agw + attr_39
train: bash ./run.sh 2 agw '' checkpoint/xxx train 39
test: bash ./run.sh 2 agw sysu_agw_p4_n8_lr_0.1_seed_0_best.t checkpoint/attr eval 39
result：
IR --> RGB
POOL:   Rank-1: 44.13% | Rank-5: 72.93% | Rank-10: 82.78%| Rank-20: 90.64%| mAP: 43.74%| mINP: 30.99%
FC:   Rank-1: 44.70% | Rank-5: 74.09% | Rank-10: 84.20%| Rank-20: 92.57%| mAP: 45.14%| mINP: 33.02%
RGB --> IR
POOL:   Rank-1: 17.11% | Rank-5: 43.00% | Rank-10: 58.03%| Rank-20: 73.78%| mAP: 23.22%| mINP: 18.34%
FC:   Rank-1: 15.64% | Rank-5: 41.09% | Rank-10: 57.30%| Rank-20: 74.38%| mAP: 22.73%| mINP: 20.21% 

# agw + attr_22
train: bash ./run.sh 2 agw '' checkpoint/xxx train 22
test: bash ./run.sh 2 agw sysu_agw_p4_n8_lr_0.1_seed_0_best.t checkpoint/attr22 eval 22
result：
IR --> RGB
POOL:   Rank-1: 46.02% | Rank-5: 74.04% | Rank-10: 83.58%| Rank-20: 91.55%| mAP: 45.78%| mINP: 34.15%
FC:   Rank-1: 45.33% | Rank-5: 74.34% | Rank-10: 85.09%| Rank-20: 93.67%| mAP: 46.24%| mINP: 35.54%
RGB --> IR
POOL:   Rank-1: 19.46% | Rank-5: 46.53% | Rank-10: 62.26%| Rank-20: 77.99%| mAP: 26.50%| mINP: 22.61%
FC:   Rank-1: 18.27% | Rank-5: 44.71% | Rank-10: 60.83%| Rank-20: 77.72%| mAP: 25.58%| mINP: 21.54%

# base + attr_22
train: bash ./run.sh 2 base '' checkpoint/xxx train 22
test: bash ./run.sh 2 base sysu_base_p4_n8_lr_0.1_seed_0_best.t checkpoint/base_attr22 eval 22
result：
IR --> RGB
POOL:   Rank-1: 47.88% | Rank-5: 74.81% | Rank-10: 83.86%| Rank-20: 91.89%| mAP: 46.29%| mINP: 32.24%
FC:   Rank-1: 45.96% | Rank-5: 74.84% | Rank-10: 85.82%| Rank-20: 94.48%| mAP: 46.19%| mINP: 34.06%
RGB --> IR
POOL:   Rank-1: 19.57% | Rank-5: 48.00% | Rank-10: 63.17%| Rank-20: 78.11%| mAP: 27.08%| mINP: 23.09%
FC:   Rank-1: 19.19% | Rank-5: 46.68% | Rank-10: 63.05%| Rank-20: 79.65%| mAP: 26.45%| mINP: 22.61%

# non_local + crosstri_loss
train: bash ./run.sh 2 crosstri '' checkpoint/xxx train 22
test: bash ./run.sh 2 crosstri sysu_crosstri_attr22_p4_n8_lr_0.1_seed_0_best.t checkpoint/crosstri_attr22 eval 22
result：
IR --> RGB
POOL:   Rank-1: 41.49% | Rank-5: 70.58% | Rank-10: 82.08%| Rank-20: 90.71%| mAP: 42.80%| mINP: 31.18%
FC:   Rank-1: 40.14% | Rank-5: 71.05% | Rank-10: 82.96%| Rank-20: 92.12%| mAP: 41.40%| mINP: 29.30%
RGB --> IR
POOL:   Rank-1: 14.39% | Rank-5: 36.50% | Rank-10: 51.23%| Rank-20: 68.24%| mAP: 20.64%| mINP: 17.76%
FC:   Rank-1: 12.72% | Rank-5: 35.06% | Rank-10: 50.04%| Rank-20: 67.30%| mAP: 19.15%| mINP: 15.23%

# non_local + rank_loss （best for IR --> RGB）
train: bash ./run.sh 2 rank '' checkpoint/xxx train 22
test: bash ./run.sh 2 rank sysu_rank_attr22_p4_n8_lr_0.1_seed_0_best.t checkpoint/rank_attr22 eval 22
result：
IR --> RGB
POOL:   Rank-1: 52.72% | Rank-5: 80.31% | Rank-10: 89.05%| Rank-20: 95.44%| mAP: 52.16%| mINP: 38.66%
FC:   Rank-1: 32.96% | Rank-5: 63.91% | Rank-10: 75.47%| Rank-20: 88.19%| mAP: 35.45%| mINP: 24.74%
RGB --> IR
POOL:   Rank-1: 16.93% | Rank-5: 42.57% | Rank-10: 57.86%| Rank-20: 74.04%| mAP: 23.58%| mINP: 19.74%
FC:   Rank-1: 9.81% | Rank-5: 31.61% | Rank-10: 46.70%| Rank-20: 64.55%| mAP: 16.97%| mINP: 14.01% 

# Note: ${gpu} means the gpu cuda device id, only use 1 gpu is enough
# GPU Memory: 11G
# If you have less GPU memory, please reduce the parameters ${args.batch-size} for train in *train.py* and ${args.test-batch} for evaluate in *train.py* and *test.py*
```
