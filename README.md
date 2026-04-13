# DiffusionDrive在轻量mini数据集上跑通全流程
本文档基于论文DiffusionDrive（CVPR 2025），在轻量化mini数据集上跑通了全流程。复现内容覆盖服务器配置、环境搭建、数据处理、项目部署及数据集缓存全流程，确保每一步可直接执行，兼顾逻辑完整性与实操性。

核心信息：
- 项目名称：DiffusionDrive（端到端自动驾驶截断扩散模型）
- 核心创新：采用截断（truncated）扩散策略+级联扩散解码器（多头注意力机制），将高斯加噪步骤截断至50步，将去噪步骤从20步压缩至2步，实现实时推理（45 FPS@RTX4090）
- 实验数据集：NAVSIM（基于OpenScene/nuPlan，本次复现使用轻量化mini子集）
- 官方代码仓库：https://github.com/hustvl/DiffusionDrive
## 一、服务器前置配置
### 1.1 系统与硬件版本核查
进入服务器后，首先核查系统、显卡、CUDA及Conda版本，确保符合项目要求：

```bash
# 查看Ubuntu系统版本，推荐使用Ubuntu22.04
lsb_release -a
# 查看NVIDIA显卡型号
grep -i nvidia
# 查看CUDA版本（本次复现过程中使用CUDA 11.8，无则需要自行安装）
nvcc -V
# 查看Conda状态（无则后续安装Miniconda）
conda --version
```

### 1.2 数据解压与合并
openscene_data.tar.gz数据压缩包为作者自己制作的轻量化mini数据集，可以先看DiffusionDrive/download中的download_mini.sh和download_maps.sh，包含1个元数据文件+32个camera数据+32个lidar数据+nuplan_map文件，每个文件大小约为3.3G，这样下载会消耗巨大的时间成功和储存空间，为了轻量化复现，在作者制作的这个压缩包中包含1个元数据文件+1个camera数据+1个lidar数据+nuplan_map文件。但是压缩过后有10G左右，目前还在制作中。

```bash
# 解压数据压缩包，最好先进入到DiffusionDrive项目根目录
tar -zxvf openscene_data.tar.gz
# 进入mini传感器数据目录，然后将数据集整理成标准形式
cd /root/DiffusionDrive/openscene_data/sensor_blobs/mini
# 合并camera文件夹数据到当前目录
rsync -av camera/ ./
# 合并lidar文件夹数据到当前目录
rsync -av lidar/ ./
# 删除合并后的空文件夹
rm -rf camera/ lidar/
# 核查目录结构（应直接显示场景文件夹）
ls
```


