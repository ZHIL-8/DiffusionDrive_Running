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
openscene_data.tar.gz数据压缩包为作者自己制作的轻量化mini数据集，可以先看DiffusionDrive/download中的download_mini.sh和download_maps.sh，包含1个元数据文件+32个camera数据+32个lidar数据+nuplan_map文件，每个文件大小约为3.3G，这样下载会消耗巨大的时间成功和储存空间，为了轻量化复现，在作者制作的这个压缩包中包含1个元数据文件+1个camera数据+1个lidar数据+nuplan_map文件。但是压缩过后有10G左右，所以还是建议根据教程的方法自行下载然后解压整理成标准目录。

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

整理好的数据目录结构如下：

```bash
$DiffusionDrive/openscene_data                
├─ navsim_logs/
│  └─ mini/                            # mini集元数据（.pkl）
├─ sensor_blobs/
│  └─  mini/                            # mini集传感器数据（图像、点云）
├─ maps/                               # nuPlan地图
│  ├─ sg-one-north/
│  ├─ us-ma-boston/
│  ├─ us-nv-las-vegas-strip/
│  └─ us-pa-pittsburgh-hazelwood/
└─ pretrained/
   ├─ resnet34.bin                     # ResNet-34预训练权重
   └─ kmeans_navsim_traj_20.npy       # 20条Anchor
```

## 二、基础环境安装

### 2.1 安装Miniconda3（如果没有的话）

```bash
# 进入根目录
cd ~
# 下载Miniconda安装包
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# 给安装包添加执行权限
chmod +x Miniconda3-latest-Linux-x86_64.sh
# 静默安装（不弹出交互界面）
./Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
# 初始化Conda
~/miniconda3/bin/conda init bash
# 生效Conda配置，生效后前面会有熟悉的(base)
source ~/.bashrc
```

### 2.2 DiffusionDrive环境配置
```bash
cd /root
git clone https://github.com/hustvl/DiffusionDrive.git
# 如果网络问题克隆不成功最好使用清华镜像或者win上下载了传输进去
# 进入DiffusionDrive项目目录
cd DiffusionDrive
# 接受 main 频道的服务条款
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
# 接受 r 频道的服务条款
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n navsim python=3.9 -y
conda activate navsim
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -e .
# 安装 DiffusionDrive 额外依赖，但是这里要注意版本问题，如果直接这样安装最新版的diffuser可能和xpu不兼容
pip install diffusers einops
```

### 2.3 设置项目环境变量

```bash
# 追加环境变量到.bashrc
echo 'export NAVSIM_DEVKIT_ROOT="/root/DiffusionDrive"' >> ~/.bashrc
echo 'export OPENSCENE_DATA_ROOT="/root/DiffusionDrive/openscene_data"' >> ~/.bashrc
echo 'export NAVSIM_EXP_ROOT="/root/DiffusionDrive/experiments"' >> ~/.bashrc
echo 'export NUPLAN_MAPS_ROOT="/root/nuplan/dataset/maps"' >> ~/.bashrc

# 立即生效环境变量
source ~/.bashrc

# 创建数据和实验目录，实际上就是在/root/DiffusionDrive下创建两个文件夹
# openscene_data（用于存放数据集，如果下载了压缩包就不用管）和experiments（用于存放缓存、训练和推理的结果）
mkdir -p $OPENSCENE_DATA_ROOT $NAVSIM_EXP_ROOT
```

| 环境变量 | 用途 |
|----------|------|
| `NAVSIM_DEVKIT_ROOT` | 项目代码根目录 |
| `OPENSCENE_DATA_ROOT` | 原始数据 (navsim_logs + sensor_blobs + maps) |
| `NAVSIM_EXP_ROOT` | 缓存后的数据、训练后的权重、推理的得分结果 |
| `NUPLAN_MAPS_ROOT` | 下载的地图存放地址 |

## 三、轻量化mini数据集下载

### 3.1 数据集下载及整理
使用aria2c多线程下载，仅下载必需的1个元数据、1个相机分卷和1个雷达分卷，节约时间和存储空间：

```bash
# 安装多线程下载工具aria2c
apt install aria2 -y
# 进入DiffusionDrive项目根目录
cd /root/DiffusionDrive
#  下载元数据
aria2c -x 16 -s 16 https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_mini.tgz
# 下载camera数据分卷
aria2c -x 16 -s 16 https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_0.tgz
# 下载lidar数据分卷
aria2c -x 16 -s 16 https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_mini_lidar/openscene_sensor_mini_lidar_0.tgz
```

需要注意的是数据集下载完成后可能压缩包名称会变成哈希值乱码，一定要先分清camera和lidar的数据然后改成下载链接中的标准名字。然后再解压出来，先移动到OPENSCENE_DATA_ROOT目录下，等待后两步都下载完成后一起整理。

### 3.2 nuPlan地图文件配置
项目依赖nuPlan地图，需下载并配置软链接，确保项目能正常读取地图：

```bash
cd $NAVSIM_DEVKIT_ROOT/download/
# 下载地图文件
bash download_maps.sh
# 移动地图到OpenScene数据目录
mv /root/DiffusionDrive/download/maps $OPENSCENE_DATA_ROOT/
# 创建nuplan地图软链接（必需步骤，否则地图读取失败）
mkdir -p ~/nuplan/dataset
ln -s $OPENSCENE_DATA_ROOT/maps ~/nuplan/dataset/maps
# 验证软链接是否生效（箭头指向正确路径即成功）
ls -l ~/nuplan/dataset/
# 验证地图文件是否存在
ls ~/nuplan/dataset/maps/
```

### 3.3 Resnet-34和20个聚类Anchor下载
下载ResNet-34预训练权重和K-means聚类锚轨迹（20个锚点），用于扩散模型：

```bash
# 创建预训练权重目录
mkdir -p $OPENSCENE_DATA_ROOT/pretrained
# 下载ResNet-34预训练权重
wget -O $OPENSCENE_DATA_ROOT/pretrained/resnet34.bin \
    "https://huggingface.co/timm/resnet34.a1_in1k/resolve/main/pytorch_model.bin"
# 下载K-means聚类锚轨迹（论文中使用的20个锚点）
wget -O $OPENSCENE_DATA_ROOT/pretrained/kmeans_navsim_traj_20.npy \
"https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_88p1_PDMS_Eval_file/kmeans_navsim_traj_20.npy"
```

然后到此一定要整理成1.2中的标准格式！！！并且要注意在`DiffusionDrive/openscene_data/sensor_blobs/mini`下的数据不要区分成camera和lidar，要将这两个的数据都整理进对应名称的以场景为名字的文件夹中，上述1.2中的整理其实就是在进行这个操作。

并且在Resnet-34和20个聚类Anchor下载完成后还需要修改一下代码文件中的配置路径（注意要写绝对路径），打开 navsim/agents/diffusiondrive/transfuser_config.py，修改第 18-19 行：

```bash
bkb_path: str = "/root/DiffusionDrive/openscene_data/pretrained/resnet34.bin"
plan_anchor_path: str = "/root/DiffusionDrive/openscene_data/pretrained/kmeans_navsim_traj_20.npy"
```

## 四、数据集缓存
### 4.1 修改缓存配置文件
因仅下载OpenScene mini子集的部分数据，需修改navmini.yaml配置文件，指定实际下载的场景log名称：
`/root/DiffusionDrive/navsim/planning/script/config/common/train_test_split/scene_filter/navmini.yaml`

```yaml
max_scenes: null
log_names:
  - "2021.06.28.16.29.11_veh-38_01415_01821"
  - "2021.10.11.02.57.41_veh-50_01522_02088"
# 注：log_names需填写实际下载的数据集log名称，若下载的log不同，需对应修改
```

### 4.2 数据集缓存
完成所有配置后，执行数据集缓存脚本，为后续模型训练和推理做准备：

```bash
python navsim/planning/script/run_dataset_caching.py \
    agent=diffusiondrive_agent \
    experiment_name=mini_training \
    train_test_split=navmini \
    cache_path="${NAVSIM_EXP_ROOT}/mini_training_cache"
```

## 五、启动训练

```bash
conda activate navsim
cd DiffusionDrive

python navsim/planning/script/run_training.py \
    agent=diffusiondrive_agent \
    experiment_name=training_diffusiondrive_agent \
    train_test_split=navtrain \
    split=trainval \
    trainer.params.max_epochs=10 \
    cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
    use_cache_without_dataset=True \
    force_cache_computation=False
```

训练参数：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `trainer.params.max_epochs` | 100 | 最大训练轮数 |
| `trainer.params.precision` | 16-mixed | 混合精度训练，节省显存 |
| `dataloader.params.batch_size` | 64 | batch |
| `dataloader.params.num_workers` | 4 | 数据加载进程数 |
| `use_cache_without_dataset` | False | 设为True跳过SceneLoader，纯缓存读取 |
| `agent.lr` | 6e-4 | 学习率 |
| `seed` | 0 | 随机种子 |

训练后的权重保存在`$NAVSIM_EXP_ROOT/training_diffusiondrive_agent/lightning_logs/version_X/checkpoints/epoch=0-step=XXX.ckpt`
训练的日志保存在`$NAVSIM_EXP_ROOT/training_diffusiondrive_agent/lightning_logs/version_X/events.out.tfevents.*`

## 六、测试模型
因为作者目前仅仅是在测试demo的阶段，为了节约时间和存储空间，没有下载专门的测试数据。期待对这项工作的SOTA得分能够进行进一步的复现。

## 七、推荐阅读代码快速上手
1. `run_training.py` — 模型训练时使用的代码文件，理解训练的流程
2. `abstract_agent.py` — 定义NAVSIM环境中所有agent的抽象基类,规范agent必须实现的核心接口和方法
3. `diffusiondrive_agent.yaml` — 理解配置如何映射到代码
4. `diffusiondrive/transfuser_model_v2.py` — 模型架构（重点）
5. `diffusiondrive/modules/blocks.py` — GridSampleCrossBEVAttention 等核心组件
6. `diffusiondrive/transfuser_loss.py` — 多目标融合损失
7. `diffusiondrive/transfuser_features.py` — 特征/目标构建
8. `planning/training/dataset.py` — 数据加载与缓存
   












