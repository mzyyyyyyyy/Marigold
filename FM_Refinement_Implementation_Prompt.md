# 实现任务：基于 Flow Matching 的冠层高度图精化器

## 背景

我在做一个遥感任务：从15m分辨率的Landsat影像绘制3m分辨率的冠层高度图（Canopy Height Map, CHM）。

目前已有的实验结果：

- **Baseline 1**: 微调 Depth Anything V2（DAv2），仅用 Landsat 输入 → R² \= 0.80  
- **Baseline 2**: SR4IR（先超分 Landsat 到 3m，再用 DAv2 预测高度）→ R² \= 0.82  
- **Baseline 3**: Marigold 式扩散模型微调 → R² \= 0.73（效果差）  
- **上界**: 用 3m PlanetScope 输入 DAv2 → R² \= 0.88

数据设定：

- 训练时可用：15m Landsat（7波段）、3m PlanetScope（4波段：R/G/B/NIR）、3m CHM 标签  
- 推理时仅可用：15m Landsat  
- Landsat 和 PlanetScope 严格空间对齐

关键发现：

- VAE roundtrip 测试表明 SD 的 VAE 对高度图的编解码质量良好（PSNR≈30, SSIM≈0.86），空间细节无明显丢失  
- Marigold 效果差的原因不是 VAE 瓶颈，而是扩散训练范式（随机噪声+多步去噪）不适合确定性回归任务  
- DAv2 的主要问题是空间细节模糊（15m 输入信息不足）

## 任务目标

在我现有的 Marigold 代码框架基础上，实现 **DAv2 粗预测 \+ Flow Matching 精化器** 的训练和推理 pipeline。我之前的pipeline是train\_mm.py.

## 架构设计

### 整体流程

训练时：

  Landsat → DAv2(冻结) → H\_coarse（粗高度图）

  H\_coarse → VAE 编码 → z\_coarse

  H\_gt     → VAE 编码 → z\_gt

  z\_t \= (1-t) \* z\_coarse \+ t \* z\_gt          \# 隐空间线性插值

  v\_target \= z\_gt \- z\_coarse                   \# 速度目标

  v\_pred \= SD\_UNet(z\_t, t) \+ ControlNet(Landsat, PlanetScope\_or\_null)

  loss \= MSE(v\_pred, v\_target)

推理时：

  Landsat → DAv2(冻结) → H\_coarse

  H\_coarse → VAE 编码 → z\_coarse

  v\_pred \= SD\_UNet(z\_coarse, t=0) \+ ControlNet(Landsat, null\_ps)

  z\_fine \= z\_coarse \+ v\_pred

  H\_fine \= VAE 解码(z\_fine)

### 关键设计选择

1. **在 VAE 隐空间操作**（不在像素空间）：VAE roundtrip 测试已确认隐空间能忠实保留高度图信息。  
     
2. **使用 SD 预训练 UNet 权重初始化**：精化任务的本质是"补充高频空间细节"，SD 的预训练权重中包含丰富的空间细节生成能力（边缘、纹理），这种能力对精化任务有用。  
     
3. **使用 ControlNet 注入条件信号**：  
     
   - ControlNet 分支接收 Landsat（7通道）和 PlanetScope（4通道）作为空间条件  
   - 在 UNet 每一层注入控制信号，浅层注入空间细节，深层注入语义信息  
   - 保持主 UNet 预训练权重不受条件注入干扰

   

4. **Flow Matching 替代扩散训练**：  
     
   - 直线路径：z\_t \= (1-t) \* z\_coarse \+ t \* z\_gt  
   - 训练目标：预测速度 v \= z\_gt \- z\_coarse  
   - 确定性推理，1-4 步即可完成  
   - 不需要噪声 schedule，不需要多步去噪

   

5. **PlanetScope 条件丢弃（Privileged Condition Dropout）**：  
     
   - 训练时以概率 p=0.3 将 PlanetScope 替换为可学习的 null token  
   - 推理时始终使用 null token（因为推理时 PlanetScope 不可用）  
   - 通过权重共享，有 PS 路径学到的空间细节知识隐式迁移到无 PS 路径

   

6. **起点是 DAv2 粗预测而非噪声**：  
     
   - 粗预测已包含正确的全局结构，FM 只需补充高频残差  
   - 路径极短（粗高度图 → 精细高度图，同域精化），远比"噪声 → 高度图"简单

### 网络结构

- **主 UNet**：SD 2.1 的 UNet，用预训练权重初始化  
- **ControlNet**：与主 UNet 编码器结构相同的分支网络，随机初始化，输入通道数改为 11（7 Landsat \+ 4 PlanetScope）  
- **VAE**：SD 2.1 的 VAE，完全冻结  
- **DAv2**：已微调的 Depth Anything V2，完全冻结

### 训练细节

- **时间步采样**：t \~ Uniform(0, 1\)  
- **损失函数**：MSE(v\_pred, v\_target)，可选加边缘损失  
- **优化器**：AdamW, lr=1e-5（UNet）/ 1e-4（ControlNet）  
- **PlanetScope 丢弃概率**：p=0.3（作为超参数，后续消融实验调整）  
- **null token**：可学习参数，shape 与 PlanetScope 输入相同，随训练更新

### 推理细节

- **积分步数**：默认 1 步（单步推理），可选 2-4 步  
- **积分方法**：欧拉法（Euler method）  
- **PlanetScope 条件**：始终使用 null token

## 代码实现要求

1. **请先阅读我现有的代码框架**，理解数据加载、训练循环、模型定义的结构，基于train\_mm.py遵循的代码框架修改即可，尤其是数据加载过程保持不变。  
2. **新建文件而非覆盖**，FM 精化器的模型定义、训练脚本、推理脚本都单独创建  
3. 需要实现的文件：  
   - `depthfm/fm_refiner.py`：FM 精化网络定义（UNet \+ ControlNet \+ FM 训练逻辑）  
   - `train_fm_refiner.py`：训练脚本  
   - `config/fm_refiner.yaml`：配置文件

## 评估指标

- R²（决定系数）  
- MAE（平均绝对误差）  
- RMSE（均方根误差）

