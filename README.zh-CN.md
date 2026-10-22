<div align="center">

# ☁️ cloud

**用 ConvLSTM 预测卫星云图的下一步**

<sub>[English](README.md) · **中文**</sub>

<sub>Himawari-8 红外云图 · Keras / TensorFlow · 2017–2019 的数据</sub>

</div>

---

## 这是什么

拿 **Himawari-8** 的一整天空红外云图（每 30 分钟一张，一天 48 张），
让 ConvLSTM 学会「看过去 12 小时，推未来 12 小时」。

模型结构很朴素：

```
输入 (48, 74, 99, 1)          ← 一天 48 帧，resize 后 74×99
   ↓
ConvLSTM2D (20 filters)       ← Encoder，return_sequences=False，只留最后一帧状态
   ↓
Lambda: repeat_last_status    ← 把最后那帧状态复制 48 份，当作 Decoder 的输入序列
   ↓
ConvLSTM2D (20 filters)       ← Decoder，return_sequences=True
   ↓
TimeDistributed Conv2D(1)     ← 逐帧压成单通道，sigmoid 归一化到 [0,1]
   ↓
输出 (48, 74, 99, 1)
```

> ( ¯꒳¯ ) Encoder 只吃最后一帧的隐状态，Decoder 再把它「回放」48 次。
> 说白了就是：**记住此刻，然后凭记忆把一天画出来**。

---

## 目录

- [文件说明](#文件说明)
- [数据](#数据)
- [怎么跑](#怎么跑)
- [模型细节](#模型细节)
- [已知问题](#已知问题)

---

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `prepare.py` | 把 tif 原始云图切成 `train_sat.npy` / `test_sat.npy` |
| `train.py` | 切训练/验证集，训练模型，存 checkpoint |
| `libs/datasets.py` | 读日期表、读 tif、resize、打包成 npy |
| `libs/model.py` | 模型定义（Encoder → repeat → Decoder） |
| `libs/inference_terms.csv` | 测试集用的日期清单 |
| `models/my_model.h5` | 训练好的权重（`save_weights_only=True`） |

---

## 数据

原始数据是 Himawari-8 的二级产品，文件名长这样：

```
NC_H08_{year}{month}{day}_{hour}{minute}_r14_FLDK.02701_02601_tbb_14_4326.tif
```

- `r14` = 红外通道 14（约 11.2 μm，亮温）
- `4326` = 已重投影到 WGS84 经纬度网格
- 原始尺寸 **994 × 745**

放成这个结构：

```
datasets/
├── train/
│   ├── 2017/
│   ├── 2018/
│   └── 2019/     ← 注意：2019 年算 test
└── test/
```

`datasets.py` 里是硬编码的判断：

```python
if year == 2019:
    phase = 'test'
else:
    phase = 'train'
```

**训练集**：从 `2017-01-01` 起连续 672 天。
**测试集**：读 `libs/inference_terms.csv` 里的日期（`test_date_1` 列）。

```csv
test_date_1,test_date_2
2019/1/2 0:00,2019/1/2 12:00
2019/1/9 0:00,2019/1/9 12:00
2019/1/16 0:00,2019/1/16 12:00
...
```

每 7 天取一个测试日，`test_date_2` 是当天中午，大概是留着做半天推理的。

---

## 怎么跑

### 依赖

```
python 3.x
tensorflow-gpu >= 2.3
numpy
opencv-python
pandas
pillow
```

> 原仓库没有 `requirements.txt`，上面这份是从 import 反推的。
> TF 2.3 现在（2026 年）已经很旧了，新版 TF 应该也能跑，但 `ConvLSTM2D` 的
> `recurrent_activation` 行为有过变化，注意对一下。

### 两步

```bash
# 1. 先把 tif 打包成 npy（很慢，会吃内存）
python prepare.py

# 2. 训练
python train.py
```

`prepare.py` 里 resize 传的是 `10`：

```python
make_sat_dataset(resize=10, phase='train')
make_sat_dataset(resize=10, phase='test')
```

于是 `994/10 ≈ 99`、`745/10 ≈ 74` —— 这就是模型里那个 `(48, 74, 99, 1)` 的来历。
想改分辨率就改这个 `resize`，但**模型里的 `input_shape` 和 `repeat_last_status`
里的 74/99 也得一起改**，不然形状对不上。

---

## 模型细节

`libs/model.py` 里那个 Lambda 是整套代码里最关键的一块：

```python
def repeat_last_status(x):
    x = tf.reshape(x, (-1, 1, 74, 99, 20))
    copied_x = tf.identity(x)
    for _ in range(47):
        x = tf.concat([x, copied_x], axis=1)
    return x
```

Encoder 输出是 `(batch, 74, 99, 20)`（`return_sequences=False`，时间维被吞了），
这里手动 reshape 补回一个长度 1 的时间维，再复制 47 次凑成 48 帧，
Decoder 才能按序列处理。

- 用的是 `tf.concat` 循环拼接，**不是 `tf.tile`** —— 效果一样但慢一些，
  改成 `tf.tile(x, [1, 48, 1, 1, 1])` 会更快
- `74` / `99` / `20` 都是写死的，改 resize 或 filters 必须同步改这里

训练配置：

```python
model.compile(loss="mae", optimizer="adam")
EarlyStopping(monitor='val_loss', patience=10)
ModelCheckpoint(filepath='./models/my_model.h5', monitor='val_loss',
                save_best_only=True, save_weights_only=True)
model.fit(..., batch_size=8, epochs=100)
```

- 损失用 **MAE** 而不是 MSE，云图这种大面积平滑的区域，MAE 不容易被极端像素带偏
- `save_weights_only=True` → `my_model.h5` 里**只有权重没有结构**，
  要推理得先 `get_model()` 再 `load_weights()`

---

## 已知问题

- **`libs/__pycache__/` 被提交进来了**（三个 `.pyc`），下次清理一下
- **没有 `requirements.txt`**
- `datasets.py` 里 `resized_data` 在第一次循环时是未定义变量，
  如果**第一帧就读失败**会直接 `UnboundLocalError`（实际跑的时候没遇到，因为数据齐全）
- 输入序列是「一天 48 帧 → 预测同样 48 帧」，本质是自回归而非真正的未来预测，
  当年大概是先跑通形状再说

---

<div align="center">

<sub>当年的一点小实验，云图数据是 Himawari-8 的 (´-ω-`)</sub>

</div>
