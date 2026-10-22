<div align="center">

# ☁️ cloud

**ConvLSTM で衛星の雲画像の次の一手を予測する**

<sub>[English](README.md) · **日本語** · [中文](README.zh-CN.md)</sub>

<sub>Himawari-8 の赤外画像 · Keras / TensorFlow · 2017–2019 のデータ</sub>

</div>

---

## これは何か

**Himawari-8** の一日分の赤外雲画像（30 分ごとに 1 枚、1 日 48 枚）を使って、
ConvLSTM に「過去 12 時間を見て、未来 12 時間を推す」ことを覚えさせます。

モデル構造はとても素朴です：

```
入力 (48, 74, 99, 1)          ← 1 日 48 フレーム、resize 後 74×99
   ↓
ConvLSTM2D (20 filters)       ← Encoder、return_sequences=False、最後のフレームの状態だけ残す
   ↓
Lambda: repeat_last_status    ← その最後の状態を 48 個に複製して、Decoder の入力系列にする
   ↓
ConvLSTM2D (20 filters)       ← Decoder、return_sequences=True
   ↓
TimeDistributed Conv2D(1)     ← フレームごとに 1 チャンネルへ圧縮、sigmoid で [0,1] に正規化
   ↓
出力 (48, 74, 99, 1)
```

> ( ¯꒳¯ ) Encoder は最後のフレームの隠れ状態だけを食べて、Decoder がそれを 48 回「再生」します。
> つまり：**この瞬間を覚えて、記憶だけで一日を描く**。

---

## 目次

- [ファイル構成](#ファイル構成)
- [データ](#データ)
- [動かし方](#動かし方)
- [モデルの詳細](#モデルの詳細)
- [既知の問題](#既知の問題)

---

## ファイル構成

| ファイル | 役割 |
| --- | --- |
| `prepare.py` | tif の元画像を `train_sat.npy` / `test_sat.npy` に切る |
| `train.py` | 訓練/検証に分割して学習、checkpoint を保存 |
| `libs/datasets.py` | 日付表を読み、tif を読み、resize して npy に固める |
| `libs/model.py` | モデル定義（Encoder → repeat → Decoder） |
| `libs/inference_terms.csv` | テスト用の日付リスト |
| `models/my_model.h5` | 学習済みの重み（`save_weights_only=True`） |

---

## データ

元データは Himawari-8 のレベル 2 プロダクトで、ファイル名はこうなっています：

```
NC_H08_{year}{month}{day}_{hour}{minute}_r14_FLDK.02701_02601_tbb_14_4326.tif
```

- `r14` = 赤外チャンネル 14（約 11.2 μm、輝度温度）
- `4326` = WGS84 の経緯度グリッドに再投影済み
- 元のサイズは **994 × 745**

この構造で置きます：

```
datasets/
├── train/
│   ├── 2017/
│   ├── 2018/
│   └── 2019/     ← 注意：2019 年は test 扱い
└── test/
```

`datasets.py` の中はハードコードされた判定です：

```python
if year == 2019:
    phase = 'test'
else:
    phase = 'train'
```

**訓練データ**：`2017-01-01` から連続 672 日。
**テストデータ**：`libs/inference_terms.csv` の日付（`test_date_1` 列）を読みます。

```csv
test_date_1,test_date_2
2019/1/2 0:00,2019/1/2 12:00
2019/1/9 0:00,2019/1/9 12:00
2019/1/16 0:00,2019/1/16 12:00
...
```

7 日ごとにテスト日を 1 つ取り、`test_date_2` はその日の正午。たぶん半日分の推論用に残してあったのでしょう。

---

## 動かし方

### 依存関係

```
python 3.x
tensorflow-gpu >= 2.3
numpy
opencv-python
pandas
pillow
```

> 元のリポジトリには `requirements.txt` がなかったので、上の一覧は import から逆算したものです。
> TF 2.3 はもう（2026 年時点で）かなり古いです。新しい TF でも動くはずですが、
> `ConvLSTM2D` の `recurrent_activation` の挙動が変わったことがあるので、そこは確認してください。

### 2 ステップ

```bash
# 1. まず tif を npy に固める（とても遅いしメモリを食います）
python prepare.py

# 2. 学習
python train.py
```

`prepare.py` で resize に渡しているのは `10`：

```python
make_sat_dataset(resize=10, phase='train')
make_sat_dataset(resize=10, phase='test')
```

すると `994/10 ≈ 99`、`745/10 ≈ 74` —— これがモデル内の `(48, 74, 99, 1)` の由来です。
解像度を変えたいならこの `resize` を変えますが、**モデル側の `input_shape` と
`repeat_last_status` の中の 74/99 も一緒に変えないと**、形が合いません。

---

## モデルの詳細

`libs/model.py` のあの Lambda が、コード全体で一番肝の部分です：

```python
def repeat_last_status(x):
    x = tf.reshape(x, (-1, 1, 74, 99, 20))
    copied_x = tf.identity(x)
    for _ in range(47):
        x = tf.concat([x, copied_x], axis=1)
    return x
```

Encoder の出力は `(batch, 74, 99, 20)` です（`return_sequences=False` なので時間次元が潰れている）。
ここで手動で reshape して長さ 1 の時間次元を戻し、さらに 47 回複製して 48 フレームに揃えます。
そうして初めて Decoder が系列として扱えます。

- `tf.concat` のループで繋いでいて、**`tf.tile` ではありません** —— 効果は同じですが少し遅いので、
  `tf.tile(x, [1, 48, 1, 1, 1])` にすると速くなります
- `74` / `99` / `20` はすべてベタ書きなので、resize や filters を変えたらここも直す必要があります

学習設定：

```python
model.compile(loss="mae", optimizer="adam")
EarlyStopping(monitor='val_loss', patience=10)
ModelCheckpoint(filepath='./models/my_model.h5', monitor='val_loss',
                save_best_only=True, save_weights_only=True)
model.fit(..., batch_size=8, epochs=100)
```

- 損失は MSE ではなく **MAE**。雲画像のような広く滑らかな領域では、MAE のほうが極端な画素に引っ張られにくいです
- `save_weights_only=True` → `my_model.h5` には**重みだけで構造は入っていません**。
  推論するには先に `get_model()` してから `load_weights()` します

---

## 既知の問題

- **`libs/__pycache__/` がコミットされてしまっている**（`.pyc` 3 つ）。今度掃除します
- **`requirements.txt` がない**
- `datasets.py` の `resized_data` は最初のループでは未定義の変数なので、
  **最初のフレームで読み込みに失敗すると**そのまま `UnboundLocalError` になります
  （実際にはデータが揃っていたので遭遇しませんでした）
- 入力系列は「1 日 48 フレーム → 同じ 48 フレームを予測」で、本質的には自己回帰であって本当の未来予測ではありません。
  当時はたぶん、まず形を通すのが先だったのでしょう

---

<div align="center">

<sub>当時の小さな実験です。雲画像データは Himawari-8 のものです (´-ω-`)</sub>

</div>
