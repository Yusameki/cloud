<div align="center">

# ☁️ cloud

**Predicting the next step of satellite cloud imagery with ConvLSTM**

<sub>**English** · [中文](README.zh-CN.md)</sub>

<sub>Himawari-8 infrared imagery · Keras / TensorFlow · 2017–2019 data</sub>

</div>

---

## What this is

Take a full day of **Himawari-8** infrared cloud imagery (one frame every 30 minutes, 48 frames a day),
and let a ConvLSTM learn to "look at the past 12 hours, predict the next 12".

The model is quite plain:

```
input (48, 74, 99, 1)          ← 48 frames a day, 74×99 after resize
   ↓
ConvLSTM2D (20 filters)        ← Encoder, return_sequences=False, keeps only the last frame's state
   ↓
Lambda: repeat_last_status     ← copies that last state 48 times as the Decoder's input sequence
   ↓
ConvLSTM2D (20 filters)        ← Decoder, return_sequences=True
   ↓
TimeDistributed Conv2D(1)      ← squashes each frame to one channel, sigmoid to [0,1]
   ↓
output (48, 74, 99, 1)
```

> ( ¯꒳¯ ) The Encoder only takes the final frame's hidden state; the Decoder "replays" it 48 times.
> In plain words: **remember this moment, then draw a whole day from memory**.

---

## Contents

- [Files](#files)
- [Data](#data)
- [How to run](#how-to-run)
- [Model details](#model-details)
- [Known issues](#known-issues)

---

## Files

| File | Purpose |
| --- | --- |
| `prepare.py` | Slices the raw tif imagery into `train_sat.npy` / `test_sat.npy` |
| `train.py` | Splits train/val, trains the model, saves checkpoints |
| `libs/datasets.py` | Reads the date table, reads tifs, resizes, packs into npy |
| `libs/model.py` | Model definition (Encoder → repeat → Decoder) |
| `libs/inference_terms.csv` | Date list used for the test set |
| `models/my_model.h5` | Trained weights (`save_weights_only=True`) |

---

## Data

The raw data is a Himawari-8 level-2 product; filenames look like this:

```
NC_H08_{year}{month}{day}_{hour}{minute}_r14_FLDK.02701_02601_tbb_14_4326.tif
```

- `r14` = infrared channel 14 (about 11.2 μm, brightness temperature)
- `4326` = already reprojected onto a WGS84 lat/lon grid
- Original size **994 × 745**

Lay it out like this:

```
datasets/
├── train/
│   ├── 2017/
│   ├── 2018/
│   └── 2019/     ← note: 2019 counts as test
└── test/
```

The check in `datasets.py` is hard-coded:

```python
if year == 2019:
    phase = 'test'
else:
    phase = 'train'
```

**Training set**: 672 consecutive days starting `2017-01-01`.
**Test set**: dates read from `libs/inference_terms.csv` (the `test_date_1` column).

```csv
test_date_1,test_date_2
2019/1/2 0:00,2019/1/2 12:00
2019/1/9 0:00,2019/1/9 12:00
2019/1/16 0:00,2019/1/16 12:00
...
```

One test day every 7 days; `test_date_2` is noon of that day, probably kept for half-day inference.

---

## How to run

### Dependencies

```
python 3.x
tensorflow-gpu >= 2.3
numpy
opencv-python
pandas
pillow
```

> The repo has no `requirements.txt`; the list above was reverse-engineered from the imports.
> TF 2.3 is very old by now (2026) — newer TF should work too, but `ConvLSTM2D`'s
> `recurrent_activation` behaviour has changed before, so double-check it.

### Two steps

```bash
# 1. pack the tifs into npy first (slow, memory-hungry)
python prepare.py

# 2. train
python train.py
```

`prepare.py` passes `10` as the resize factor:

```python
make_sat_dataset(resize=10, phase='train')
make_sat_dataset(resize=10, phase='test')
```

Hence `994/10 ≈ 99` and `745/10 ≈ 74` — that's where the `(48, 74, 99, 1)` in the model comes from.
To change resolution, change this `resize`, but **the model's `input_shape` and the 74/99 inside
`repeat_last_status` have to change with it**, or the shapes won't line up.

---

## Model details

That Lambda in `libs/model.py` is the most important piece in the whole codebase:

```python
def repeat_last_status(x):
    x = tf.reshape(x, (-1, 1, 74, 99, 20))
    copied_x = tf.identity(x)
    for _ in range(47):
        x = tf.concat([x, copied_x], axis=1)
    return x
```

The Encoder output is `(batch, 74, 99, 20)` (`return_sequences=False` swallows the time axis).
This manually reshapes to put back a time axis of length 1, then copies it 47 times to make 48 frames,
so the Decoder can process it as a sequence.

- It uses a `tf.concat` loop, **not `tf.tile`** — same result but slower;
  `tf.tile(x, [1, 48, 1, 1, 1])` would be faster
- `74` / `99` / `20` are all hard-coded; changing resize or filters means changing them here too

Training config:

```python
model.compile(loss="mae", optimizer="adam")
EarlyStopping(monitor='val_loss', patience=10)
ModelCheckpoint(filepath='./models/my_model.h5', monitor='val_loss',
                save_best_only=True, save_weights_only=True)
model.fit(..., batch_size=8, epochs=100)
```

- Loss is **MAE** rather than MSE — for the large smooth regions in cloud imagery, MAE isn't thrown off by extreme pixels
- `save_weights_only=True` → `my_model.h5` holds **weights only, no architecture**;
  to run inference you need `get_model()` first, then `load_weights()`

---

## Known issues

- **`libs/__pycache__/` got committed** (three `.pyc` files) — clean it up sometime
- **No `requirements.txt`**
- In `datasets.py`, `resized_data` is an undefined variable on the first iteration,
  so if **the very first frame fails to load** it raises `UnboundLocalError` (never hit in practice, the data was complete)
- The input sequence is "48 frames a day → predict the same 48 frames", which is autoregressive rather than genuine
  future prediction — probably the goal back then was just to get the shapes working

---

<div align="center">

<sub>A little experiment from back then; the cloud imagery is Himawari-8's (´-ω-`)</sub>

</div>
