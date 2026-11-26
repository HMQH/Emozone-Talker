# Pretrained weights

Download `emozone-talker-weights-iter10000.zip` from the
[pretrained weights release](https://github.com/HMQH/Emozone-Talker/releases/tag/weights-v1).
Extract it from the repository root to create the directories expected by `render.py`.

```text
weights/<subject>/
  point_cloud/iteration_10000/
    point_cloud.ply
    deformation.pth
    deformation_table.pth
    deformation_accum.pth
```

## Included subjects

| Dataset | Subjects |
| --- | --- |
| Neutral talking-head set | `obama`, `may`, `lieu`, `macron` |
| MEAD | `M009`, `M019`, `M022`, `W009`, `W015`, `W019`, `W028` |

Rendering also requires the corresponding processed subject data. Follow the
[data preparation instructions](../README.md#3-processed-subject-data), then
pass `-m weights/<subject>` and `--iteration 10000` to `render.py`.
