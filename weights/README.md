# Pretrained weights

Pretrained weights are intentionally not committed to the source repository.
After downloading the inference-only iteration-10,000 snapshots separately,
place each identity in the layout expected by `render.py`:

```text
weights/<subject>/
  point_cloud/iteration_10000/
    point_cloud.ply
    deformation.pth
    deformation_table.pth
    deformation_accum.pth
```

Included subjects:

| Dataset | Subjects |
| --- | --- |
| Neutral talking-head set | `obama`, `may`, `lieu`, `macron` |
| MEAD | `M009`, `M019`, `M022`, `W009`, `W015`, `W019`, `W028` |

The release package should contain only the final renderer state. Optimizer
states, training checkpoints, intermediate iterations, TensorBoard files,
evaluation renders, and third-party evaluation networks are excluded.
