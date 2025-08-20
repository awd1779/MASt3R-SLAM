# 3D Reconstruction Comparison (No Alignment)

## Results

| Method        | Points    | GT Points   |   Extent (m) |   GT Extent (m) |   Scale Ratio |   Chamfer (m) |   Accuracy (m) |   Completeness (m) |   Hausdorff (m) |   F@0.010 |   F@0.020 |   F@0.050 |   F@0.100 |   F@0.200 |   F@0.500 |
|:--------------|:----------|:------------|-------------:|----------------:|--------------:|--------------:|---------------:|-------------------:|----------------:|----------:|----------:|----------:|----------:|----------:|----------:|
| Monocular2Map | 1,327,818 | 589,517     |        9.893 |           7.308 |         1.354 |        0.5256 |         0.3554 |             0.6957 |          2.3502 |     0.015 |     0.033 |     0.078 |     0.146 |     0.273 |     0.578 |
| HOV-SG        | 915,878   | 589,517     |        7.248 |           7.308 |         0.992 |        0.075  |         0.0052 |             0.1448 |          1.7122 |     0.685 |     0.803 |     0.834 |     0.873 |     0.909 |     0.942 |

## Key Insights

- **Scale Ratio**: How much larger/smaller predicted vs GT
- **Extent**: Diagonal of bounding box
- **No alignment applied** - shows raw reconstruction quality
- Lower distance metrics are better
- Higher F-scores are better
