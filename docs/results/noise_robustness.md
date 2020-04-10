---

title: Noise Robustness
parent: Results
nav_exclude: true

---

# Noise Generalization

What advantages might be conveyed by a representation that can reconstruct as well as classify?

One possibility is improved noise tolerance.
We took each of our previously trained models, and measured their accuracy at categorizing the clothing objects in test images corrupted by increasing levels of additive pixel noise.

![Figure 2](https://raw.githubusercontent.com/elijahc/vae/master/figures/pub/figure_2_2x.png)

Our findings suggest noise tolerance as another independent explanation for why the VS might use a composite computational objective.
VS classification accuracy measured in humans tolerates noise corrupted images much better than DCNNs optimized for image classification alone.
In contrast, convolutional model’s optimizing the composite objective demonstrate better noise tolerance compared to identical models trained solely for classification.
