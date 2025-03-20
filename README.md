# HSTU-Tensorflow

This is an open-source TensorFlow implementation of the HSTU model (Hierarchical Sequential Transduction Units), inspired by the original PyTorch version developed by Meta AI. The HSTU model is a cutting-edge solution for generative recommendation systems。

This implementation adapts code from the [FPARec](https://github.com/NetEase-Media/FPARec) repository, a positional-attention-based sequential recommendation system developed by the author of this project. Building on this foundation, our TensorFlow implementation of the HSTU model seeks to broaden its accessibility for research and industrial deployment, as TensorFlow remains a dominant framework in many production recommendation systems.

Unlike the original PyTorch implementation, we apply a full softmax for the autoregressive loss calculation and normalize attention weights using the actual sequence length rather than the maximum length. These modifications improve model performance

## Performance

We use ml-1m dataset as the benchmark.

| Method                    | HR@10  | NDCG@10 | HR@100 | NDCG@100 | HR@200 | HR@200 |
|---------------------------|--------|---------|--------|----------|--------|--------|
| HSTU-large (paper result) | 0.3294 | 0.1893  | -      | -        | 0.7839 | 0.2771 |
| HSTU (This repo)          | 0.3419 | 0.1944  | 0.7089 | 0.2702   | 0.7949 | 0.2827 |

## Train the model

Training the model on ml-1m with all default parameters:

```python
bash scripts/train.sh
```

An NVIDIA 1080Ti GPU (11GB VRAM) is sufficient to train the model on the ml-1m dataset.

## Dependencies:

```
TensorFlow (>= 1.10, <= 1.15)
```

## Acknowledgement

We express our gratitude to the following:

- The authors of the original paper: [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152).
- The Meta AI team for their PyTorch implementation in the [generative-recommenders](https://github.com/facebookresearch/generative-recommenders) repository, which inspired this TensorFlow version.
- The [SASRec](https://github.com/kang205/SASRec) authors (Kang et al.) and [TiSASRec](https://github.com/JiachengLi1995/TiSASRec) authors (Li et al.) for their foundational work on sequential recommendation, adapted in this project.