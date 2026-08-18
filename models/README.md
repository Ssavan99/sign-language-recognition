# Released model

`asl_alphabet_cnn_seed42.pt` is the reproducible compact CNN checkpoint used by
the maintained prediction and demo paths.

## Model card

- Task: classify one isolated, static ASL alphabet image as A-Z.
- Architecture: three convolutional blocks, adaptive average pooling, and a
  compact classifier head.
- Parameters: 164,546.
- Input: RGB image resized to 64 x 64 and normalized according to the
  preprocessing contract embedded in the checkpoint.
- Training data: 51,376 images from the prepared primary-source training
  manifest.
- Model selection: lowest loss on a separate 11,024-image validation manifest.
- Seed: 42.
- Best epoch: 12 of 12.
- Size: 674,997 bytes.
- SHA-256:
  `8b3d071082615de4a21a6303c8e9ca5496a747c6f5707064f26b3d6a9f6c40a7`.

The checkpoint stores the architecture identifier, state dictionary, A-Z class
order, image size, preprocessing contract, seed, best epoch/loss, parameter
count, training configuration, and train/validation manifest hashes. PyTorch
loads it with `weights_only=True` in the supported runtime.

## Intended use

The model supports local demonstrations and reproducible analysis of controlled,
single-hand still images. It is not a continuous sign-language recognizer,
translator, hand detector, or validated accessibility system.

The same-source test score is 99.82%, but accuracy falls to 17.56% on a separate
external image source. This gap is evidence of severe capture-domain bias. Do
not use the same-source score as evidence of real-world reliability. See the
[current result report](../docs/results/current/README.md) for the complete
evaluation contract and limitations.

## Prediction

```bash
asl-recognition predict \
  --checkpoint models/asl_alphabet_cnn_seed42.pt \
  --device cpu \
  path/to/hand.jpg
```

Predictions below the configured confidence threshold are returned with a
warning rather than presented as reliable classifications.
