# Local demo

The optional Gradio interface runs the released checkpoint on the local machine. It accepts an
uploaded image or webcam frame, uses the same `Predictor` preprocessing and A-Z class mapping as
the command-line interface, and does not create a public share link.

Install the optional dependency and launch from the repository root:

```powershell
python -m pip install -e ".[demo]"
asl-recognition demo --checkpoint models/asl_alphabet_cnn_robust_seed42.pt --device cpu
```

Open the printed local URL. The built-in **External-domain sample — true label A** example provides
a deterministic check of the interface. The released model predicts L with 76.2% confidence for
this image. That incorrect, confident result is retained deliberately: it demonstrates why the UI
always displays the measured 31.67% external-domain accuracy and warns that softmax confidence is
not evidence of real-world reliability.

![Working local demo showing the domain-shift example](demo-screenshot.png)

The sample is `A/A0001_test.jpg` from the public
[`danrasband/asl-alphabet-test`](https://www.kaggle.com/datasets/danrasband/asl-alphabet-test)
dataset and is provided under the source listing's CC0 license. Its SHA-256 is
`4c22cf21532007e15036d69e6d511ed050cea28eeb6bbb65488fe19dd70bba71`.

This interface classifies isolated still images. It is not a hand detector, continuous-signing
translator, or validated accessibility system. J and Z require motion in real signing and cannot
be represented fully by one frame.
