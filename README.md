# Training Digits Dataset Using Moments

Computer-vision experiment that classifies handwritten digits from scikit-learn’s `load_digits` dataset using **image moments and contour features** instead of raw pixels.

Moments summarize shape properties (area, centroid, orientation, and higher-order statistics). Contours give the digit outline. Those features feed a small neural network.

## What’s in the repo

| File | Description |
|------|-------------|
| `Moments.ipynb` | End-to-end pipeline: preprocess → moments → train → evaluate |
| `moments.py` | Script version of the same pipeline |
| `Expanded.ipynb` | Extra contour / geometry exploration (incomplete — see note below) |
| `Report.pdf` | Project report |

## Pipeline (`Moments.ipynb`)

1. Load the digits dataset (8×8 grayscale images)
2. Scale pixels to `uint8` and threshold (Otsu)
3. Find contours with OpenCV
4. Compute moment features per digit and pad feature vectors to a fixed length
5. Train / test split
6. Train a Keras MLP (`Dense(128)` → 10-class softmax), 100 epochs
7. Report test accuracy and plot a confusion matrix

### Result from the saved notebook run

**Test accuracy ≈ 66.7%**

That is modest compared with pixel-based models on digits, which is expected: moments discard a lot of local texture and the feature vectors are relatively short. The point of the project is to practice shape descriptors, not to maximize leaderboard score.

## Tech stack

Python, OpenCV (`cv2.moments`, contours), TensorFlow / Keras, scikit-learn, NumPy, Pandas, Matplotlib, Seaborn, Jupyter

## How to run

```bash
git clone https://github.com/mahmouduskudar/training-digits-dataset-using-moments.git
cd training-digits-dataset-using-moments
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python tensorflow jupyter
jupyter notebook Moments.ipynb
```

The digits data comes from scikit-learn; no separate download is required.

## Note on `Expanded.ipynb`

`Expanded.ipynb` experiments with extra contour measurements (area, perimeter, bounding boxes, enclosing circles, ellipse / line fits) under transforms. It is **not a finished training notebook** and still needs cleanup before it matches the main pipeline. Prefer `Moments.ipynb` as the working entry point.
