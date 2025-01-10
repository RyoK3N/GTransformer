# KTPFormer: Keypoint Transformer for Camera Matrix Prediction

A deep learning model that predicts camera matrices from 2D keypoints using a transformer-based architecture with graph convolutions.

## Features
- Graph-based keypoint processing
- Transformer architecture for sequence modeling
- Combined Frobenius norm and reconstruction loss
- TensorBoard visualization support
- MongoDB integration for data management

## Project Structure

- `model/`: Contains the model architecture and training logic.
- `data/`: Handles data loading and preprocessing.
- `utils/`: Utility functions for data visualization and logging.
- `runs/`: Stores model checkpoints and TensorBoard logs.
- `weights/`: Saved model weights.
- `Readme.md`: This file.

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/your-username/ktpformer.git
cd ktpformer
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your environment variables:

```bash
cp .env.example .env
```

4. Run the training script:

```bash
python train.py
```


