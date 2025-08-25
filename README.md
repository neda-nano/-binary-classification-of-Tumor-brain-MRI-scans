# -binary-classification-of-Tumor-brain-MRI-scans
Deep Learning pipeline for binary classification of brain MRI scans (98 Normal, 155 Tumor) using PyTorch + ResNet18. Includes data loaders, training, evaluation, and metrics.
Brain Tumor MRI — Deep Learning Classification

This repository contains a PyTorch-based deep learning pipeline for classifying brain MRI scans into two categories:

🧠 Normal — 98 images

🎯 Tumor — 155 images

The model leverages transfer learning (ResNet18) for binary classification, with reproducible training, validation, and testing steps.

🚀 Features

Organized dataset loader (normal/, tumor/)

Image preprocessing + data augmentation

Training & validation pipeline with PyTorch

Evaluation metrics: Accuracy, Precision, Recall, F1, Confusion Matrix

Model checkpointing (best_model.pth)

Easy-to-extend modular codebase

📂 Dataset Structure

Organize your data as follows:

dataset/
   normal/    # 98 images
   tumor/     # 155 images


Accepted formats: .png, .jpg, .jpeg

⚙️ Installation

Clone the repo and install dependencies:

git clone https://github.com/<your-username>/brain-tumor-mri-dl.git
cd brain-tumor-mri-dl
pip install -r requirements.txt

🏃‍♀️ Training

Train the model:

python src/train.py --data-dir dataset --epochs 30 --batch-size 16 --out-dir experiments --pretrained


Arguments:

--data-dir → path to dataset

--epochs → number of training epochs

--batch-size → training batch size

--out-dir → save directory for checkpoints

--pretrained → use ImageNet pretrained weights

📊 Evaluation

After training, the best model is saved in experiments/best_model.pth.
The script automatically evaluates on the test set and prints:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

🧩 Example Results

(Insert a confusion matrix plot or sample prediction results here when you run the model.)

🔧 Future Work

Add EfficientNet/DenseNet backbones

Hyperparameter tuning

Data augmentation improvements

Explainability with Grad-CAM

📜 License

This project is licensed under the MIT License.
