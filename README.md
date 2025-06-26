# AI-Powered Disease Detection from Medical Imaging
**Chest X-ray Pneumonia Detection using Deep Learning**

This project focuses on detecting pneumonia from chest X-ray images using deep learning and computer vision. Using a subset of the NIH Chest X-ray dataset, the model is trained to classify images as 'Pneumonia' or 'No Finding'.

---

## Goals

- Use computer vision to analyze medical images
- Apply deep learning techniques to real-world medical imaging
- Build and train a binary classification model to detect Pneumonia
- Gain hands-on experience with transfer learning and convolutional neural networks (CNNs)

---
## Data Wrangling Summary

- Loaded metadata (`Data_Entry_2017.csv`)
- Assigned binary labels:
  - `1` for pneumonia (even if co-labeled)
  - `0` for no findings
- Removed irrelevant/multilabel entries
- Balanced dataset to 1,431 pneumonia and 1,431 no-finding images
- Checked for missing values and duplicates
- Split into train/validation/test (70/15/15)
- Saved CSVs for reproducibility

---

## Dataset Source

**NIH Chest X-ray Dataset**  
- **URL:** https://www.kaggle.com/datasets/nih-chest-xrays/data  
- **Provided by:** National Institutes of Health Clinical Center  
- **Total images:** 112,120  
- **Total patients:** 30,805  
- **Labels:** 14 disease categories extracted from radiology reports using NLP

---

## Subset Used

For this project, a focused subset of the dataset was selected:
- **Image count:** 2862
- **Disease focus:** Pneumonia (binary classification: pneumonia vs. no pneumonia)
- **Metadata file used:** `Data_Entry_2017.csv`
- **Note:** Dataset not included in this repository due to size. You can download it from the source and extract a subset into './data'.

---

## Preprocessing & Filtering Steps

1. **Filtered** the CSV metadata to extract records with and without Pneumonia.
2. **Selected** a balanced subset from each class.
3. **Images** were resized and normalized before model training.
4. **Augmentation** (rotation, flipping, zoom) applied during training.

---

## Model Comparison Summary

| Model         | Val Acc | Test Acc | Notes                                       |
|---------------|---------|----------|---------------------------------------------|
| ResNet50      |  51%    |  19%     | Severe overfitting                          |
| DenseNet121   |  65%    |  56%     | Moderate generalization                     |
| EfficientNetB0|  52%    |  20%     | Underfit, unstable                          |
| VGG16         |  69%    |  73.7%   | Best overall performer (trained separately) |

*VGG16 was trained in a separate notebook (VGG16Experiment.ipynb) to allow for two-stage fine-tuning and separate control of training.*

---

## Scalable Model Development

To simulate a real-world deployment, the VGG16 model was retrained on an expanded dataset using 1,431 pneumonia and 15,000 normal chest X-ray images from the NIH dataset. Images were streamed from Google Drive using Keras' `ImageDataGenerator`, and class imbalance was addressed using weighted loss. 

Key choices included:
- **Model**: VGG16 with top 4 layers fine-tuned
- **Input**: 224x224 images, batch size 32
- **Optimizer**: Adam (1e-5), with early stopping
- **Training environment**: Google Colab Pro (A100 GPU)

**Best Validation Accuracy**: 76.5%  
**Training Accuracy**: 72.7%  
**Validation Loss**: 0.5458

The final model balances performance and efficiency, demonstrating its scalability on large and imbalanced medical imaging data.

---

## Project Structure

```text
chest_xray_pneumonia_detection/
├── data/
│   └── pneumonia_subset/
├── notebooks/
│   ├── Data_Wrangling.ipynb
│   └── Folder_Organization_for_Scale.ipynb
│   └── Image_Folder_Organization.ipynb
│   └── Model_Experiments.ipynb
│   └── Scaled_Model_Training.ipynb
│   └── VGG16_Experiment.ipynb
├── images/
│   └── sample_gradcam.png
├── metadata/
│   ├── Data_Entry_2017.csv
│   ├── balanced_labels.csv
│   ├── train_labels.csv
│   ├── val_labels.csv
│   └── test_labels.csv
├── Project_Proposal_Pneumonia.txt
├── requirements.txt
├── README.md
└── LICENSE
```
---

## Tools & Techniques

- Python, NumPy, Pandas, Matplotlib
- TensorFlow/Keras (CNNs, transfer learning)
- Image preprocessing and augmentation
- Model evaluation (Accuracy, F1, AUC, Confusion Matrix)
- Git, GitHub for version control and sharing

---

## License

This project is licensed under the MIT License. See 'LICENSE' for details.

---

## Acknowledgements

- NIH Clinical Center  
- Kaggle  
- Researchers & radiologists who shared open datasets


