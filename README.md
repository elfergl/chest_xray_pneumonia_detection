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
- **Image count:** 15,000
- **Disease focus:** Pneumonia (binary classification: pneumonia vs. no pneumonia)
- **Metadata file used:** `Data_Entry_2017.csv`
- **Note:** Dataset not included in this repository due to size. You can download it from the source and extract a subset (~15k images) into './data'.

---

## Preprocessing & Filtering Steps

1. **Filtered** the CSV metadata to extract records with and without Pneumonia.
2. **Selected** a balanced subset from each class.
3. **Images** were resized and normalized before model training.
4. **Augmentation** (rotation, flipping, zoom) applied during training.

---

## Project Structure

```text
chest_xray_pneumonia_detection/
├── data/
│   ├── pneumonia_subset_labels.csv
│   ├── data_README.md
│   └── pneumonia_subset/
├── notebooks/
│   ├── 01_create_subset.ipynb
│   └── 02_train_model.ipynb
├── scripts/
│   ├── create_pneumonia_subset.py
│   └── load_preprocess_pneumonia_data.py
├── images/
│   └── sample_gradcam.png
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


