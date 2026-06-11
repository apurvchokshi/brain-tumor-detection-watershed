# Brain Tumor Detection using Watershed and Deep Learning

This project is a brain tumor detection application built using image preprocessing, watershed segmentation, and a trained deep learning model. The app allows users to upload a brain MRI image and receive a predicted tumor class through a Gradio interface.

## 🎯 Live Demo

**Hugging Face Space**: https://huggingface.co/spaces/apurv20/Brain_Tumor_Detection_with_Watershed

## ✨ Features

- **MRI Image Upload**: Simple interface to upload brain MRI scans
- **Advanced Preprocessing**: Multi-stage image enhancement pipeline
  - Anisotropic diffusion for noise reduction
  - Skull stripping to isolate brain tissue
  - Top-hat morphological enhancement
  - Histogram equalization for contrast improvement
- **Watershed Segmentation**: Accurate tumor region detection and boundary extraction
- **Deep Learning Classification**: CNN-based tumor classification
- **Confidence Scoring**: Display prediction confidence and class probabilities
- **Visualization**: See preprocessed image with watershed segmentation overlay
- **User-Friendly Interface**: Web-based Gradio application for easy interaction

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.7+ |
| Deep Learning | TensorFlow / Keras |
| Image Processing | OpenCV |
| Scientific Computing | NumPy |
| Web Interface | Gradio |
| Deployment | Hugging Face Spaces |

## 📁 Project Structure

```
brain-tumor-detection-watershed/
├── README.md                                      # Project documentation
├── requirements.txt                               # Python dependencies
├── app.py                                         # Gradio web application
├── Brain_tumor_detection_with_watershed.ipynb    # Training & development notebook
├── best_model.keras                               # Trained CNN model (generated)
└── label_map.json                                 # Class label mapping (generated)
```

## 📋 Requirements

```
gradio
numpy
opencv-python-headless
scikit-image
tensorflow
pillow
```

## 🚀 Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/apurvchokshi/brain-tumor-detection-watershed.git
cd brain-tumor-detection-watershed
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Train the model** (if needed):
   - Open and run the Jupyter notebook: `Brain_tumor_detection_with_watershed.ipynb`
   - This generates:
     - `best_model.keras` - Trained CNN model
     - `label_map.json` - Class labels mapping

4. **Run the application**:
```bash
python app.py
```

The app will launch at `http://127.0.0.1:7860` (or another local port if 7860 is busy)

## 💡 How to Use

1. Open the Gradio web interface
2. Click "Upload brain MRI" to select an MRI image
3. Click "Submit" to process the image
4. View results:
   - **Predicted class**: The detected tumor type
   - **Confidence score & probabilities**: Model confidence levels
   - **Preprocessed image**: Input image with segmentation overlay

## 🔍 Algorithm Pipeline

### Image Preprocessing
```
Raw MRI Image
    ↓
Grayscale Conversion
    ↓
Anisotropic Diffusion (edge-preserving noise reduction)
    ↓
Skull Stripping (brain tissue extraction)
    ↓
Top-hat Enhancement (structure highlighting)
    ↓
Histogram Equalization (contrast improvement)
    ↓
Resize to 224×224 pixels
    ↓
Normalized RGB format for model input
```

### Watershed Segmentation
- Otsu's automatic thresholding
- Morphological opening (remove noise)
- Distance transform computation
- Connected component labeling
- Marker-based watershed algorithm
- Post-processing morphological operations

### Deep Learning Classification
- CNN model trained on preprocessed MRI images
- Multi-class tumor classification
- Probability distribution over all tumor types

## 📝 Key Components

### `app.py` - Web Application
Gradio interface that:
- Loads pre-trained model and label mapping
- Accepts MRI image uploads
- Runs full preprocessing and prediction pipeline
- Returns classification and visualization

### `Brain_tumor_detection_with_watershed.ipynb` - Training Notebook
Contains:
- Dataset loading and exploration
- Data preprocessing and augmentation
- CNN model architecture
- Training with validation
- Model evaluation and metrics
- Automatic model saving

## ⚠️ Important Disclaimer

**This project is for educational and research purposes only.** 

⛔ **NOT for clinical use**: This application is NOT a medical diagnostic tool and should NOT be used for actual medical diagnosis or clinical decision-making.

⚕️ **Consult professionals**: Always consult qualified medical professionals and radiologists for MRI interpretation and diagnosis.

## 📊 Model Information

- **Architecture**: Deep Convolutional Neural Network (CNN)
- **Input Size**: 224×224×3 (RGB)
- **Output**: Multi-class probability distribution
- **Training Data**: Brain tumor MRI dataset
- **Deployment**: Hosted on Hugging Face Spaces for easy access

## 📂 Data Notes

- The trained model is hosted separately through Hugging Face
- The dataset is not included in this GitHub repository due to:
  - Large file size
  - Licensing and privacy considerations
  - Medical data sensitivity

For dataset access or model details, visit the [Hugging Face Space](https://huggingface.co/spaces/apurv20/Brain_Tumor_Detection_with_Watershed)

## 🎓 Educational Value

This project demonstrates:
- Classical image processing techniques in medical imaging
- Integration of traditional computer vision with deep learning
- Proper preprocessing workflows for MRI analysis
- Model deployment with user-friendly web interfaces
- Best practices in machine learning applications

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

## 📚 References

- Watershed Segmentation: Marker-based image segmentation techniques
- Anisotropic Diffusion: Perona-Malik diffusion (1990)
- MRI Preprocessing: Standard neuroimaging pipelines
- TensorFlow/Keras: Deep learning documentation

## 📄 License

This project is open source and available for academic and research use.

## 👤 Author

**Apurv Chokshi**

GitHub: [@apurvchokshi](https://github.com/apurvchokshi)

---

**Last Updated**: June 2026

For questions, issues, or suggestions, please open an issue on the GitHub repository or visit the [Hugging Face Space](https://huggingface.co/spaces/apurv20/Brain_Tumor_Detection_with_Watershed).
