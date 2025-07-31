# Brain-Tumor
# 🧠 Brain MRI Tumor Classifier

This is a Streamlit web application that allows users to upload brain MRI images and get tumor type predictions using a trained deep learning model.

## 💻 Requirements

- Python 3.10 or 3.11 (TensorFlow is not compatible with Python 3.13)
- TensorFlow
- Streamlit
- NumPy
- Pillow

## 🚀 Getting Started

1. Clone the repository  
   `git clone https://github.com/your-username/brain-tumor-streamlit-app.git`

2. Navigate into the folder  
   `cd brain-tumor-streamlit-app`

3. Create a virtual environment (optional but recommended)  
   `python -m venv venv`  
   `source venv/bin/activate` *(or `venv\Scripts\activate` on Windows)*

4. Install dependencies  
   `pip install -r requirements.txt`

5. Run the app  
   `streamlit run brain_tumor_app.py`

## 📂 Model

Make sure the `best_model.h5` file (your trained model) is placed in the same directory.

## 🧪 Output

The app will display:

- Tumor type prediction (Glioma, Meningioma, Pituitary)
- Confidence score
- Full probability breakdown
