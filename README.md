# 🧠 Suicide Ideation and Depression Detection System

## 📌 Overview
This project is a machine learning-based web application that detects depression and suicidal ideation from user input text. It uses Natural Language Processing (NLP) techniques and classification algorithms to analyze user emotions and predict risk levels.

---

## 🎯 Features
- Detects depression from user input
- Identifies suicidal ideation risk levels (Low / Moderate / High)
- Keyword highlighting for better transparency
- Displays prediction confidence
- Simple and user-friendly web interface

---

## 🛠️ Technologies Used

### Backend
- Python
- Flask

### Machine Learning
- Scikit-learn
- TF-IDF Vectorizer
- Classification Algorithms (Naive Bayes / Logistic Regression / SVM)

### Frontend
- HTML
- CSS
- JavaScript

---

## 📂 Project Structure
Suicide_Detection_Project/
│
├── app.py / server.py
├── model.pkl
├── vectorizer.pkl
├── templates/
│ ├── index.html
│ ├── result.html
│
├── static/
├── dataset/
├── requirements.txt
└── README.md

---

## ⚙️ Installation & Setup

### Step 1: Clone the Repository
git clone https://github.com/sowmyapotharlanka1319-ux/Suicidal_Ideation_Detection.git

cd Suicide_Detection_Project

### Step 2: Install Dependencies
pip install -r requirements.txt

### Step 3: Run the Application
python app.py

### Step 4: Open in Browser
http://127.0.0.1:5987/


---

## 📊 How It Works
1. User enters text input
2. Text is preprocessed (cleaning, tokenization)
3. TF-IDF converts text into numerical format
4. Machine learning model predicts risk level
5. Result is displayed with highlighted keywords

---

## 📁 Dataset
The dataset contains text data related to depression and suicidal thoughts used for training the model.

---

## 🚀 Future Enhancements
- Integration with deep learning models (BERT)
- Real-time chatbot support
- Mobile application development
- Improved dataset and model accuracy

---

## ⚠️ Disclaimer
This project is for educational purposes only and should not be used as a replacement for professional mental health advice.

---

## 👩‍💻 Author
Sowmya Potharlanka