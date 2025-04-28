# Hate Speech Classification

## 📚 Project Overview
This project focuses on building a robust hate speech classification system that can detect and differentiate between:

- Hate speech
- Offensive language
- Neutral content

It leverages Natural Language Processing (NLP) techniques, classical Machine Learning algorithms, and advanced Deep Learning models (RNNs, GRUs, LSTMs) to analyze and classify text data from social media and online platforms.

## 🗂 Project Structure
```
.
├── Data
│   ├── Hate Speech.tsv        # Raw dataset
│   └── cleaned_tweets.csv      # Preprocessed dataset
├── Notebook
│   ├── Hate Speech Classification.ipynb       # Traditional ML models
│   └── Text_Classification_using_RNNs.ipynb    # Deep learning models (RNN, GRU, LSTM)
└── README.md
```

## ⚙️ Features
- **Data Preprocessing:** Cleaning, tokenization, and normalization of tweets.
- **Exploratory Data Analysis (EDA):** Insights and visualizations on the dataset distribution.
- **Text Vectorization:** Using techniques like TF-IDF and word embeddings.
- **Model Building:**
  - Classical ML models (Logistic Regression, Random Forest, SVM)
  - Deep Learning models (Simple RNN, GRU, LSTM)
- **Model Evaluation:** Accuracy, Precision, Recall, F1-score.
- **Comparison:** Evaluation and comparison of different models and approaches.

## 🛠 Technologies Used
- Python
- Scikit-learn
- TensorFlow / Keras
- Pandas, NumPy
- Matplotlib, Seaborn
- Natural Language Toolkit (NLTK)

## 🚀 How to Run
1. **Clone the repository:**
```bash
git clone https://github.com/your-username/Hate-speech-classification.git
cd Hate-speech-classification
```
2. **Install the required dependencies:**
```bash
pip install -r requirements.txt
```
3. **Navigate to the Notebook/ folder and open the notebooks:**
- `Hate Speech Classification.ipynb`
- `Text_Classification_using_RNNs.ipynb`

Follow the notebook cells to preprocess data, train models, and evaluate performance.

## 📊 Results
The project compares multiple models and shows that deep learning models (especially LSTMs) tend to outperform traditional machine learning methods on the hate speech dataset after proper tuning and preprocessing.
