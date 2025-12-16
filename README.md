# Sentiment Analysis of Amazon Reviews with Streamlit Dashboard

This project focuses on analyzing Amazon product reviews to automatically classify them as Positive or Negative using Natural Language Processing (NLP) and Machine Learning techniques. The goal is to help businesses quickly identify negative customer feedback and improve product quality and customer satisfaction.

## 📊 Dataset
- Source: Kaggle  
- Dataset: Datafiniti Amazon Consumer Reviews  
- Reference Notebook:  
  https://www.kaggle.com/code/lele1995/amazon-reviews-sentiment-analysis

## 📁 Project Components

### 📓 Jupyter Notebook
**File:** `amazon_sentiment_analysis.ipynb`

Used for:
- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA)  
- Word cloud visualization  
- TF-IDF feature extraction  
- Training ML models (Logistic Regression, Naive Bayes, SVM)  
- Model evaluation and comparison  

### 🖥 Streamlit Application
**File:** `app.py`

- User interface for sentiment prediction  
- Loads trained SVM model and TF-IDF vectorizer  
- Accepts user input and predicts sentiment  
- Displays results visually  

### ⚙️ Supporting Files
- `requirements.txt` – Lists all Python dependencies  
- `svm_model.pkl` – Saved trained Support Vector Machine model  
- `tfidf_vectorizer.pkl` – Saved TF-IDF vectorizer used during training  

## ⚙️ Setup & Run

git clone https://github.com/jaganadeepika/SentimentAnalysis.git
cd SentimentAnalysis

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet punkt
streamlit run app.py

▶️ Usage

Open the Jupyter Notebook:
notebook/amazon_sentiment_analysis.ipynb
and run all cells to train models and save the SVM model and TF-IDF vectorizer.

Launch the Streamlit app:
streamlit run app.py

Enter an Amazon product review.

Click Predict Sentiment to classify the review as Positive or Negative.

📸 Screenshots
<img width="1919" height="836" alt="image" src="https://github.com/user-attachments/assets/0c1994c8-a17e-41a2-8b0d-230da5f1d594" />
<img width="1900" height="643" alt="image" src="https://github.com/user-attachments/assets/eef37ce6-2cc9-41ea-a177-cf02f3f11e4d" />
<img width="1384" height="672" alt="image" src="https://github.com/user-attachments/assets/afd32260-eb55-431a-a834-aa0ca2972b73" />
<img width="1316" height="494" alt="image" src="https://github.com/user-attachments/assets/66ba2b5b-2a4b-4ed1-a1dc-ce65d6fcb573" />
<img width="1322" height="559" alt="image" src="https://github.com/user-attachments/assets/125ab076-2c48-4502-ad4f-d3f169622d3c" />
<img width="1307" height="542" alt="image" src="https://github.com/user-attachments/assets/d51d4423-8101-4a6d-85fd-0b8e3a53fb11" />
<img width="1345" height="751" alt="image" src="https://github.com/user-attachments/assets/2fde6e27-948c-492e-8441-57f3c635c116" />
<img width="1344" height="733" alt="image" src="https://github.com/user-attachments/assets/be894e30-f8f5-4ff5-9aea-f03abcbd474c" />

## 🤖 Models Used

### 1️⃣ Logistic Regression
- A linear classification algorithm used as a baseline model.  
- Works well with high-dimensional text data.  
- Helps understand feature importance in sentiment prediction.

---

### 2️⃣ Naive Bayes
- A probabilistic classifier based on Bayes’ Theorem.  
- Efficient and fast for large text datasets.  
- Performs well when features are assumed to be independent.

---

### 3️⃣ Support Vector Machine (SVM)
- Used as the **final and best-performing model**.  
- Finds the optimal hyperplane to separate positive and negative reviews.  
- Handles sparse TF-IDF vectors effectively.  
- Achieved the highest accuracy among all models.

---

### 📌 Model Selection
- All models were trained and evaluated using the same TF-IDF features.  
- Performance was compared using accuracy scores and confusion matrices.  
- **SVM was selected for deployment** in the Streamlit application.

<img width="258" height="166" alt="image" src="https://github.com/user-attachments/assets/74f3e6f8-0439-483f-8e0c-8a8c37fddeed" />
