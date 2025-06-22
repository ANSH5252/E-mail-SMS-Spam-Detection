# 📧 E-mail/SMS Spam Detection

This is a Machine Learning-based Spam Classifier that identifies whether a given message (email or SMS) is **Spam** or **Not Spam**. It uses Natural Language Processing (NLP) techniques to preprocess text and applies classification algorithms to make accurate predictions. The project is built using Python and deployed as a **Streamlit** web application.



## 🚀 Features

- Classifies a message as **Spam** or **Not Spam**  
- Real-time message input and prediction
- Text preprocessing using custom NLP pipeline
- Trained on a labeled dataset with binary classification
- Optimized with feature extraction using TF-IDF vectorization
- Deployable locally as a Streamlit app



## 📊 Tech Stack

**Language:**  
- Python 🐍

**Libraries & Tools:**  
- `pandas`, `numpy` – Data handling  
- `nltk` – Text preprocessing (tokenization, stopwords, stemming)  
- `scikit-learn` – Model training and evaluation  
- `streamlit` – Web application interface  
- `pickle` – Model and vectorizer serialization  

**Model:**  
- Classifier (e.g., MultinomialNB, SVC, etc.)  
- Vectorizer: `TfidfVectorizer`



## 📂 Project Structure
```
E-mail-SMS-Spam-Detection/
├── spam.csv 
├── vectorizer.pkl 
├── model.pkl 
├── app.py 
├── s_c.ipynb 
├── README.md 
└── requirements.txt 
```




## 📥 How to Run

### 1. Clone the Repository

```
git clone https://github.com/ANSH5252/E-mail-SMS-Spam-Detection.git
cd E-mail-SMS-Spam-Detection
```
### 2. Install Dependencies  

Make sure you have Python 3.x installed. Then run:

```
pip install -r requirements.txt
```
### 3. Run the Application
Launch the web app locally using Streamlit:
```
streamlit run app.py
```
## 🧠 How It Works  
The spam detection system follows a three-step pipeline: Text Preprocessing, Feature Extraction, and Prediction.

🔹 1. Text Preprocessing
- Converts text to lowercase for consistency

- Uses TweetTokenizer to split text into tokens

- Removes punctuation and stopwords (common English words)

- Applies PorterStemmer to reduce words to their root form

- Joins processed tokens into a clean string for analysis

🔹 2. Feature Extraction
- Uses TF-IDF Vectorizer to convert cleaned text into numerical features

- Captures the importance of words based on their frequency and uniqueness

- Produces a sparse matrix used for model training and predictions

🔹 3. Model Prediction  
- A trained ML model (e.g., Naive Bayes or Voting Classifier) takes the vectorized input

- Outputs a prediction: Spam or Not Spam

- Result is displayed instantly via the Streamlit interface

## 🧪 Example Usage
**Input:**

Congratulations! You've won a free iPhone. Click here to claim.

**Output:**

Spam  
## 📸 Screenshots  
![Screenshot 2025-06-22 142236](https://github.com/user-attachments/assets/a31618ee-260c-4fb1-a7ab-b3ed64ce0af9)
![Screenshot 2025-06-22 142213](https://github.com/user-attachments/assets/fd8541cb-0a08-45da-ae68-8761f1d8d14f)
![Screenshot 2025-06-22 142143](https://github.com/user-attachments/assets/78ec6f2a-d4ff-49b2-aaa5-ef3411cf8624)
![Screenshot 2025-06-22 141755](https://github.com/user-attachments/assets/e22205b4-a3bd-4896-a290-1e4376633d9b)
![Screenshot 2025-06-22 141720](https://github.com/user-attachments/assets/0c20fcd4-2a3c-4500-a422-c674db6cd0e7)

## 🤝 Contributing
Feel free to fork the repo, create issues, or submit pull requests. Suggestions for improvements and feature enhancements are always welcome!

## 📄 License
This project is licensed under the MIT License.

## 🙋‍♂️ Author
Anshuman Dash
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/ANSH5252)
