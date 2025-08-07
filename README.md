SMS Spam Detection using Machine Learning

A machine learning project to classify SMS messages as spam or not spam using the Naive Bayes algorithm, text preprocessing techniques, and the TF-IDF vectorizer. This project demonstrates the practical application of NLP in filtering unwanted messages.


---

📑 Table of Contents

1. Abstract


2. Problem Statement


3. Dataset Used


4. Approach and Methodology


5. Results and Evaluation


6. Member-wise Contribution


7. Challenges Faced


8. Learnings


9. Future Scope


10. References




---

🧠 Abstract

This project focuses on the implementation of a spam detection system for SMS messages using machine learning techniques. The model is trained to differentiate between spam and non-spam (ham) messages using natural language processing. The system converts raw SMS messages into numerical features using the TF-IDF approach and classifies them using the Multinomial Naive Bayes algorithm. The model is evaluated based on accuracy, precision, recall, and F1-score, showing high effectiveness in spam detection.


---

📌 Problem Statement

The exponential increase in mobile communication has led to an increase in spam messages that annoy users and sometimes lead to scams. The objective of this project is to develop a machine learning model that can efficiently classify SMS messages into spam or ham categories, improving user experience and safety.


---

📂 Dataset Used

We used the SMS Spam Collection Dataset from Kaggle, which contains 5,572 English SMS messages labeled as either "ham" or "spam". The dataset is ideal for binary classification tasks and includes a wide variety of real-world message patterns.

> 📦 Source: Kaggle - SMS Spam Collection Dataset




---

⚙ Approach and Methodology

The project follows these key steps:

Data Preprocessing: Clean the dataset and label the messages numerically (ham = 0, spam = 1).

Splitting the Data: Use an 80-20 split for training and testing.

Text Vectorization: Convert text messages to numeric form using TF-IDF Vectorizer.

Model Building: Train a Multinomial Naive Bayes model on the processed data.

Prediction: Use the trained model to classify new or unseen messages.

Evaluation: Analyze model performance using accuracy and classification metrics.


The model was implemented and tested using Python in Visual Studio Code.


---

📊 Results and Evaluation

The trained model achieved a high accuracy of approximately 97% on the test set. It also performed well across other evaluation metrics such as precision, recall, and F1-score. These results demonstrate that the Naive Bayes classifier is highly effective for text-based spam detection.

Example:

Accuracy: 0.97
Precision: 0.96
Recall: 0.95


---

👨‍💻 Member-wise Contribution

Member 1 (Team Leader): Supervised the project, coordinated roles, and ensured timely progress.

Member 2 (Python Coder): Developed the core machine learning code, handled model training, and testing.

Member 3 (Documentation Writer): Created the project documentation, report writing, and README preparation.



---

🧩 Challenges Faced

Understanding and cleaning real-world text data with inconsistent formatting.

Choosing the right vectorization method for better model performance.

Avoiding overfitting while ensuring good accuracy on unseen messages.



---

🎓 Learnings

Gained hands-on experience with Natural Language Processing (NLP).

Understood the role of TF-IDF and Naive Bayes in text classification.

Learned to handle real-world datasets and evaluate machine learning models.



---

🚀 Future Scope

Extend the model to support multilingual SMS detection.

Build a real-time SMS filtering app using the trained model.

Experiment with advanced models like LSTM or transformers (BERT) for improved accuracy.

Deploy the model using Flask or Streamlit for a web-based interface.



---

📚 References

Kaggle Dataset – SMS Spam Collection

Scikit-learn Documentation

TF-IDF Vectorizer – Scikit-learn

Naive Bayes Classifier
