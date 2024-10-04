# !pip install scikit-learn
# !pip install nltk
#required packages library!
import nltk
nltk.download('stopwords')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import CounterVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
## data processing:
#load data:
df = pd.read_csv("spam_emails_detection.csv")
print(df.head(),end = "\n\n")
print(df.info(),end='\n\n')
print(df.describe(),end='\n\n')
#data visualization:
plt.figure(dpi=100)
# sns.set_style("white")
# sns.countplot(df['spam'])
plt.bar(df['spam'],height=10)
plt.title('Spam count frequency')
plt.show()
#data cleaning:
## handling missing data for each column
print(df.isnull().sum())
## remove duplicates:
df.drop_duplicates(inplace=True)
## clean data from punctualtion and stop words and the tokenizing it into words(tokens)
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = text.lower().split()
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)
df['text'] = df['text'].apply(preprocess_text)
## Feature Extraction: Convert the text data into numerical features using TF-IDF.
vectorizer = TfidfVectorizer(max_features = 3000)
x = vectorizer.fit_transform(df['text'])
y = df['spam']
# split the data set to train and testing
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=0)
# Model creation and training
model = MultinomialNB() # multinomial naive bayes (MultinomialNB)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report: {classification_report(y_test, y_pred)}')
#model evalutaion using confusion matrix
cm = confusion_matrix(y_test,y_pred)
plt.figure(dpi=100)
sns.heatmap(cm,annot = True)
plt.title('confusion matrix')
plt.show()
'''
TN (Top-left, 0,0): 880 — The model correctly classified 880 non-spam emails as non-spam (True Negatives).

FP (Top-right, 0,1): 6 — The model incorrectly classified 6 non-spam emails as spam (False Positives).

FN (Bottom-left, 1,0): 15 — The model incorrectly classified 15 spam emails as non-spam (False Negatives).

TP (Bottom-right, 1,1): 250 — The model correctly classified 250 spam emails as spam (True Positives).
'''
def classify(user_email):
    # Preprocess the text
    cleaned_email = preprocess_text(user_email)
    # Vectorize the text (Assuming vectorizer was fitted on the training data)
    email_vector = vectorizer.transform([cleaned_email])
    # Predict using the trained model
    if  model.predict(email_vector):
        print(f"The email: '{user_email}' is classified as SPAM.")
    else:
        print(f"The email: '{user_email}' is classified as NOT SPAM.")
# Example email text from user
user_email = "Congratulations! You've won a $1,000 gift card. Click here to claim your prize."
classify(user_email)
user_email = "Thank you for joining this cousrse"
classify(user_email)
