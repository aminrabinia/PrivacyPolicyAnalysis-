import nltk
from nltk import SnowballStemmer, NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from sklearn import svm, naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from tempPyFile import download_text_from_web, download_links_from_DB, stor_in_db
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

sensetive, non_sens = [], []
all_words = [[],[]]

def sensitivity(text):   #checks sentence sensitivity
    txtoken=word_tokenize(text)
    for term in txtoken:
        if(re.match(r"share|use|consent|permiss|collect|process|control|access|you|your|personal|stor|agree",term)):
            return True;

def preprocess(text):

    tokenizer = RegexpTokenizer(r'\w+')
    sen_tknzd = tokenizer.tokenize(text) #removes punctuations
    sen_filtered = ""
    ss = SnowballStemmer("english")
    stop_words = set(stopwords.words('english'))
    for w in sen_tknzd:             #removes stop words
        if not w in stop_words:
            sen_filtered += str(ss.stem(w) + " ") #stemming
    return sen_filtered

def data_labling(data):
    for sen in data:
        sen_filtered=preprocess(sen)
        if len(sen_filtered)>10:
            all_words[0].append(sen_filtered)
            if sensitivity(sen_filtered):
                sensetive.append(sen_filtered)
                all_words[1].append(1)
            else:
                non_sens.append(sen_filtered)
                all_words[1].append(0)

def knn_Classifier():
    vectorizer = CountVectorizer()
    X_train, X_test, y_train, y_test = train_test_split(all_words[0], all_words[1], test_size=0.2, random_state=0)
    xtesttemp = X_test
    vectorizer.fit(X_train)
    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)

    for k in range(1, 2):  # range 1+
        cls = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
        cls.fit(X_train, y_train)
        tested = cls.predict(X_test)

        print(cls.score(X_test, y_test))
def naiveBayse():
    # fit the training dataset on the NB classifier
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(Train_X_Tfidf,Train_Y)
    # predict the labels on validation dataset
    predictions_NB = Naive.predict(Test_X_Tfidf)
    # Use accuracy_score function to get the accuracy
    print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y))

pps_urls=download_links_from_DB()      #returns links as a list

for i in range(0,len(pps_urls)):
    webdata=download_text_from_web(pps_urls[i])   #returns  web data as sentence tokenized
    data_labling(webdata)

X_train, X_test, y_train, y_test = train_test_split(all_words[0], all_words[1], test_size=0.3)
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(y_train)
Test_Y = Encoder.fit_transform(y_test)
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(all_words[0])
Train_X_Tfidf = Tfidf_vect.transform(X_train)
Test_X_Tfidf = Tfidf_vect.transform(X_test)


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y))




def classified_storage():   #classifies the whole data and stores them

    for i in range(0, len(pps_urls)):
        webdata = download_text_from_web(pps_urls[i])  # returns  web data as sentence tokenized
        doc=[]
        for sen in webdata:
            doc.append(preprocess(sen))

        prdct=SVM.predict(Tfidf_vect.transform(doc))

        dbtemp=""
        for n in range (0,len(doc)):
            if prdct[n]==1:
                dbtemp += str(doc[n])

        stor_in_db(dbtemp)

classified_storage()

