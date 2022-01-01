from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
import warnings
import string
import numpy as np
import pandas as pd
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

tfidfvectorizer = TfidfVectorizer(analyzer="word")

warnings.filterwarnings('ignore')

'''
This Python script contains the functions used to perform the exploratory data analysis and ML classificaiton.
'''

def gather_csv(middle=False):
    path = "/Users/nawal/PycharmProjects/NawalPython/605_744_Information_Retrieval/Project/archive"
    text_files = [f for f in os.listdir(path) if f.endswith('.csv')]

    cat = []

    for file in text_files:
        frame = pd.read_csv(file)
        cat.append(frame)

    dframe = pd.concat(cat)

    dframe = dframe.astype({"Headline": "str",
                            "Content": "str"})

    dframe.reset_index(drop=True, inplace=True)
    dframe = dframe.drop_duplicates()
    dframe = dframe.drop_duplicates(subset='Headline', keep="last")
    dframe = dframe.drop_duplicates(subset='Content', keep="last")
    dframe.reset_index(drop=True, inplace=True)

    dframe = dframe[dframe["Headline"] != "Error"]
    dframe.reset_index(drop=True, inplace=True)

    with open('bias.pickle', 'rb') as file:
        adfontes_bias = pickle.load(file)

    with open('reliability.pickle', 'rb') as file:
        adfontes_reliability = pickle.load(file)

    pubs = dframe["Source"].to_list()

    if not middle:
        leaning = []
        leaning_encode = []
        for each in pubs:
            rating = adfontes_bias[each]
            if rating > 0:
                leaning.append("Right")
                leaning_encode.append(1)
            else:
                leaning.append("Left")
                leaning_encode.append(0)

    if middle:
        leaning = []
        leaning_encode = []
        for each in pubs:
            rating = adfontes_bias[each]
            if rating > 5:
                leaning.append("Right")
                leaning_encode.append(1)
            elif rating < -5:
                leaning.append("Left")
                leaning_encode.append(0)
            else:
                leaning.append("Middle")
                leaning_encode.append(2)

    dframe["Leaning"] = pd.Series(leaning)
    dframe["Leaning Encode"] = pd.Series(leaning_encode)
    dframe = dframe.drop(["URL"],axis=1)

    for i in range(dframe.shape[0]):
        dframe["Headline"].iloc[i].replace(" | AP News", "")
        dframe["Headline"].iloc[i].replace("Associated Press", "")
        dframe["Headline"].iloc[i].replace("Palmer Report", "")
        dframe["Headline"].iloc[i].replace("Occupy Democrats", "")
        dframe["Headline"].iloc[i].replace("Fox News", "")
        dframe["Headline"].iloc[i].replace("CNN", "")

    return dframe, adfontes_bias, adfontes_reliability

'''
This function normalizes a string by replacing underscores and hyphens with spaces, lowercasing all words,
and removing other punctuation.

Input: string
Output: normalized string
'''


def normalize(strng):
    # replace certain punctuations
    strng = strng.replace("-", " ")
    strng = strng.replace("_", " ")
    strng = strng.replace(";", " ")
    strng = strng.replace("/", " ")
    strng = strng.replace("'", "")
    strng = strng.replace("”", "")
    strng = strng.replace("’", "")


    # remove news outlet specific terms
    strng = strng.replace("CNN", "")
    strng = strng.replace("Fox", "")
    strng = strng.replace("FOX", "")
    strng = strng.replace("OAN", "")
    strng = strng.replace("New York Post", "")
    strng = strng.replace("NY Post", "")
    strng = strng.replace("Occupy Democrats", "")
    strng = strng.replace("Palmer Report", "")
    strng = strng.replace("Palmer", "")
    strng = strng.replace("One America News", "")

    # replace punctuations
    strng = strng.translate(str.maketrans('', '', string.punctuation))

    # lowercase all tokens
    strng = strng.lower()

    return strng

def lemmatize(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

'''
This function removes the stop words from a string.

Input: string
Output: string with stop words removed
'''


def stops(strng, custom=False, stop_list=[], lemma=False):
    tokens = strng.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]

    if custom:
        tokens = [word for word in tokens if word not in stop_list]

    if lemma:
        tokens = lemmatize(tokens)

    return ' '.join(tokens)

def stemming(strng, object):
    tokens = strng.split()
    stemmed = [object.stem(word) for word in tokens]
    return ' '.join(stemmed)

def test_train_split(df, test_size=0.20, random_state=0, headline=True, content=False):
    if content:
        headline = False

    if headline:
        content = False

    if random_state != 0:
        np.random.seed(random_state)

    shuffled = df.sample(frac=1)
    shuffled.reset_index(drop=True, inplace=True)

    test_index = int(np.floor(df.shape[0] * test_size))

    test = shuffled[:test_index]

    if headline:
        test_data = test["Normalized Headline"].values

    if content:
        test_data = test["Normalized Content"].values

    y_test = test["Leaning Encode"].values

    train = shuffled[test_index:]

    if headline:
        train_data = train["Normalized Headline"].values

    if content:
        train_data = train["Normalized Content"].values

    y_train = train["Leaning Encode"].values

    return shuffled, test_data, y_test, train_data, y_train, test, train

def chunk(xs, n):
    k, m = divmod(len(xs), n)
    return [xs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def normalize_data(df, feature, stem=False, stemmer="porter", custom=False, stop_list=[], lemma=False):
    normalized = []

    for each in df[feature]:
        each = str(each)
        normed = normalize(each)
        if custom:
            stopped = stops(normed, custom=custom, stop_list=stop_list)

        elif lemma:
            stopped = stops(normed, lemma=lemma)

        else:
            stopped = stops(normed)

        #normalized.append(stops(normed))

        if stem:
            if stemmer == "porter":
                stem_obj = PorterStemmer()
            elif stemmer == "lancaster":
                stem_obj = LancasterStemmer()

            stemmed = stemming(stopped, stem_obj)

            normalized.append(stemmed)
        else:
            normalized.append(stopped)

    new_feature = "Normalized {0}".format(feature)
    df[new_feature] = pd.Series(normalized)

    return df

def kfold_cross_validation(clf, df, fold_count=5, repetitions=3, headline=True, content=False, verbose=False, random_state = 0):
    '''if headline:
        df = normalize_data(df, feature="Headline")

    if content:
        df = normalize_data(df, feature="Content")'''

    if content:
        headline = False

    if headline:
        content = False

    indices = list(range((df.shape[0])))

    accs = []
    precs = []
    recs = []
    f1s = []

    for _ in range(repetitions):

        if verbose:
            print("Repetition Number: ", _ + 1)

        if random_state != 0:
            np.random.seed(random_state+1)

        np.random.shuffle(indices)
        folds = chunk(indices, fold_count)

        fold_count = 0

        for fold in folds:
            fold_count += 1
            test_data = df.iloc[fold]
            train_indices = [idx not in fold for idx in indices]
            train_data = df.iloc[train_indices]

            test_data.reset_index(drop=True, inplace=True)
            train_data.reset_index(drop=True, inplace=True)

            y_test = test_data["Leaning Encode"].values
            y_train = train_data["Leaning Encode"].values

            if verbose:
                print("Fold Number: ", fold_count)
                print("Test Data Size: ", test_data.shape[0])
                print("Train Data Size: ", train_data.shape[0])
                print(" ")

            if headline:
                test_data = test_data["Normalized Headline"].values
                train_data = train_data["Normalized Headline"].values
            if content:
                test_data = test_data["Normalized Content"].values
                train_data = train_data["Normalized Content"].values


            train_features = tfidfvectorizer.fit_transform(train_data).toarray()
            test_features = tfidfvectorizer.transform(test_data).toarray()

            clf.fit(train_features, y_train)
            y_pred = clf.predict(test_features)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            accs.append(acc)
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)

    return [accs, precs, recs, f1s]

def cat_text(df, feature):
    text = df[feature].values
    concatenated = " ".join(text)
    return concatenated, concatenated.split()

def get_means(metrics):
    return [np.mean(metrics[0]), np.mean(metrics[1]), np.mean(metrics[2]), np.mean(metrics[3])]

def metrics_dictionary(dictionary, model, metrics):
    mean = get_means(metrics)
    dictionary[model] = [mean[0], mean[1], mean[2], mean[3]]
    return dictionary

def summarize_metrics(dictionary):
    print(" Acc. Prec. Rec. F1")

    for key in dictionary:
        metrics = dictionary[key]
        acc, prec, rec, f1 = format(metrics[0], ".4f"), format(metrics[1], ".4f"), format(metrics[2], ".4f"), format(
            metrics[3], ".4f")
        print("{0} {1} {2} {3} {4}".format(key, acc, prec, rec, f1))

def create_custom_stop_list(df, feature, n=50):
    text, text_split = cat_text(df=df, feature=feature)
    stopped = [word for word in text_split if word not in stopwords.words('english')]
    stopped = list(pd.Series(stopped).value_counts().head(n).keys())

    for i in range(len(stopped)):
        stopped[i] = normalize(stopped[i])

    return stopped

def stratify_data(df, feature="Source", n=1000):
    np.random.seed(500)
    df = df.groupby(feature, group_keys=False).apply(lambda x: x.sample(min(len(x), n)))
    df.reset_index(drop=True, inplace=True)

    return df

def value_counts_percentage(df, feature):
    counts = df[feature].value_counts()
    percent = df[feature].value_counts(normalize=True)
    percent100 = df[feature].value_counts(normalize=True).mul(100).round(2).astype(str)+"%"
    frame = pd.DataFrame({"Counts": counts,
                         "Fraction": percent,
                         "Percentage": percent100})

    frame.sort_values(by="Fraction", ascending=False, inplace=True)

    return frame

def generate_wordcloud(data, title=""):
    word_cloud = WordCloud(collocations=False, background_color='white').generate(data)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.show()

def word_count(df, feature):
    counts = []
    for i in range(df.shape[0]):
        data = df[feature].iloc[i]
        data_split = data.split()
        n = len(data_split)
        counts.append(n)
    count_data = pd.Series(counts)
    df["{0} Length".format(feature)] = count_data
    return df

def display_topics(model, feature_names, no_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                          for i in topic.argsort()[:-no_top_words - 1:-1]]))

def present_metrics(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("Accuracy: ", acc)
    print("Precision: ", prec)
    print("Recall: ", rec)
    print("F1: ", f1)

def present_cfm(y_test, y_pred, title="Confusion Matrix"):
    cfm = confusion_matrix(y_test, y_pred)
    ax = plt.axes()
    sns.heatmap(cfm, annot=True, fmt='g')
    ax.set_title(title)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()

def knn_validation_curve(train_features, test_features, y_train, y_test, klow=1, khigh=21):
    accs = []
    precs = []
    recs = []
    f1s = []
    ks = list(range(1, 21))

    for i in range(klow, khigh):
        clf = KNeighborsClassifier(n_neighbors=i)
        clf.fit(train_features, y_train)
        y_pred = clf.predict(test_features)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    return ks, accs, precs, recs, f1s

def baseline_cross_validation(df, fold_count=5, repetitions=1, random_state = 0):
    indices = list(range((df.shape[0])))

    accs = []
    precs = []
    recs = []
    f1s = []

    for _ in range(repetitions):
        if random_state != 0:
            np.random.seed(random_state+1)

        np.random.shuffle(indices)
        folds = chunk(indices, fold_count)

        fold_count = 0

        for fold in folds:
            fold_count += 1
            test_data = df.iloc[fold]
            train_indices = [idx not in fold for idx in indices]
            train_data = df.iloc[train_indices]

            test_data.reset_index(drop=True, inplace=True)
            train_data.reset_index(drop=True, inplace=True)

            majority = train_data["Leaning Encode"].value_counts().sort_index(ascending=False).index[0]
            y_pred = [majority] * test_data.shape[0]
            y_test = test_data["Leaning Encode"].values

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            accs.append(acc)
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)

    return [accs, precs, recs, f1s]
