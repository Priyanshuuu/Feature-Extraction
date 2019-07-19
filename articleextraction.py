# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 05:22:38 2019

@author: ishan
"""
import functions
import csv
import pandas as pd
soup_data = functions.link_opener('https://thehackernews.com/')
import warnings
warnings.filterwarnings(action='ignore')

import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import ShuffleSplit
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support as score
import time

# headings extraction
topic_list=[]
topic_link_list=[]
first_cat1=[]
sec_cat1=[]
thr_cat1=[]
first_cat1_dates=[]
sec_cat1_dates=[]
thr_cat1_dates=[]
cat=[]
dates=[]
articles=[]

topics=soup_data.find_all("li", {"class","show-menu"})
l=len(topics)
url='https://thehackernews.com/'
for topic in range(l) :
    topicy= topics[topic].a.text
    topic_list.append(topicy)
    topic_list=topic_list[0:3]
    link=url+topics[topic].a['href']
    topic_link_list.append(link)
    masterl=topic_link_list[0:3]
    for topicl in range(len(masterl)):
        if topicl==0:
            first_cat1.append(functions.link_opener(masterl[topicl]))
        elif topicl==1:
            sec_cat1.append(functions.link_opener(masterl[topicl]))
        else :
            thr_cat1.append(functions.link_opener(masterl[topicl]))

master=[]  
dic={}
dic2={}
dic3={}
gen_list = [first_cat1[0], sec_cat1[0], thr_cat1[0]]


    
for ge in range(1,len(gen_list)):
    if ge==0:
        # print(functions.headline(gen_list[ge]))
        # print(functions.Date(gen_list[ge]))
        fpage02 = functions.nextpag(gen_list[ge])
        # print(functions.headline(fpage2))
        # print(functions.Date(fpage2))
        fpage03 = functions.nextpag(fpage02)
        # print(functions.headline(fpage3))
        # print(functions.Date(fpage3))        
        fpage04 = functions.nextpag(fpage03)
       
        dic["arcticle_text1"]=functions.article(gen_list[ge])
        dic["arcticle_text2"]=functions.article(fpage02)
        dic["arcticle_text3"]=functions.article(fpage03)
        dic["arcticle_text4"]=functions.article(fpage04)
       
        
        dic["head1"] = functions.headline(gen_list[ge])
        dic["date1"] = functions.Date(gen_list[ge])
        dic["Cat1"]=functions.cat_li1(functions.headline(gen_list[ge]))
        #for he in range(len(dic["head1"])):
         #   dic.setdefault(cat, []).append("Misc")
        #functions.cat_pri(dic,"head1",topic_list)
        dic["head2"] = functions.headline(fpage02)
        dic["date2"] = functions.Date(fpage02)
        dic["Cat2"]=functions.cat_li1(functions.headline(gen_list[ge]))
        #for he in range(len(dic["head2"])):
        #    dic["category"]="Misc"
        #functions.cat_pri(dic,"head1",topic_list)    
        
        dic["head3"] = functions.headline(fpage03)
        dic["date3"] = functions.Date(fpage03)
        dic["Cat3"]=functions.cat_li1(functions.headline(gen_list[ge]))
        #for he in range(len(dic["head3"])):
        #    dic["category"].append("Misc")
        #functions.cat_pri(dic,"head1",topic_list)
        
        dic["head4"] = functions.headline(fpage04)
        dic["date4"] = functions.Date(fpage04)
        dic["Cat4"]=functions.cat_li1(functions.headline(gen_list[ge]))
        
        
        
        master.append(dic)
        #master.append(functions.cat_li1(functions.headline(gen_list[ge])))
        """
        for x in range(len(dic["head1"])):
            f.write(dic["head1"][x]+"\n")
        for x in range(len(
        """
    elif ge==1:
        fpage12=functions.nextpag(gen_list[ge])
        fpage13=functions.nextpag(fpage12)
        fpage14 = functions.nextpag(fpage13)
        fpage15 = functions.nextpag(fpage14)
        fpage16 = functions.nextpag(fpage15)
        fpage17 = functions.nextpag(fpage16)
        fpage18 = functions.nextpag(fpage17)
        fpage19 = functions.nextpag(fpage18)
        fpage20 = functions.nextpag(fpage19)
        linkoop= functions.link_finder2(gen_list[ge])
        for loop in range(len(linkoo)):
            rough.append(functions.article(functions.link_opener(linkoo[loop])))
            #print(functions.article(functions.link_opener(linkoo[loo])))
        dic2["art"]=rough    
        """
        dic2["arcticle_text1"]=functions.article(gen_list[ge])
        dic2["arcticle_text2"]=functions.article(fpage12)
        dic2["arcticle_text3"]=functions.article(fpage13)
        dic2["arcticle_text4"]=functions.article(fpage14)
        dic2["arcticle_text5"]=functions.article(fpage15)
        dic2["arcticle_text6"]=functions.article(fpage16)
        dic2["arcticle_text7"]=functions.article(fpage17)
        dic2["arcticle_text8"]=functions.article(fpage18)
        dic2["arcticle_text9"]=functions.article(fpage19)
        dic2["arcticle_text010"]=functions.article(fpage20)
        """
        
        dic2["head1"] = functions.headline(gen_list[ge])
        dic2["date1"] = functions.Date(gen_list[ge])
        dic2["Cat1"]=functions.cat_li2(functions.headline(gen_list[ge]))
        #for he in range(len(dic2["head1"])):
        #    dic["category"].append("Data Breach")
        dic2["head2"]=functions.headline(fpage12)
        dic2["date2"]=functions.Date(fpage12)
        dic2["Cat2"]=functions.cat_li2(functions.headline(gen_list[ge]))
        #for he in range(len(dic2["head2"])):
        #    dic["category"].append("Data Breach")
        dic2["head3"]=functions.headline(fpage13)
        dic2["date3"]=functions.Date(fpage13)
        dic2["Cat3"]=functions.cat_li2(functions.headline(gen_list[ge]))
        #for he in range(len(dic2["head3"])):
        #    dic["category"].append("Data Breach")
        
        dic2["head4"]=functions.headline(fpage14)
        dic2["date4"]=functions.Date(fpage14)
        dic2["Cat4"]=functions.cat_li2(functions.headline(gen_list[ge]))
        
        dic2["head5"]=functions.headline(fpage15)
        dic2["date5"]=functions.Date(fpage15)
        dic2["Cat5"]=functions.cat_li2(functions.headline(gen_list[ge]))
        
        dic2["head6"]=functions.headline(fpage16)
        dic2["date6"]=functions.Date(fpage16)
        dic2["Cat6"]=functions.cat_li2(functions.headline(gen_list[ge]))
        
        dic2["head7"]=functions.headline(fpage17)
        dic2["date7"]=functions.Date(fpage17)
        dic2["Cat7"]=functions.cat_li2(functions.headline(gen_list[ge]))
        
        dic2["head8"]=functions.headline(fpage18)
        dic2["date8"]=functions.Date(fpage18)
        dic2["Cat8"]=functions.cat_li2(functions.headline(gen_list[ge]))
        
        dic2["head9"]=functions.headline(fpage19)
        dic2["date9"]=functions.Date(fpage19)
        dic2["Cat9"]=functions.cat_li2(functions.headline(gen_list[ge]))
        
        dic2["head010"]=functions.headline(fpage20)
        dic2["date010"]=functions.Date(fpage20)
        dic2["Cat010"]=functions.cat_li2(functions.headline(gen_list[ge]))
        
        
        master.append(dic2)
        #master.append(functions.cat_li2(functions.headline(gen_list[ge])))
    
    else :
        fpage22=functions.nextpag(gen_list[ge])
        fpage23=functions.nextpag(fpage22)
        fpage24 = functions.nextpag(fpage23)
        fpage25 = functions.nextpag(fpage24)
        fpage26 = functions.nextpag(fpage25)
        fpage27 = functions.nextpag(fpage26)
        fpage28 = functions.nextpag(fpage27)
        fpage29 = functions.nextpag(fpage28)
        fpage30 = functions.nextpag(fpage29)
        linkoop= functions.link_finder2(gen_list[ge])
        for loop in range(len(linkoo)):
            rough.append(functions.article(functions.link_opener(linkoo[loop])))
            #print(functions.article(functions.link_opener(linkoo[loo])))
        dic3["art"]=rough    
        """
        dic3["arcticle_text1"]=functions.article(gen_list[ge])
        dic3["arcticle_text2"]=functions.article(fpage22)
        dic3["arcticle_text3"]=functions.article(fpage23)
        dic3["arcticle_text4"]=functions.article(fpage24)
        dic3["arcticle_text5"]=functions.article(fpage25)
        dic3["arcticle_text6"]=functions.article(fpage26)
        dic3["arcticle_text7"]=functions.article(fpage27)
        dic3["arcticle_text8"]=functions.article(fpage28)
        dic3["arcticle_text9"]=functions.article(fpage29)
        dic3["arcticle_text010"]=functions.article(fpage30)
        """
        
        dic3["head1"]=functions.headline(gen_list[ge])
        dic3["date1"]=functions.Date(gen_list[ge])
        dic3["Cat1"]=functions.cat_li3(functions.headline(gen_list[ge]))
        #for he in range(len(dic3["head1"])):
        #    dic["category"].append("Data Breach")
        dic3["head2"]=functions.headline(fpage22)
        dic3["date2"]=functions.Date(fpage22)
        dic3["Cat2"]=functions.cat_li3(functions.headline(gen_list[ge]))
        #for he in range(len(dic3["head2"])):
        #    dic["category"].append("Data Breach")
        dic3["head3"]=functions.headline(fpage23)
        dic3["date3"]=functions.Date(fpage23)
        dic3["Cat3"]=functions.cat_li3(functions.headline(gen_list[ge]))
        #for he in range(len(dic3["head3"])):
        #    dic["category"].append("Data Breach")
        dic3["head4"]=functions.headline(fpage24)
        dic3["date4"]=functions.Date(fpage24)
        dic3["Cat4"]=functions.cat_li3(functions.headline(gen_list[ge]))
        
        
        dic3["head5"]=functions.headline(fpage25)
        dic3["date5"]=functions.Date(fpage25)
        dic3["Cat5"]=functions.cat_li3(functions.headline(gen_list[ge]))
        
        
        dic3["head6"]=functions.headline(fpage26)
        dic3["date6"]=functions.Date(fpage26)
        dic3["Cat6"]=functions.cat_li3(functions.headline(gen_list[ge]))
        
        
        dic3["head7"]=functions.headline(fpage27)
        dic3["date7"]=functions.Date(fpage27)
        dic3["Cat7"]=functions.cat_li3(functions.headline(gen_list[ge]))
        
        
        dic3["head8"]=functions.headline(fpage28)
        dic3["date8"]=functions.Date(fpage28)
        dic3["Cat8"]=functions.cat_li3(functions.headline(gen_list[ge]))
        
        
        dic3["head9"]=functions.headline(fpage29)
        dic3["date9"]=functions.Date(fpage29)
        dic3["Cat9"]=functions.cat_li3(functions.headline(gen_list[ge]))
        
        
        dic3["head010"]=functions.headline(fpage30)
        dic3["date010"]=functions.Date(fpage30)
        dic3["Cat010"]=functions.cat_li3(functions.headline(gen_list[ge]))
        
        master.append(dic3)
        #master.append(functions.cat_li3(functions.headline(gen_list[ge])))
topi=dic2["head1"]+dic2["head2"]+dic2["head3"]+dic2["head4"]+dic2["head5"]+dic2["head6"]+dic2["head7"]+dic2["head8"]+dic2["head9"]+dic2["head010"]+dic3["head1"]+dic3["head2"]+dic3["head3"]+dic3["head4"]+dic3["head5"]+dic3["head6"]+dic3["head7"]+dic3["head8"]+dic3["head9"]+dic3["head010"]
cati=dic2["Cat1"]+dic2["Cat2"]+dic2["Cat3"]+dic2["Cat4"]+dic2["Cat5"]+dic2["Cat6"]+dic2["Cat7"]+dic2["Cat8"]+dic2["Cat9"]+dic2["Cat010"]+dic3["Cat1"]+dic3["Cat2"]+dic3["Cat3"]+dic3["Cat4"]+dic3["Cat5"]+dic3["Cat6"]+dic3["Cat7"]+dic3["Cat8"]+dic3["Cat9"]+dic3["Cat010"]
#text_arc=dic2["arcticle_text1"]+dic2["arcticle_text2"]+dic2["arcticle_text3"]+dic2["arcticle_text4"]+dic2["arcticle_text5"]+dic2["arcticle_text6"]+dic2["arcticle_text7"]+dic2["arcticle_text8"]+dic2["arcticle_text9"]+dic2["arcticle_text010"]+dic3["arcticle_text1"]+dic3["arcticle_text2"]+dic3["arcticle_text3"]+dic3["arcticle_text4"]+dic3["arcticle_text5"]+dic3["arcticle_text6"]+dic3["arcticle_text7"]+dic3["arcticle_text8"]+dic3["arcticle_text9"]+dic3["arcticle_text010"]
text_arc=dic2["art"]+dic3["art"]
cati=cati[0:390]
text_arc=text_arc[0:390]
"""
catii=cati[0:99]
topi=topi[0:99]
text_arc=text_arc[0:99]
"""
dic_link={}
dic_link2={}
dic_link3={}

master_link=[]
for ge in range(1,len(gen_list)):
    if ge==0:
        dic_link["link1"]=functions.headlinks(gen_list[ge])
        # print(functions.headline(gen_list[ge]))
        # print(functions.Date(gen_list[ge]))
        fpage02 = functions.nextpag(gen_list[ge])
        # print(functions.headline(fpage2))
        # print(functions.Date(fpage2))
        fpage03 = functions.nextpag(fpage02)
        dic_link["link2"]=functions.headlinks(fpage02)
        dic_link["link3"]=functions.headlinks(fpage03)
        master_link.append(dic_link)
    elif ge==1:
        dic_link2["link1"]=functions.headlinks(gen_list[ge])
        # print(functions.headline(gen_list[ge]))
        # print(functions.Date(gen_list[ge]))
        fpage12 = functions.nextpag(gen_list[ge])
        # print(functions.headline(fpage2))
        # print(functions.Date(fpage2))
        fpage13 = functions.nextpag(fpage12)
        dic_link2["link2"]=functions.headlinks(fpage12)
        dic_link2["link3"]=functions.headlinks(fpage13)
        master_link.append(dic_link2)
    else :
        dic_link3["link1"]=functions.headlinks(gen_list[ge])
        # print(functions.headline(gen_list[ge]))
        # print(functions.Date(gen_list[ge]))
        fpage22 = functions.nextpag(gen_list[ge])
        # print(functions.headline(fpage2))
        # print(functions.Date(fpage2))
        fpage23 = functions.nextpag(fpage22)
        dic_link3["link2"]=functions.headlinks(fpage22)
        dic_link3["link3"]=functions.headlinks(fpage23)
        master_link.append(dic_link3)
linky=dic_link2["link2"]+dic_link2["link3"]+dic_link2["link1"]+dic_link3["link2"]+dic_link3["link3"]+dic_link3["link3"]
master_linkpd=pd.DataFrame(linky)        
linky=linky[0:100]

#print(dic == dic2)
#print(dic2 == dic3)
masterpd=pd.DataFrame(cati)
#masterpd["headlines"]=topi
masterpd["sabkatext"]=text_arc
masterpd.columns=['category','sabkatext']
masterpd_2=pd.DataFrame(cati)
masterpd_2["headlines"]=topi
#masterpd_2["links"]=linky
masterpd_2.columns=['category','headlines']

filename="tables3.txt"
f=open(filename,"w",encoding="utf-8")
masterpd_2.to_csv(f,sep=',',index=False)
f.close()

filename="tables2.txt"
f=open(filename,"w",encoding="utf-8")
masterpd.to_csv(f,sep=',',index=False)
f.close()
#masterpd['body_len'] = masterpd['sabkatext'].apply(lambda x: len(x) - x.count(" "))
import string

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

#masterpd['punct%'] = masterpd['sabkatext'].apply(lambda x: count_punct(x))

#from matplotlib import pyplot
#import numpy as np
masterpd['tokens'] = masterpd['sabkatext'].apply(lambda x: token(x.lower()))
stopword =nltk.corpus.stopwords.words('english')

masterpd['tokensstop'] = masterpd['tokens'].apply(lambda x: rem(x))

ps = nltk.PorterStemmer()
masterpd['tokensstopt'] = masterpd['tokensstop'].apply(lambda x: (x))


masterpd['tokensstopt2'] = masterpd['tokensstopt'].apply(lambda x: stemmin(x))

wn = nltk.WordNetLemmatizer()

    

masterpd['tokensstoplemm'] = masterpd['tokensstopt2'].apply(lambda x: lemma(x))


X=masterpd.category
y=masterpd.tokensstoplemm
ss = ShuffleSplit(n_splits=10, test_size=0.2)


for train_index, test_index in ss.split(X):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(masterpd['sabkatext'])
xtrain_tfidf =  tfidf_vect.transform(X_train)
xvalid_tfidf =  tfidf_vect.transform(X_test)
# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(masterpd['sabkatext'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(X_train)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions,y_test )


def create_model_architecture(input_size):
    # create input layer 
    input_layer = layers.Input((input_size, ), sparse=True)
    
    # create hidden layer
    hidden_layer = layers.Dense(100, activation="relu")(input_layer)
    
    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier 

classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])
accuracy = train_model(classifier, xtrain_tfidf_ngram, y_train, xvalid_tfidf_ngram, is_neural_net=True)
print ("NN, Ngram Level TF IDF Vectors",  accuracy)
#classifier.predict(xtrain_tfidf_ngram)

import trendfunc
import pandas as pd
soup_data = trendfunc.link_opener('https://blog.trendmicro.com/category/vulnerabilities/')
import warnings
warnings.filterwarnings(action='ignore')


# headings extraction
topic_list=[]
topic_link_list=[]
first_cat1=[]
sec_cat1=[]
thr_cat1=[]
first_cat1_dates=[]
sec_cat1_dates=[]
thr_cat1_dates=[]
cat=[]
dates=[]
articles=[]

topics=soup_data.find_all("li", {"class","menu-item menu-item-type-taxonomy menu-item-object-category current-menu-item"})
l=len(topics)
url='https://thehackernews.com/'
for topic in range(l) :
    topicy= topics[topic].a.text
    topic_list.append(topicy)
    topic_list=topic_list[0:1]
    link=topics[topic].a['href']
    topic_link_list.append(link)
    masterl=topic_link_list[0:1]
    for topicl in range(len(masterl)):
        if topicl==0:
            first_cat1.append(trendfunc.link_opener(masterl[topicl]))

master=[]  
dic={}
dic2={}
dic3={}
gen_list = [first_cat1[0]]


    
for ge in range(len(gen_list)):
    if ge==0:
        dic["arcticle_text1"]=trendfunc.article(gen_list[ge])
        master.append(dic)
text_arc1=dic["arcticle_text1"]

def newdata_predictions(new_data):
    new_data =  tfidf_vect_ngram.transform(new_data)
    predictions = classifier.predict(new_data)
    return predictions
    
target_new_data = newdata_predictions(text_arc1) 
print(target_new_data)
    
    
