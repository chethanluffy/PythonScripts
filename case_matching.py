import argparse 
from sklearn.metrics import pairwise_distances
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
from collections import Counter 
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import concurrent.futures
from nltk.corpus import stopwords
from prettytable import PrettyTable
from tabulate import tabulate
from beautifultable import BeautifulTable

stop_words = set(stopwords.words('english'))

def nlp_preprocessing(total_text, index, column):


    
    total_text=str(total_text)
    if type(total_text) is not int:
        string = ""
        for words in total_text.split():
            # remove the special chars in review like '"#$@!%^&*()_+-~?>< etc.
            word = ("".join(e for e in words if e.isalnum()))
            # Conver all letters to lower-case
            word = word.lower()

            if word =='nan':
                word =" "
            # stop-word remova
            if not word in stop_words:
                string += word + " "

        if len(string)<=8:

            main_df[column][index] = " "

        else:
            main_df[column][index] = string




def calculate_distance(keywords,vectorizer_features,metric):
    if metric=='euclidean':
        distance = pairwise_distances(vectorizer_features, keywords,metric='euclidean',n_jobs=-1)
        return distance
    if metric=='pair_wise':
        distance = pairwise_distances(vectorizer_features, keywords,n_jobs=-1)
        return distance
    if metric=='cosine':
        distance = pairwise_distances(vectorizer_features,keywords, metric='cosine',n_jobs=-1)

        return distance
def clean_tranform_features(vocab): 
    main_df['Primary Failure Part Name'] = main_df['Primary Failure Part Name'].str.replace(r"\)"," ")
    main_df['Primary Failure Part Name'] = main_df['Primary Failure Part Name'].str.replace(r"\("," ")
    main_df["total_text"] = main_df["Subject"].map(str) + " " + \
                        main_df["Outline (Symptom)"].map(str) +" " + \
                        main_df["Cuase (Defect)"].map(str) + " " + \
                        main_df["Measure (Remedy)"].map(str)+ " "+ \
                        main_df["Primary Failure Part Name"].map(str)

    for index, row in main_df.iterrows():
        nlp_preprocessing(row['total_text'], index,'total_text')

    main_df['total_text']=main_df['total_text'].replace('', np.nan)
    main_df['total_text']=main_df['total_text'].replace(' ', np.nan)

    main_df['total_text'].drop_duplicates(inplace=True)
    main_df['total_text'].dropna(inplace=True)
    # cases=main_df['total_text'].copy()
    

    cases=main_df['total_text']
    cases.to_csv("test.csv")

    # cases=cases.replace(' ', np.nan)
    # cases=cases.str.lower()

    # cases=cases.str.strip()
    # cases.drop_duplicates(inplace=True)
    # cases.dropna(inplace=True)

    return cases



# this function will add the vectors of each word and returns the avg vector of given sentance
def build_avg_vec(sentence, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0   
    for word in sentence.split():
        nwords += 1
        if word in vocab:            
            featureVec = np.add(featureVec, model[word])
    if(nwords>0):
        featureVec = np.divide(featureVec, nwords)
    # returns the avg vector of given sentance, its of shape (1, 300)
    return featureVec




    
def build_tfidf_vectors(data,glove_words,tfidf_words,dictionary):
    tfidf_w2v_vectors = []; 
    for sentence in data: 
        vector = np.zeros(300) 
        tf_idf_weight =0; 
        for word in sentence.split(): 
            if (word in glove_words) and (word in tfidf_words):
                vec = model[word]
                tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
                vector += (vec * tf_idf) # calculating tfidf weighted w2v
                tf_idf_weight += tf_idf
        if tf_idf_weight != 0:
            vector /= tf_idf_weight
        tfidf_w2v_vectors.append(vector)
    return tfidf_w2v_vectors

#tfidf weighted word2vec




    
def build_keyword_vectors(data,tfidf_words,dictionary):
    keyword_vectors = []; 
    vector = np.zeros(300) 
    tf_idf_weight =0; 
    for word in data.split():
        if (word in glove_words) and (word in tfidf_words):
            vec = model[word] 
            tf_idf = dictionary[word]*(data.count(word)/len(data.split())) 
            vector += (vec * tf_idf) 
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    keyword_vectors.append(vector)
    return keyword_vectors

def tfidf_w2vec(input_keywords,count,metric,cases,args):

    new_df=pd.DataFrame()
    new_df['cleaned data']=cases

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    tfidf_vectorizer_features = tfidf_vectorizer.fit_transform(cases)

    dictionary = dict(zip(tfidf_vectorizer.get_feature_names(), list(tfidf_vectorizer.idf_)))
    tfidf_words = set(tfidf_vectorizer.get_feature_names())
    tfidf_w2v_vectors=build_tfidf_vectors(cases,vocab,tfidf_words,dictionary)
    tfidf_w2v_vectors=np.array(tfidf_w2v_vectors)


    keyword_vector=build_keyword_vectors(input_keywords,tfidf_words,dictionary)
    keyword_vector=np.array(keyword_vector)
    pairwise_dist = calculate_distance(keyword_vector,tfidf_w2v_vectors,metric)
    indices = np.argsort(pairwise_dist.flatten())[0:count]
    df_indices = list(new_df.index[indices])

    for i in range(0,len(indices)):



        
        
        table.append_row([main_df['Subject'].loc[df_indices[i]],main_df['Outline (Symptom)'].loc[df_indices[i]],main_df['Cuase (Defect)'].loc[df_indices[i]],main_df['Measure (Remedy)'].loc[df_indices[i]],main_df['Primary Failure Part Name'].loc[df_indices[i]]])
        
    print(table)                               
    
def word2vec(input_keywords,count,metric,cases,args):
    # to_csv=[]
    # temp_df=pd.DataFrame()
    w2v_title = []
    for i in cases:
        w2v_title.append(build_avg_vec(i, 300))
    w2v_title = np.array(w2v_title)
    new_df=pd.DataFrame()
    new_df['cleaned data']=cases

    keyword_vector=build_avg_vec(input_keywords, 300)
    keyword_vector=keyword_vector.reshape(1,-1)
    pairwise_dist = calculate_distance(keyword_vector,w2v_title,metric)
    indices = np.argsort(pairwise_dist.flatten())[0:count]
    df_indices = list(new_df.index[indices])
    for i in range(0,len(indices)):
        table.append_row([main_df['Subject'].loc[df_indices[i]],main_df['Outline (Symptom)'].loc[df_indices[i]],main_df['Cuase (Defect)'].loc[df_indices[i]],main_df['Measure (Remedy)'].loc[df_indices[i]],main_df['Primary Failure Part Name'].loc[df_indices[i]]])
        
    print(table)


def tfidf(input_keywords,count,metric,cases,args):
    to_csv=[]
    temp_df=pd.DataFrame()
    new_df=pd.DataFrame()
    
    new_df['cleaned data']=cases
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_vectorizer_features = tfidf_vectorizer.fit_transform(cases)
    print(tfidf_vectorizer.get_feature_names())

    if not input_keywords:
        print('no keywords passed')
    else:
        keywords=tfidf_vectorizer.transform([input_keywords])
        pairwise_dist = calculate_distance(keywords,tfidf_vectorizer_features,metric)
        indices=np.argsort(pairwise_dist.flatten())[0:int(count)]
        df_indices = list(new_df['cleaned data'].index[indices])
        for i in range(0,len(indices)):
            table.append_row([main_df['Subject'].loc[df_indices[i]],main_df['Outline (Symptom)'].loc[df_indices[i]],main_df['Cuase (Defect)'].loc[df_indices[i]],main_df['Measure (Remedy)'].loc[df_indices[i]],main_df['Primary Failure Part Name'].loc[df_indices[i]]])
        
        print(table)
def bow(input_keywords,count,metric,cases,args):

    new_df=pd.DataFrame()
    new_df['cleaned data']=cases
    bag_of_title_vectorizer = CountVectorizer(min_df = 0,stop_words='english')
    bag_title_features = bag_of_title_vectorizer.fit_transform(cases)
    if not input_keywords:
        print('no keywords passed')
    else:
        keywords=bag_of_title_vectorizer.transform([input_keywords])
        pairwise_dist = calculate_distance(keywords,bag_title_features,metric)
        indices=np.argsort(pairwise_dist.flatten())[0:int(count)]
        df_indices = list(new_df['cleaned data'].index[indices])
        for i in range(0,len(indices)):
            table.append_row([main_df['Subject'].loc[df_indices[i]],main_df['Outline (Symptom)'].loc[df_indices[i]],main_df['Cuase (Defect)'].loc[df_indices[i]],main_df['Measure (Remedy)'].loc[df_indices[i]],main_df['Primary Failure Part Name'].loc[df_indices[i]]])
        
        print(table)

            
def call_model(args,cases):
    keywords=args.keywords
    keywords=" ".join(keywords)
    model=args.model
    count=args.count
    metric=args.metric
    if model =='bow':
        bow(keywords,count,metric,cases,args)
    elif model =='tfidf':
        tfidf(keywords,count,metric,cases,args)
    elif model =='word2vec':
        word2vec(keywords,count,metric,cases,args)
    elif model =='tfidf_w2vec':
        tfidf_w2vec(keywords,count,metric,cases,args)
    else :
        print('Incorrect Model Name')



            
if __name__=="__main__":

    colnames=['Subject','Outline (Symptom)','Cuase (Defect)','Measure (Remedy)','Primary Failure Part Name']
    table = BeautifulTable(max_width=200)
    table.column_headers = colnames



    
    parser= argparse.ArgumentParser()
    parser.add_argument('--keywords',type=str,nargs='*',
                    help='input one or more keywords ')
    parser.add_argument('--model',type=str,
                    help='enter "bow" or "tfidf" or "word2vec" or "tfidf_w2vec"')
    parser.add_argument('--count',type=int,
                    help='enetr a valid intiger')
    parser.add_argument('--metric',type=str,
                    help="enter a distannce metric enter 'euclidean' or 'cosine' 'pair_wise' ")
    parser.add_argument('--csv_path',type=str,
                    help='enetr a valid path to warranty file')
    # parser.add_argument('--colum_name',type=str,
    #                 help='enetr a valid path to warranty file')
    parser.add_argument('--word2vec_file_path',type=str,
                    help='path to word to vec pickle file')
    args = parser.parse_args()
    pickle_path=str(args.word2vec_file_path)
    main_df=pd.read_csv(args.csv_path,usecols=colnames)
    with open(pickle_path, 'rb') as handle:
        model = pickle.load(handle)
        vocab = model.keys()
    cases=clean_tranform_features(vocab)
    glove_words=model.keys()
    call_model(args,cases)