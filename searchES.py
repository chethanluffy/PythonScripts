import json
import time
import sys
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import csv
import tensorflow as tf
import tensorflow_hub as hub
from beautifultable import BeautifulTable



def connect2ES():
    # connect to ES on localhost on port 9200
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    if es.ping():
            print('Connected to ES!')
    else:
            print('Could not connect!')
            sys.exit()

    print("*********************************************************************************");
    return es

def keywordSearch(es, q):
    #Search by Keywords
    # b={
    #         'query':{
    #             'match':{
    #                 "title":q
    #             }
        #         }
    table = BeautifulTable()
    table.column_headers = ['score','outline', 'cause','measure']
    b={"query": {"multi_match": {"query": q, "fields": ["outline", "cause","measure"]}}}
    res= es.search(index='warranty-index',body=b)
    print("Keyword Search:\n")

    for hit in res['hits']['hits']:
        table.append_row([hit['_score'],hit['_source']['outline'],hit['_source']['cause'],hit['_source']['cause']])

    print(table)

        # print(str(hit['_score']) + "\t" + hit['_source'] )
# ['_source']['outline']
    print("*********************************************************************************");

    return


# Search by Vec Similarity
def sentenceSimilaritybyNN(embed, es, sent):
    query_vector = tf.make_ndarray(tf.make_tensor_proto(embed([sent]))).tolist()[0]
    b = {"query" : {
                "script_score" : {
                    "query" : {
                        "match_all": {}
                    },
                    "script" : {
                        "source": "cosineSimilarity(params.query_vector, 'outline_vector') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
             }
        }


    #print(json.dumps(b,indent=4))
    res= es.search(index='warranty-index',body=b)
    table = BeautifulTable()
    table.column_headers = ['score','outline', 'cause','measure']
    print("Semantic Similarity Search:\n")
    for hit in res['hits']['hits']:
        table.append_row([hit['_score'],hit['_source']['outline'],hit['_source']['cause'],hit['_source']['cause']])
    print(table)
    print("*********************************************************************************");



if __name__=="__main__":

    es = connect2ES();
    embed = hub.load("/home/carlton/Downloads/universal-sentence-encoder_4")

    while(1):
        query=input("Enter a Query:");

        start = time.time()
        if query=="END":
            break;

        print("Query: " +query)
        keywordSearch(es, query)
        sentenceSimilaritybyNN(embed, es, query)

        end = time.time()
        print(end - start)
