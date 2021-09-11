from elasticsearch import Elasticsearch
import numpy as np
import pandas as pd
from elasticsearch import helpers
from elasticsearch.helpers import bulk
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import re
import argparse
import requests
import re
parser = argparse.ArgumentParser(description='Data manipulation')
parser.add_argument('--id',help='enter the id')
parser.add_argument('--cause',help='Enter ur cause')
parser.add_argument('--symptom',help='Enter ur  symptom')
parser.add_argument('--measure',help='Enter ur measure')
args=parser.parse_args()

cause=args.cause
symptom=args.symptom
measure=args.measure
index_id=int(args.id) 
#loading the universal sentence encoder
model = hub.load(r"universal-sentence-encoder_4")
## Concecting to the elastic Data_base
ES_HOST = {"host" : "localhost", "port" : 9200}
es = Elasticsearch(hosts = [ES_HOST])
index_name="kubota" #Index name
def text_to_vect(text):
    vector=tf.make_ndarray(tf.make_tensor_proto(model([text]))).tolist()[0]
    return vector


def modifiy_data(index_id,cause,symptom,measure):
    
    cause=cause
    symptom=symptom
    measure=measure
    data={"doc": {"id":id,"cause_text":cause,"cause_vector":text_to_vect(cause),"measure_text":measure,"measure_vector":text_to_vect(measure),"symptom_text":symptom,"symptom_vector":text_to_vect(symptom)}}
    es.update(index=index_name, id=index_id, body=data)
    print("Successfully_updated at ID",id)  

modifiy_data(index_id,cause,symptom,measure)