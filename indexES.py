import json
import time
import sys
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import csv
import tensorflow as tf
import tensorflow_hub as hub


# connect to ES on localhost on port 9200
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
if es.ping():
	print('Connected to ES!')
else:
	print('Could not connect!')
	sys.exit()

print("*********************************************************************************");


# index in ES = DB in an RDBMS
# Read each question and index into an index called questions
# Indexing only titles for this example to improve speed. In practice, its good to index CONCATENATE(title+body)
# Define the index


#Refer: https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping.html
# Mapping: Structure of the index
# Property/Field: name and type  
b = {"mappings": {
  	"properties": {
    		"outline": {
      			"type": "text"
    		},
    		    "cause": {
      			"type": "text"
    		},
    		    "measure": {
      			"type": "text"
    		},
    		"outline_vector": {
      			"type": "dense_vector",
      			"dims": 512
		},
		    		"cause_vector": {
      			"type": "dense_vector",
      			"dims": 512
		},
				    "measure_vector": {
      			"type": "dense_vector",
      			"dims": 512
		}
	}
     }
   }


ret = es.indices.create(index='warranty-index', ignore=400, body=b) #400 caused by IndexAlreadyExistsException, 
print(json.dumps(ret,indent=4))

# TRY this in browser: http://localhost:9200/questions-index

print("*********************************************************************************");



#load USE4 model

embed = hub.load("universal-sentence-encoder_4")




# CONSTANTS
NUM_QUESTIONS_INDEXED = 200000

# Col-Names: Id,OwnerUserId,CreationDate,ClosedDate,Score,Title,Body
cnt=0

with open('document.csv',encoding="latin1") as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',' )
	next(readCSV, None) 
	i=1 # skip the headers 
	for row in readCSV:
		#print(row[0], row[5])
		doc_id = i;
		i=i+1
		outline = row[0];
		cause = row[1]
		measure = row[2]

		data=outline+cause+measure
		outline_vec = tf.make_ndarray(tf.make_tensor_proto(embed([data]))).tolist()[0]		
		cause_vec = tf.make_ndarray(tf.make_tensor_proto(embed([cause]))).tolist()[0]
		measure_vec = tf.make_ndarray(tf.make_tensor_proto(embed([measure]))).tolist()[0]


		b = {"outline":outline,
			"outline_vector":outline_vec,
			"cause":cause,
			"cause_vector":cause_vec,
			"measure":measure,
			"measure_vector":measure_vec,

			}	
				
		
		res = es.index(index="warranty-index", id=doc_id, body=b)
		#print(res)
		

		# keep count of # rows processed
		cnt += 1
		if cnt%100==0:
			print(cnt)
		
		if cnt == NUM_QUESTIONS_INDEXED:
			break;

	print("Completed indexing....")

	print("*********************************************************************************");
	
