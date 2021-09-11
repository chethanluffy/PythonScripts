from elasticsearch import Elasticsearch
from elasticsearch import helpers
from elasticsearch.helpers import bulk
import argparse
parser = argparse.ArgumentParser(description='Data manipulation')
parser.add_argument('--id',help='Enter ur Query')
args=parser.parse_args()
index_of_id=int(args.id)
## Concecting to the elastic Data_base
ES_HOST = {"host" : "localhost", "port" : 9200}
es = Elasticsearch(hosts = [ES_HOST])
index_name="index" #Index name
def Delete_data(index_of_id):
    
    es.delete(index=index_name,id=index_of_id)
    print("Data Sucessfully Deleted at ID",index_of_id )

Delete_data(index_of_id)