from pymongo import MongoClient
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import MONGODB_URI, MONGODB_DATABASE_NAME

def get_mongo_client(app_name=None):
    return MongoClient(MONGODB_URI, appname=app_name)

def get_mongo_db(client):
    return client[MONGODB_DATABASE_NAME]