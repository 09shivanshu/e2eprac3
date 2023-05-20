import pymongo
import pandas as pd
import json
from dataclasses import dataclass
import os

# Provide the mongodb localhost url to connect python to mongodb.

@dataclass
class EnviromentVariable:
    mongo_db_url:str = os.getenv("MONGO_DB_URL")
    aws_access_key_id:str = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key:str = os.getenv("AWS_SECRET_ACCESS_KEY")

env_var = EnviromentVariable()

mongoclient = pymongo.MongoClient(env_var.mongo_db_url)


