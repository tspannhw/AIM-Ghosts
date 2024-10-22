import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import pandas as pd
import numpy as np
from pymilvus import MilvusClient
from PIL import Image
import json
import sentence_transformers
from sentence_transformers import SentenceTransformer
import requests
import os
import boto3
from botocore.client import Config
from datetime import datetime
from pymilvus import (
   utility,
   FieldSchema, CollectionSchema, DataType,
   Collection, AnnSearchRequest, RRFRanker, connections,
)
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction
from pymilvus.model.sparse import SpladeEmbeddingFunction
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# Multi
bge_m3 = BGEM3EmbeddingFunction(
    model_name = 'BAAI/bge-m3', 
    device = 'cpu', 
    use_fp16 = False
)

# BGE-M3 Sparse
dense_dim = bge_m3.dim["dense"] 

# Slade Sparse
splade_ef = SpladeEmbeddingFunction(
    model_name="naver/splade-cocondenser-selfdistil", 
    device="cpu"
)

# Load CLIP model
model = SentenceTransformer('clip-ViT-B-32')

# Big Text
textmodel = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)

### Constants
S3_URL = "http://192.168.1.166:9000"
MINIO_USER = 'minioadmin'
MINIO_PASSWORD = 'minioadmin'
MINIO_REGION = 'us-east-1'
BUCKET_NAME = "images"

### where to upload file
s3_client = boto3.client('s3',
                          endpoint_url=S3_URL,
                          config=boto3.session.Config(signature_version='s3v4'))
        

# Connect to Milvus Standalone on Docker
from pymilvus import connections
from pymilvus import utility
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection

MILVUS_URL = "http://192.168.1.166:19530" 

## MILVUS_URL = "./milvus_ghostsdemo.db"

# Connect to Zilliz Cloud with Public Endpoint and API Key
#client = MilvusClient(
#    uri=ZILLIZ_PUBLIC_ENDPOINT,
#    token=ZILLIZ_API_KEY)

milvus_client = MilvusClient(uri=MILVUS_URL)

COLLECTION = "ghosts"

DIMENSION = 512
TEXTDIMENSION = 768

ghost_classes = ["Fake", "Art", "TV", "Unclassified", "Class I", "Class II", "Class III", "Class IV", "Class V", "Class VI", "Class VII"]
categories = ["Ghost", "Deity", "Unstable", "Environmental", "Vathek", "Legend", "Video Game"]

connections.connect(uri=MILVUS_URL )

fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='ghostclass', dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name='filename', dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name='s3path', dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name='description', dtype=DataType.VARCHAR, max_length=65000),
    FieldSchema(name='category', dtype=DataType.VARCHAR, max_length=256), 
    FieldSchema(name='identification', dtype=DataType.VARCHAR, max_length=50),  
    FieldSchema(name='location', dtype=DataType.VARCHAR, max_length=256),  
    FieldSchema(name='country', dtype=DataType.VARCHAR, max_length=4),
    FieldSchema(name='latitude', dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name='longitude', dtype=DataType.VARCHAR, max_length=20),      
    FieldSchema(name='zipcode', dtype=DataType.VARCHAR, max_length=20),  
    FieldSchema(name='timestamp', dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name='s3timestamp', dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    FieldSchema(name="text_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="text_vector2", dtype=DataType.FLOAT_VECTOR, dim=TEXTDIMENSION),
    FieldSchema(name="text_vector3", dtype=DataType.FLOAT_VECTOR, dim=dense_dim)
]

# with partitioning
schema = CollectionSchema(fields=fields, partition_key_field="ghostclass", num_partitions=8 )

if milvus_client.has_collection(collection_name=COLLECTION):
    print(COLLECTION + " exists")
else:
    milvus_client.create_collection(COLLECTION, schema=schema,  auto_id=True)

    index_params = milvus_client.prepare_index_params()
    
    index_params.add_index(field_name = "id", index_type="STL_SORT")
    index_params.add_index(field_name = "ghostclass", index_type="TRIE")
    index_params.add_index(field_name = "zipcode",  index_type="INVERTED", index_name="inverted_index")
    index_params.add_index(field_name = "vector", metric_type="COSINE")
    index_params.add_index(field_name = "text_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    index_params.add_index(field_name = "text_vector2", metric_type="COSINE")
    index_params.add_index(field_name = "text_vector3", metric_type= "L2", index_type="IVF_FLAT", params = {"nlist": 128} )

    milvus_client.create_index( collection_name=COLLECTION, index_params=index_params)
    
    res = milvus_client.get_load_state( collection_name = COLLECTION)
    
    print(res)

milvus_client.load_collection(COLLECTION)
res = milvus_client.get_load_state( collection_name = COLLECTION )
print(res)

### for hybrid search
collection = Collection(name=COLLECTION, schema=schema)

### Start the page
st.set_page_config(layout="wide", page_title="Ghost Detection", page_icon="favicon.ico")

st.image("./pics/milvuslogo.png", width=150)
st.write("## Capture your ghost in Milvus")
st.sidebar.write("## Capture your ghost in Milvus ")

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Download the image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def save():
    try:
        milvus_client = MilvusClient("http://192.168.1.166:19530" )
        current_timestamp = datetime.now()
        text_embedding = textmodel.encode(str(st.session_state.description)) 
        sparse_embedding = splade_ef.encode_documents([str(st.session_state.description)])
        bgem3_embedding = bge_m3.encode_documents([str(st.session_state.description)])
   
        milvus_client.insert(COLLECTION, {"ghostclass": str(st.session_state.ghostclass),
                                          "filename": str(st.session_state.filename),
                                          "s3path": str(st.session_state.imgurl),
                                          "description": str(st.session_state.description), 
                                          "category": str(st.session_state.category),
                                          "identification":  str(st.session_state.identification),
                                          "location": str(st.session_state.location),
                                          "country": str( st.session_state.country),
                                          "latitude": str(st.session_state.latitude),
                                          "longitude": str(st.session_state.longitude),
                                          "zipcode":str(st.session_state.zipcode),
                                          "timestamp": str(current_timestamp),
                                          "s3timestamp": str(st.session_state.s3timestamp),
                                          "vector": st.session_state.image_embeddings[0], 
                                          "text_vector": sparse_embedding, 
                                          "text_vector2": text_embedding, 
                                          "text_vector3": bgem3_embedding["dense"][0]} )
    except Exception as e:
        print("save exception: ")
        print(e)
        st.sidebar.write(e)
    print(str(st.session_state.ghostclass) + " Ghost stored")

# -------- image
def fix_image(upload,filename):
    image = Image.open(upload)
    st.sidebar.write("Original Image :camera:")
    st.sidebar.image(image)
    images = [image]
    image_embeddings = model.encode(images)

    try:    
        with BytesIO() as upload:
            image.save(upload, format=image.format)
            upload.seek(0)
            s3_client.upload_fileobj(upload, BUCKET_NAME, filename)
    
        st.success(f"File '{filename}' uploaded to '{BUCKET_NAME}'")
        st.session_state.imgurl = str(str(S3_URL) + "/images/" + str(filename))
        st.session_state.s3timestamp = str(datetime.now())
        print("Complete: " + str(filename))
    except Exception as e:
        st.session_state.imgurl = ""
        st.error(f"Error uploading file '{filename}': {e}")
        print(e)

    #st.camera_input("Take a picture")
    st.session_state.image_embeddings = image_embeddings
    st.session_state.filename = filename
    st.session_state.category = st.selectbox("Category:", ["Fake", "Ghost", "Deity", "Unstable", "Environmental", "Vathek", "Legend", "Video Game"])
    st.session_state.ghostclass = st.selectbox("Ghost Class:", ["Fake", "Art", "TV", "Unclassified","Class I", "Class II", "Class III", "Class IV", "Class V", "Class VI", "Class VII"])
    st.session_state.identification = st.text_input("identification")
    st.session_state.location = st.text_input("location")
    st.session_state.country = st.text_input("country") 
    st.session_state.latitude = st.text_input("latitude")
    st.session_state.longitude = st.text_input("longitude")
    st.session_state.zipcode = st.text_input("zipcode")
    st.session_state.description = st.text_area("Description")
    st.button("Add Ghost", on_click=save)

# https://github.com/streamlit/llm-examples/blob/main/Chatbot.py
# https://github.com/omgtofo/locationinfo/blob/main/location.py
# https://github.com/aghasemi/streamlit_js_eval
# https://github.com/omgtofo/locationinfo/blob/main/location.py
# https://github.com/milvus-io/bootcamp/blob/master/bootcamp/tutorials/quickstart/apps/cir_with_milvus/ui.py
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 50MB.")
    else:
        fix_image(upload=my_upload, filename=my_upload.name)
