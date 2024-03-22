import os
import gradio as gr
import cmlapi
import pinecone
from typing import Any, Union, Optional
from pydantic import BaseModel
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import requests
import json
import time
from typing import Optional
import boto3
from botocore.config import Config
import chromadb
from chromadb.utils import embedding_functions

from huggingface_hub import hf_hub_download

# Set any of these to False, if not using respective parts of the lab
USE_PINECONE = False 
USE_CHROMA = True 

EMBEDDING_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"

if USE_PINECONE:
    print('Not Implemented')

if USE_CHROMA:
    # Connect to local Chroma data
    chroma_client = chromadb.PersistentClient(path="/home/cdsw/chroma-data")
    
    EMBEDDING_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
    EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

    COLLECTION_NAME = 'cml-default'

    print("initialising Chroma DB connection...")

    print(f"Getting '{COLLECTION_NAME}' as object...")
    try:
        chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
        print("Success")
        collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
    except:
        print("Creating new collection...")
        collection = chroma_client.create_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
        print("Success")

    # Get latest statistics from index
    current_collection_stats = collection.count()
    print('Total number of embeddings in Chroma DB index is ' + str(current_collection_stats))
    
    
## TO DO GET MODEL DEPLOYMENT
## Need to get the below prgramatically in the future iterations
client = cmlapi.default_client(url=os.getenv("CDSW_API_URL").replace("/api/v1", ""), cml_api_key=os.getenv("CDSW_APIV2_KEY"))
projects = client.list_projects(include_public_projects=True, search_filter=json.dumps({"name": "Shared LLM Model for Hands on Lab"}))
project = projects.projects[0]

## Here we assume that only one model has been deployed in the project, if this is not true this should be adjusted (this is reflected by the placeholder 0 in the array)
model = client.list_models(project_id=project.id)
selected_model = model.models[0]

## Save the access key for the model to the environment variable of this project
MODEL_ACCESS_KEY = selected_model.access_key

MODEL_ENDPOINT = os.getenv("CDSW_API_URL").replace("https://", "https://modelservice.").replace("/api/v1", "/model?accessKey=")
MODEL_ENDPOINT = MODEL_ENDPOINT + MODEL_ACCESS_KEY

#MODEL_ACCESS_KEY = os.environ["CML_MODEL_KEY"]
#MODEL_ENDPOINT = "https://modelservice.ml-8ac9c78c-674.se-sandb.a465-9q4k.cloudera.site/model?accessKey=" + MODEL_ACCESS_KEY


def main():
    # Configure gradio QA app 
    print("Configuring gradio app")

    DESC = "This AI-powered assistant showcases the flexibility of Cloudera Machine Learning to work with 3rd party solutions for LLMs and Vector Databases, as well as internally hosted models and vector DBs. The prototype does not yet implement chat history and session context - every prompt is treated as a brand new one."
    
    # Create the Gradio Interface
    demo = gr.ChatInterface(
        fn=get_responses, 
        #examples=[["What is Cloudera?", "AWS Bedrock Claude v2.1", 0.5, "100"], ["What is Apache Spark?", 0.5, "100"], ["What is CML HoL?", 0.5, "100"]], 
        title="Enterprise Custom Knowledge Base Chatbot",
        description = DESC,
        additional_inputs=[gr.Radio(['Local Mistral 7B', 'AWS Bedrock Claude v2.1'], label="Select Foundational Model", value="AWS Bedrock Claude v2.1"), 
                           gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.5, label="Select Temperature (Randomness of Response)"),
                           gr.Radio(["50", "100", "250", "500", "1000"], label="Select Number of Tokens (Length of Response)", value="250"),
                           gr.Radio(['None', 'Pinecone', 'Chroma'], label="Vector Database Choices", value="None")],
        retry_btn = None,
        undo_btn = None,
        clear_btn = None,
        autofocus = True
        )

    # Launch gradio app
    print("Launching gradio app")
    demo.launch(share=True,   
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_READONLY_PORT')))
    print("Gradio app ready")

# Helper function for generating responses for the QA app
def get_responses(message, history, model, temperature, token_count, vector_db):
    
    if model == "Local Mistral 7B":
        
        if vector_db == "None":
            context_chunk = ""
            response = get_model_response_with_context(message, context_chunk, temperature, token_count)
        
            # Stream output to UI
            for i in range(len(response)):
                time.sleep(0.02)
                yield response[:i+1]
                
        elif vector_db == "Pinecone":
            # Add reference to specific document in the response
            response = "Not Implemented"
            
            # Stream output to UI
            for i in range(len(response)):
                time.sleep(0.02)
                yield response[:i+1]
                
        elif vector_db == "Chroma":
            # Vector search in Chroma
            context_chunk, source = get_nearest_chunk_from_chroma_vectordb(collection, message)
            
            # Call CML hosted model
            response = get_model_response_with_context(message, context_chunk, temperature, token_count)
            
            # Add reference to specific document in the response
            response = f"{response}\n\n For additional info see: {url_from_source(source)}"
            
            # Stream output to UI
            for i in range(len(response)):
                time.sleep(0.02)
                yield response[:i+1]
    
    elif model == "AWS Bedrock Claude v2.1":
        if vector_db == "None":
            # No context call Bedrock
            response = "Not Implemented"
        
            # Stream output to UI
            for i in range(len(response)):
                time.sleep(0.02)
                yield response[:i+1]
                
        elif vector_db == "Pinecone":
            response = "Not Implemented"
            
            # Stream output to UI
            for i in range(len(response)):
                time.sleep(0.01)
                yield response[:i+1]
                
        elif vector_db == "Chroma":
            # Add reference to specific document in the response
            response = "Not Implemented"
            
            # Stream output to UI
            for i in range(len(response)):
                time.sleep(0.02)
                yield response[:i+1]

def url_from_source(source):
    url = source.replace('/home/cdsw/data/https:/', 'https://').replace('.txt', '.html')
    return f"[Reference 1]({url})"
    

# Return the Knowledge Base doc based on Knowledge Base ID (relative file path)
def load_context_chunk_from_data(id_path):
    with open(id_path, "r") as f: # Open file in read mode
        return f.read()


# Get embeddings for a user question and query Chroma vector DB for nearest knowledge base chunk
def get_nearest_chunk_from_chroma_vectordb(collection, question):
    ## Query Chroma vector DB 
    ## This query returns the two most similar results from a semantic search
    response = collection.query(
                    query_texts=[question],
                    n_results=1
                    # where={"metadata_field": "is_equal_to_this"}, # optional filter
                    # where_document={"$contains":"search_string"}  # optional filter
    )
    #print(results)
    
    return response['documents'][0][0], response['ids'][0][0]

    
# Pass through user input to LLM model with enhanced prompt and stop tokens
def get_model_response_with_context(question, context, temperature, token_count):
    
    mistral_sys = f"<s>[INST]You are a helpful, respectful and honest assistant. If you are unsure about an answer, truthfully say \"I don't know\"."
    
    if context == "":
        # Following LLama's spec for prompt engineering
        mistral_inst = f"Please answer the user question.[/INST]</s>"
        question_and_context = f"{mistral_sys} {mistral_inst} \n [INST] {question} [/INST]"
    else:
        # Add context to the question
        mistral_inst = f"Answer the user's question based on the following information:\n {context}[/INST]</s>"
        question_and_context = f"{mistral_sys} {mistral_inst} \n[INST] {question} [/INST]"
        
    try:
        # Build a request payload for CML hosted model
        data={ "request": {"prompt":question_and_context,"temperature":temperature,"max_new_tokens":token_count,"repetition_penalty":1.0} }
        
        r = requests.post(MODEL_ENDPOINT, data=json.dumps(data), headers={'Content-Type': 'application/json'})
        
        # Logging
        print(f"Request: {data}")
        print(f"Response: {r.json()}")
        
        no_inst_response = str(r.json()['response']['prediction']['response'])[len(question_and_context)-6:]
            
        return no_inst_response
        
    except Exception as e:
        print(e)
        return e


if __name__ == "__main__":
    main()
