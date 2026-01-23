import requests
import os
import json
import numpy as np
import pandas as pd
import joblib

def create_embedding(text_list):
    
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding

#inside function to reuse create embedding for question embeddings
def build_embedding():
    jsons = os.listdir("RAG_MODEL/mergejsons")  # List all the jsons 
    my_dicts = []
    chunk_id = 0

    for json_file in jsons:
        filepath=os.path.join("RAG_MODEL/mergejsons",json_file)
        with open(filepath,"r",encoding="utf-8") as f:
            content = json.load(f)
        print(f"Creating Embeddings for {json_file}")
        embeddings = create_embedding([c['text'] for c in content['chunks']])
           
        for i, chunk in enumerate(content['chunks']):
            chunk['chunk_id'] = chunk_id
            chunk['embedding'] = embeddings[i]
            chunk_id += 1
            my_dicts.append(chunk) 
    # print(my_dicts)
    
    df = pd.DataFrame.from_records(my_dicts)
    # Save this dataframe
    joblib.dump(df, 'RAG_MODEL/embeddings.joblib')
    
if __name__ == "__main__":
    build_embedding()
    print("Embeddings Created Succesfully !")