import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests
from create_embeds import create_embedding
import networkx as nx
import re
import json  


EMBEDDINGS_FILE = 'RAG_MODEL/embeddings.joblib'
GRAPH_FILE = 'RAG_MODEL/knowledge_graph.joblib'
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral:7b" 

def sec_to_mmss(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:02d}"

def inference(prompt):
    """
    Streams the response word-by-word so it feels instant.
    """
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": True  # <--- MUST BE TRUE
        }, stream=True)
        r.raise_for_status()

        full_response = ""
        print("Response: ", end="", flush=True)

        # Iterate over the stream line by line
        for line in r.iter_lines():
            if line:
                try:
                    body = json.loads(line)
                    token = body.get("response", "")
                    full_response += token
                    # Print immediately to console
                    print(token, end="", flush=True)
                except json.JSONDecodeError:
                    continue

        print() # Newline at the end
        return {"response": full_response}

    except Exception as e:
        print(f"\nError calling LLM: {e}")
        return {"response": "Error generating response."}

def get_smart_graph_context(G, query):
    """
    Scans the query for entities present in the graph using n-grams.
    Handles case-insensitivity and multi-word phrases.
    """
    if not G:
        return ""

    # 1. Map lowercase names to actual node names
    node_map = {str(n).lower(): n for n in G.nodes()}
    
    # 2. Break query into words
    words = re.findall(r'\w+', query.lower())
    found_nodes = set()
    
    # 3. Check for 3-word, 2-word, and 1-word matches
    for n in range(3, 0, -1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i+n])
            if phrase in node_map:
                actual_node = node_map[phrase]
                found_nodes.add(actual_node)

    # 4. Get 1-hop neighbors
    paths = []
    for node in found_nodes:
        try:
            neighbors = G[node]
            for neighbor, attrs in neighbors.items():
                if isinstance(attrs, dict) and 0 in attrs: 
                    relation = attrs[0].get('relation', 'related_to')
                else:
                    relation = attrs.get('relation', 'related_to')
                paths.append(f"{node} --[{relation}]--> {neighbor}")
        except Exception:
            continue

    if not paths:
        return ""
    
    return "Knowledge Graph Insights:\n" + "\n".join(paths[:15])


if __name__ == "__main__":
    print("Loading Embeddings and Graph...")
    df = joblib.load(EMBEDDINGS_FILE)
    G = joblib.load(GRAPH_FILE)
    print("System Ready! (Type 'q' to exit)")

    # Added loop so you don't have to restart script every time
    while True:
        incoming_query = input("\nAsk a Question: ")
        if incoming_query.lower() == 'q':
            break
        
    
        print(">> Searching video transcripts...")
        question_embedding = create_embedding([incoming_query])[0] 
        
        similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
        
        top_results = 5
        max_indx = similarities.argsort()[::-1][0:top_results]
        
      
        new_df = df.iloc[max_indx].copy() 

        new_df["start"] = new_df["start"].apply(sec_to_mmss)
        new_df["end"] = new_df["end"].apply(sec_to_mmss)

        graph_context = get_smart_graph_context(G, incoming_query)

        # Prompt
        prompt = f'''
        I am teaching web development in my Sigma web development course.

        CONTEXT FROM VIDEOS:
        {new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}

        CONTEXT FROM KNOWLEDGE GRAPH:
        {graph_context}

        ---------------------------------

        USER QUESTION:
        "{incoming_query}"

        INSTRUCTIONS:
        - Answer in a natural, human teaching style.
        - Clearly mention which video number and title contains the answer.
        - Mention the exact timestamp range in mm:ss format.
        - Briefly explain what is taught in that part.
        - Guide the user to watch that specific portion of the video.
        - Do NOT mention JSON, chunks, or the above format.
        - Use the Knowledge Graph context to verify facts or mention related concepts.
        - If the question is unrelated, say you can only answer questions related to the course.
        '''

   
        with open("RAG_MODEL/prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

        print(">> Generating Answer...")
        
        
        response_data = inference(prompt)
        response_text = response_data.get("response", "")

        
        with open("RAG_MODEL/response.txt", "w", encoding="utf-8") as f:
            f.write(response_text)