import os
import json
import requests
import networkx as nx
import joblib
import re
import time
from tqdm import tqdm

# --- CONFIGURATION ---
MERGE_DIR = "RAG_MODEL/mergejsons"
TRIPLES_FILE = "RAG_MODEL/extracted_triples.jsonl"
PROCESSED_LOG_FILE = "RAG_MODEL/processed_files.txt" 
GRAPH_FILE = "RAG_MODEL/knowledge_graph.joblib"
GRAPHML_FILE = "RAG_MODEL/knowledge_graph.graphml"

# LLM Config
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma:2b"

def clean_llm_response(response_text):
    """Cleans Markdown code blocks from LLM response."""
    cleaned = re.sub(r"```json", "", response_text)
    cleaned = re.sub(r"```", "", cleaned)
    return cleaned.strip()

def extract_triples(text, retries=3):
    """
    Extracts triples with retries and a hardened prompt.
    """
    if not text or len(text) < 20: return []

    prompt = f"""
    Task: Analyze the text and extract knowledge triples.
    Format: JSON list of objects with keys 'head', 'relation', 'tail'.
    
    Example Input: "Apple Inc. was founded by Steve Jobs in California."
    Example Output: [
        {{"head": "Apple Inc.", "relation": "founded_by", "tail": "Steve Jobs"}},
        {{"head": "Apple Inc.", "relation": "founded_in", "tail": "California"}}
    ]

    Rules:
    1. Output ONLY the JSON array.
    2. Use concise entities.
    3. If no facts found, output [].

    Text to Analyze: {text}
    """

    for attempt in range(retries):
        try:
            r = requests.post(OLLAMA_URL, json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1} 
            }, timeout=120)

            if r.status_code != 200:
                print(f" [!] API Error {r.status_code}. Retrying...")
                time.sleep(2)
                continue

            response_json = r.json()
            cleaned_text = clean_llm_response(response_json.get('response', ''))
            
            data = json.loads(cleaned_text)
            if isinstance(data, list):
                return data
            else:
                return [data]

        except json.JSONDecodeError:
            print(f" [x] JSON Error on attempt {attempt+1}. Retrying...")
        except requests.exceptions.RequestException as e:
            print(f" [x] Connection Error: {e}")
        
        time.sleep(1)

    return []

def get_processed_files():
    """Reads the log of files we have completely finished."""
    if not os.path.exists(PROCESSED_LOG_FILE):
        return set()
    with open(PROCESSED_LOG_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def mark_file_as_processed(filename):
    """Updates the log file."""
    with open(PROCESSED_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(filename + "\n")

def process_files_and_save():
    os.makedirs(os.path.dirname(TRIPLES_FILE), exist_ok=True)

    all_files = [f for f in os.listdir(MERGE_DIR) if f.endswith(".json")]
    completed_files = get_processed_files()
    
    files_to_process = [f for f in all_files if f not in completed_files]
    
    print(f"Found {len(all_files)} files. {len(completed_files)} already done.")
    print(f"Remaining: {len(files_to_process)}")

    if not files_to_process:
        return

    with open(TRIPLES_FILE, "a", encoding="utf-8") as out_f:
        for filename in tqdm(files_to_process, desc="Processing"):
            filepath = os.path.join(MERGE_DIR, filename)
            
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            chunks = data.get("chunks", [])
            
            for chunk in chunks:
                chunk_text = chunk.get("text", "")
                
                triples = extract_triples(chunk_text)
                
                if triples:
                    for triple in triples:
                        if "head" in triple and "tail" in triple:
                            triple["source_file"] = filename
                            triple["chunk_id"] = chunk.get("number", 0)
                            
                            out_f.write(json.dumps(triple) + "\n")
                    out_f.flush()
            
            mark_file_as_processed(filename)

def build_graph():
    if not os.path.exists(TRIPLES_FILE):
        print("No triples file found.")
        return

    print("Building NetworkX Graph...")
    
    # Use MultiDiGraph to allow directed edges and multiple relations between same entities
    G = nx.MultiDiGraph() 

    with open(TRIPLES_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                
                # --- FIX START: Sanitize inputs for None values ---
                # Force entities to string, default to "Unknown" if None
                head = str(data.get('head') if data.get('head') is not None else "Unknown")
                tail = str(data.get('tail') if data.get('tail') is not None else "Unknown")
                
                # Handle relation specifically: if it's None, use default
                raw_relation = data.get('relation')
                if raw_relation is None:
                    relation = "related_to"
                else:
                    relation = str(raw_relation)

                # Clean Metadata: Convert None values to empty strings
                # GraphML crashes if value is None
                meta = {k: (str(v) if v is not None else "") 
                        for k, v in data.items() 
                        if k not in ['head', 'tail', 'relation']}
                # --- FIX END ---
                
                G.add_node(head, label=head)
                G.add_node(tail, label=tail)
                G.add_edge(head, tail, label=relation, relation=relation, **meta)
            except Exception as e:
                continue

    print(f"Stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    # Save Joblib (Python use)
    joblib.dump(G, GRAPH_FILE)
    print(f"Saved binary graph to {GRAPH_FILE}")
    
    # Save GraphML (Visualization use - Gephi/Cytoscape)
    try:
        nx.write_graphml(G, GRAPHML_FILE)
        print(f"Saved GraphML to {GRAPHML_FILE}")
    except Exception as e:
        print(f"Could not save GraphML (likely encoding issue): {e}")

if __name__ == "__main__":
    process_files_and_save()
    build_graph()