🧠 Local GraphRAG: Knowledge Graph Retrieval Engine
A privacy-first RAG system that turns unstructured text into a queryable Knowledge Graph using Local LLMs.

📖 Overview
This project implements GraphRAG (Graph Retrieval-Augmented Generation) entirely locally. Unlike traditional RAG which relies solely on vector similarity, this system extracts structured knowledge (Entities & Relationships) from your documents to build a directed Knowledge Graph.

It allows for Multi-Hop Reasoning—answering complex questions by traversing the connections between concepts (e.g., "How does X relate to Y through Z?")—which standard RAG often fails to do.

🚀 Key Features
Automated Knowledge Extraction: Uses local LLMs (Gemma:2b) to read text and extract "Triples" (Head -> Relation -> Tail) at scale.

Graph-Based Retrieval: Stores data in a structured network format (NetworkX), allowing for precise, deterministic retrieval of relationships.

Context-Aware Chat: A chat interface that combines graph traversal with LLM reasoning to provide grounded, hallucination-resistant answers.

Visualization Ready: Automatically exports your entire knowledge base to GraphML for analysis in tools like Gephi or Cytoscape.

Parallel Processing: optimized multi-threaded pipeline for fast ingestion on Apple Silicon and Consumer GPUs.

🛠️ Tech Stack
LLM Backend: Ollama (Gemma:2b for fast extraction, Mistral:7b for chat).

Graph Logic: NetworkX for graph data structure and pathfinding.

Data Processing: Pandas & Joblib for efficient data handling and serialization.

Storage: Local binary graph storage (no external graph DB required).

📂 Project Structure
Bash
├── build_graph.py       # Main Engine: Extracts triples & builds the graph
├── processing.py        # Chat Interface: Queries the graph to answer questions
├── repair_graph.py      # Utility: Cleans graph data for visualization tools
├── RAG_MODEL/           # Storage folder for the binary graph and logs
│   ├── knowledge_graph.joblib   # The "Brain" (Binary Graph)
│   ├── knowledge_graph.graphml  # Visualization file
│   └── extracted_triples.jsonl  # Raw extracted facts
└── requirements.txt     # Python dependencies
⚙️ Installation
1. Prerequisites

Python 3.10+

Ollama installed and running.

2. Clone & Install Dependencies

Bash
git clone https://github.com/yourusername/local-graphrag.git
cd local-graphrag
pip install -r requirements.txt
3. Pull Local Models

Bash
ollama pull gemma:2b   # Optimized for fast Triple Extraction
ollama pull mistral:7b # Optimized for Chat/Reasoning
⚡ Usage Guide
Step 1: Ingest Data & Build Graph

Place your JSON/Text files in the mergejsons folder and run the builder.

Bash
python build_graph.py
What happens: The system reads your files, uses Gemma to identify entities and relationships, and constructs a network. It supports stopping and resuming automatically.

Step 2: Chat with your Data

Start the interactive query engine.

Bash
python processing.py
Example Query: "How is 'React' related to 'Virtual DOM'?"

The Logic: The system finds the 'React' node, traverses its edges to find neighbors, and uses Mistral to summarize the relationship.

Step 3: Visualize

Open RAG_MODEL/knowledge_graph.graphml in Gephi to see a visual map of your data.


Created by Harshit Singh
