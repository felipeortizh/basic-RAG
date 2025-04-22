# basic-RAG

This script implements a retrieval‑augmented generation (RAG) pipeline using LangChain: it loads Wikipedia pages on a given topic with the WikipediaLoader document loader 
GitHub
, semantically chunks them with the experimental SemanticChunker backed by OpenAI embeddings 
Medium
, indexes those chunks in a FAISS vector store for fast similarity search 
GitHub
, and at query time retrieves relevant passages, stuffs them into a hub‑pulled RAG prompt template 
GitHub
, then generates answers via the ChatOpenAI wrapper calling GPT‑3.5‑Turbo 
Medium
.

🔑 Features
Web scraping & parsing via WikipediaLoader (2 articles) and bs4.SoupStrainer to limit HTML parse scope 
crummy.com
.

Semantic chunking with SemanticChunker(OpenAIEmbeddings()) to split texts at meaning shifts 
Medium
.

Vector indexing using FAISS.from_documents(...) for local, efficient storage and retrieval 
GitHub
.

RAG chain composition using RunnablePassthrough, a hub‑pulled prompt (jclemens24/rag-prompt), and StrOutputParser to condition GPT‑3.5‑Turbo answers on retrieved context 
GitHub
.

Answer generation through ChatOpenAI(model_name="gpt-3.5-turbo") for conversational-quality responses 
Medium
.

🛠️ Prerequisites
Python 3.8+ with venv support; create envs per the official guide 
Python documentation
.

Dependencies installed via pip install -r requirements.txt, including langchain, langchain-openai, langchain-experimental, langchain_community, chromadb or faiss-cpu 
GitHub
.

OpenAI API key stored in OPENAI_API_KEY and assigned to openai.api_key as recommended by the OpenAI Python library 
GitHub
.

🚀 Installation

# 1. Clone
git clone https://github.com/your‑org/rag_webqa.git
cd rag_webqa

# 2. Create & activate virtualenv
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install deps
pip install -r requirements.txt
▶️ Usage

# Ensure OPENAI_API_KEY is set:
export OPENAI_API_KEY="sk-…"

# Run the RAG script:
python rag_webqa.py
You’ll see console logs as the loader, chunker, and vector store initialize, then a final printed answer.

📂 Directory Layout

.
├── rag_webqa.py      # your RAG pipeline script  
├── requirements.txt  # Python deps  
└── README.md         # this documentation  
🔍 How It Works

Step	Component	Description
1. Load	WikipediaLoader	Fetches full articles for a query, limited by BeautifulSoup’s SoupStrainer filter 
crummy.com
.
2. Split	SemanticChunker(OpenAIEmbeddings())	Splits documents at semantic boundaries rather than fixed lengths 
Medium
.
3. Index	FAISS.from_documents(...)	Embeds chunks & stores vectors locally for similarity search 
GitHub
.
4. Retrieve	vectorstore.as_retriever()	Performs k‑nearest‐neighbors search over embeddings.
5. Prompt	hub.pull("jclemens24/rag-prompt")	Loads a community‑curated prompt template that “stuffs” context + question 
GitHub
.
6. Answer	ChatOpenAI(model_name="gpt‑3.5‑turbo")	Generates the final answer conditioned on provided context 
Medium
.
🔧 Customization
Change topic by editing the query argument in WikipediaLoader.

Persist FAISS DB to disk by passing persist_directory="db" to FAISS.from_documents().

Adjust retrieval with retriever = vectorstore.as_retriever(search_kwargs={"k": 4}).

Swap models to gpt-4o or gpt-4 in ChatOpenAI(...).

Prompt engineering: fork or modify the RAG prompt on LangChain Hub 
GitHub
.

Error handling: extend the try/except blocks around OpenAI calls per production best practices 
Medium
.

🌐 Environment Variables

Name	Purpose
OPENAI_API_KEY	Your OpenAI secret key for embeddings & chat calls.
⚠️ Limitations
Prototype quality: minimal error handling, no rate‑limit back‑off, and limited logging.

Single‑topic loader: only 2 Wikipedia pages by default; feel free to extend to multi‑page sites.

Costs: each embedding + chat call consumes OpenAI credits.

📜 License
Licensed under MIT. See LICENSE for details.

🙋 Author
Felipe Ortiz • @felipeortizh • fortiz.huerta@gmail.com
