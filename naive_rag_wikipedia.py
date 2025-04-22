#env: traints
import os, time
from langchain_community.document_loaders import WebBaseLoader, WikipediaLoader
import bs4
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import chromadb
#from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker


os.environ['OPENAI_API_KEY'] = 'sk-xxxxxxxxxxxx'
openai.api_key = os.environ['OPENAI_API_KEY']
# Set default User-Agent if not already set
if "USER_AGENT" not in os.environ:
    os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

try:
    # Verify OpenAI API key is set
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    print("Starting document loading...")
    #### INDEXING ####
    loader = WikipediaLoader(
        query="Artificial Intelligence",  # Your Wikipedia topic
        load_max_docs=2,  # Number of articles to load
        lang="en"  # Language of the articles
    )
    
    docs = loader.load()
    print(f"Documents loaded successfully: {len(docs)} documents")
    
    print("Creating text splits...")
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} text splits")
    
    print("Creating vector store...")

    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),)

    print("Vector store created successfully")

    #vectorstore.persist()
    #print("💾 Persisted to disk")
    

    retriever = vectorstore.as_retriever()
    print("Retriever created successfully")
    
    #### RETRIEVAL and GENERATION ####
    print("Loading prompt template...")
    prompt = hub.pull("jclemens24/rag-prompt")
    print("Prompt template loaded")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    print("Setting up language model...")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    print("Language model created successfully")
    
    rag_chain = (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
             | prompt
             | llm
             | StrOutputParser()
    )
    print("RAG chain created successfully")
    
    print("\nGenerating response...")
    result = rag_chain.invoke("What are the main areas of Artificial Intelligence?")
    print("\nResult:", result)

except openai.RateLimitError as e:
    print(f"OpenAI API rate limit exceeded: {str(e)}")
except openai.APIError as e:
    print(f"OpenAI API error: {str(e)}")
except Exception as e:
    print(f"Unexpected error: {str(e)}")
    print(f"Error type: {type(e)}")
