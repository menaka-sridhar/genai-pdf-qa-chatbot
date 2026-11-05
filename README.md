## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:

Traditional document search methods—such as manual scanning or keyword searches—are inefficient for extracting specific insights from extensive, unstructured PDF documents (e.g., technical manuals, research papers, or legal reports). The challenge is to build an intelligent, context-aware conversational agent that can parse multi-page PDF documents, split text into manageable chunks, create vector embeddings, store them in a vector database, and retrieve precise answers to natural language queries using a Large Language Model (LLM) via LangChain.

### DESIGN STEPS:

#### STEP 1:
Use a library like PyPDFLoader to load the PDF document. Split the content into smaller text chunks using a TextSplitter (e.g., RecursiveCharacterTextSplitter) for efficient information retrieval.

#### STEP 2:
Convert the text chunks into numerical embeddings using OpenAIEmbeddings and store them in a vector database (e.g., Chroma) to enable similarity-based search.

#### STEP 3:
Use the vector database's retriever to fetch relevant text chunks based on user queries. Set the retriever parameters (e.g., search type) for accurate results.

#### STEP 4:
Use an OpenAI language model (e.g., GPT-4 or GPT-3.5) and a retrieval chain like RetrievalQA or ConversationalRetrievalChain to combine document context with conversational responses.

#### STEP 5:
Create a conversational interface (e.g., a command-line or GUI-based system) where users can input queries, and the chatbot responds with context-aware answers based on the PDF.

### PROGRAM:
```
%pip install langchain==0.2.17 langchain-community==0.2.19 langchain-openai==0.1.25 langchain-text-splitters==0.2.4 chromadb pypdf python-dotenv openai

import os
import datetime

from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv(find_dotenv())

openai_api_key = os.environ["OPENAI_API_KEY"]

import os

print("Current folder:", os.getcwd())
print("Available files:", os.listdir())

pdf_file_path = "Generative_AI_Study_Guide_10_Pages.pdf"

import os

print("PDF found:", os.path.isfile(pdf_file_path))

loader = PyPDFLoader(pdf_file_path)

documents = loader.load()

print("PDF loaded successfully!")
print("Total pages:", len(documents))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

docs = text_splitter.split_documents(documents)

embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="docs/chroma"
)

print("Vector database created successfully!")

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

persist_directory = "docs/chroma/"

embedding = OpenAIEmbeddings()

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

print("Total chunks:", vectordb._collection.count())

question = "What are the major topics in Generative AI?"

docs = vectordb.similarity_search(question, k=3)

print("Number of documents:", len(docs))

for doc in docs:
    print(doc.page_content)

from langchain_openai import ChatOpenAI

llm_name = "gpt-4o-mini"

llm = ChatOpenAI(
    model_name=llm_name,
    temperature=0
)

print("LLM initialized successfully!")

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever()
)

question = "What is Generative AI?"

result = qa_chain.invoke({"query": question})

print(result["result"])

from langchain.prompts import PromptTemplate

template = """
Use the following context to answer the question.
If you don't know the answer, say you don't know.
Use three sentences maximum.
Always say "thanks for asking!" at the end.

{context}

Question: {question}

Helpful Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

print("Prompt created successfully!")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)
#### question 1:
question = "What is Generative AI?"

result = qa_chain.invoke({"query": question})

print("Question:", question)
print("\nAnswer:", result["result"])

print("\nSource Document:")

if result["source_documents"]:
    print(result["source_documents"][0].page_content)
else:
    print("No source documents retrieved.")

print("Total chunks:", vectordb._collection.count())
#### question 2:
question = "What are the main applications of Generative AI?"

result = qa_chain.invoke({"query": question})

print("Question:", question)
print("\nAnswer:", result["result"])

print("\nSource Document:")
if result["source_documents"]:
    print(result["source_documents"][0].page_content)
else:
    print("No source documents retrieved.")
```

### OUTPUT:
### QUESTION 1:
<img width="623" height="568" alt="image" src="https://github.com/user-attachments/assets/7a29ec1c-4bf6-473b-928a-53d51c80a301" />

### QUESTION 2:
<img width="626" height="673" alt="image" src="https://github.com/user-attachments/assets/86f69e32-7251-4e23-bc58-969b34eb9082" />


### RESULT:
Thus, the PDF-Based Question-Answering Chatbot was successfully developed using LangChain, Retrieval-Augmented Generation (RAG), OpenAI embeddings, and Chroma vector database. The system successfully loads the PDF document, extracts and splits its content, stores the text embeddings, and retrieves relevant information based on user questions. The chatbot generates answers from the PDF content and displays the corresponding source documents. Therefore, the objective of developing an interactive PDF-based question-answering chatbot was achieved.
