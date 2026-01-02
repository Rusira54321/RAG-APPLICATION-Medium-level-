📄 LangChain RAG Chatbot with PDF & Chat History
```
This project demonstrates a Retrieval-Augmented Generation (RAG) chatbot built using LangChain, OpenAI, and ChromaDB.
The chatbot can:

Load a PDF document

Split it into chunks

Store embeddings in a vector database

Answer questions using retrieved context

Maintain chat history for better follow-up answers
```
🚀 Features
```
📄 PDF document loading using PyPDF

✂️ Recursive text splitting

🧠 OpenAI embeddings (text-embedding-3-small)

🗂️ Vector storage using Chroma

💬 Context-aware Q&A (RAG)

🕘 Chat history aware retriever

🔁 Question rewriting for follow-up queries
```
🛠️ Tech Stack
```
Python

LangChain

OpenAI

ChromaDB

Google Colab

PyPDF
```
📦 Installation
```
Install all required packages:

pip install langchain -U
pip install langchain-openai -U
pip install langchain-chroma -U
pip install langchain_community -U
pip install pypdf -U
```
🔑 Environment Setup
```
Set your OpenAI API key (Google Colab):

import os
from google.colab import userdata

os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_KEY")
```
📄 Load the PDF
```
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("/content/codeprolk.pdf")
docs = loader.load()
```
✂️ Split the Document
```
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

splits = text_splitter.split_documents(docs)
```
🧠 Create Embeddings & Vector Store
```
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_model
)

retriever = vectorstore.as_retriever()
```
🤖 Basic RAG Chain
```
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

system_prompt = """
You are an intelligent chatbot.
Use the following context to answer the question.
If you don't know the answer, say you don't know politely.

{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)
```
🕘 Chat History Aware RAG
```
This improves answers for follow-up questions.

Question Rewriting Prompt
from langchain_core.prompts import MessagesPlaceholder

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Reformulate the question if needed using chat history."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

History-Aware Retriever
history_aware_retriever = (
    rewrite_prompt
    | llm
    | (lambda x: x.content)
    | retriever
)
```
💬 Final RAG Chain with Memory
```
from langchain_core.runnables import RunnableLambda

final_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

rag_chain = (
    {
        "context": history_aware_retriever,
        "input": RunnablePassthrough(),
        "chat_history": RunnableLambda(lambda x: x["chat_history"])
    }
    | final_prompt
    | llm
)
```

🧪 Example Usage
```
from langchain_core.messages import HumanMessage, AIMessage

chat_history = []

question = "Can you describe it briefly?"

response = rag_chain.invoke({
    "input": question,
    "chat_history": chat_history
})

print(response.content)

chat_history.append(HumanMessage(content=question))
chat_history.append(AIMessage(content=response.content))
```

📌 Use Cases
```
Chat with PDF documents

Knowledge-based assistants

Company profile chatbots

Research document Q&A

Student project / internship portfolio
```

🙌 Author
```
Rusira DinuJaya
Software Engineering Intern | LangChain & AI Enthusiast
```
