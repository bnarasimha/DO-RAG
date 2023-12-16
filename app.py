import os
import boto3
from openai import OpenAI
import gradio as gr
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions

def download_assets():
    session = boto3.session.Session()
    client = session.client('s3',
                        endpoint_url='https://blr1.digitaloceanspaces.com', # Find your endpoint in the control panel, under Settings. Prepend "https://".
                        region_name='blr1', # Use the region in your endpoint.
                        aws_access_key_id='DO00T4NGNZT7M9QH3TWY', # Access key pair. You can create access key pairs using the control panel or API.
                        aws_secret_access_key=os.getenv('SPACES_SECRET')) # Secret access key defined through an environment variable.

    client.download_file('validin-knowledge-base', 'rag/main.pdf', 'assets/main.pdf')

download_assets()

def vectorize_assets():
    loaders = [
        PyPDFLoader("assets/main.pdf")
    ]

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 150
    )
    splits = text_splitter.split_documents(docs)

    client = weaviate.Client(
        embedded_options = EmbeddedOptions()
    )

    vectorstore = Weaviate.from_documents(
        client = client,    
        documents = splits,
        embedding = OpenAIEmbeddings(),
        by_text = False
    )
    return docs,vectorstore

docs, vectorstore = vectorize_assets()

def getAnswer(message, history):
    
    question = message
    context = docs

    template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    retriever = vectorstore.as_retriever()

    rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()} 
        | prompt 
        | llm
        | StrOutputParser() 
    )

    result = rag_chain.invoke(message)

    return result

demo = gr.ChatInterface(
    getAnswer,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me anything about Validin", container=False, scale=7),
    title="Ask Validin",
    description="Ask Validin any question",
    theme="soft",
    examples=["What are your services", "How do you mitigate cyber risk?"],
    cache_examples=True,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear")

if __name__ == "__main__":
    demo.launch(show_api=False, debug=False, server_name="0.0.0.0")