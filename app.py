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
from langchain.agents import tool
from langchain.agents import AgentType, initialize_agent
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import PlaywrightURLLoader

def download_assets():
    session = boto3.session.Session()
    client = session.client('s3',
                        endpoint_url='https://blr1.digitaloceanspaces.com',
                        region_name='blr1', 
                        aws_access_key_id='DO00T4NGNZT7M9QH3TWY',
                        aws_secret_access_key=os.getenv('SPACES_SECRET')) 

    client.download_file('validin-knowledge-base', 'rag/main.pdf', 'assets/main.pdf')
    client.download_file('validin-knowledge-base', 'rag/docs.pdf', 'assets/docs.pdf')
    

def vectorize_assets():
    urls = [
        "https://app.validin.com/pricing",
    ]
    loaders = [
        PyPDFLoader("assets/main.pdf"),
        PyPDFLoader("assets/docs.pdf"),
        PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])
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

    # action = "Pricing Details"
    # if(action in message):
    #     tools = [getRestApiResponse()]
    #     agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
    #     result = agent.run(message)
    # else:
    #     result = rag_chain.invoke(message)
    result = rag_chain.invoke(message)
    return result

download_assets()

docs, vectorstore = vectorize_assets()

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

# @tool
# def getRestApiResponse():
#     return "This is response from one of the tools from within Validin ecosystem"

if __name__ == "__main__":
    demo.launch(show_api=False, debug=False, server_name="0.0.0.0")