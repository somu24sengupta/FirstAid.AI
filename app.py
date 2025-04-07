
from langchain_groq import ChatGroq
from flask import Flask, render_template, jsonify, request
#from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
#from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()


pinecone_api_key= os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

from langchain.embeddings import HuggingFaceHubEmbeddings

def download_embeddings_hfhub():
    embeddings = HuggingFaceHubEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    return embeddings

embeddings=download_embeddings_hfhub()


index_name = "medichat"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":10})


# Initialize Groq LLM
llm = ChatGroq(
    model_name="llama3-70b-8192",  # or other models
    api_key=groq_api_key,
    temperature=0.4,
    max_tokens=500,
)

# Define the system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "The goal is to answer medical questions based on the provided context. "
    "If you don't know the answer, say 'I don't know'. "
    "Keep the answer concise and under 10 sentences.\n\n{context}"
)

# Create prompt template for user interaction
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)