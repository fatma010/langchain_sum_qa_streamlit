
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings # provides embeddings for text processing
import tempfile
import time
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter #splits text into smaller chunks
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

# load environment variables
load_dotenv()

# check open_ai_api_key in env, if it is not defined as a variable
#it can be added manually in the code below

if os.environ.get("OPEN_AI_API_KEY") is None or os.environ.get("OPEN_AI_API_KEY") =="":
    print("open_ai_api_key is not set as environment variable")
else:
    print("Open AI API Key is set")

#get open_ai_api_key
OPEN_AI_API_KEY= os.environ.get("OPEN_AI_API_KEY")

# define llm
llm = OpenAI(openai_api_key='OPEN_AI_API_KEY', temperature=0)

st.title("PDF Summarizer & Question Answering")
st.info("Upload a pdf file, get a summary and start asking questions")

# split text not to increase token size
def text_splitter(text):
    # split the text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def pdf_summarizer(pdf_file):
    summary = []
    if pdf_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(pdf_file.read())
            pdf_file_path = tmp_file.name
            loader = PyPDFLoader(pdf_file_path)
            pages = loader.load_and_split()
            pdf_content = ''.join([p.page_content for p in pages])
            texts = text_splitter(pdf_content)
            docs = [Document(page_content = t ) for t in texts]
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary = chain.run(docs)

            os.remove(pdf_file_path)

    return summary




def pdf_questions(pdf_file,question):
    if pdf_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(pdf_file.read())
                pdf_file_path = tmp_file.name
                loader = PyPDFLoader(pdf_file_path)
                pages = loader.load_and_split()
                pdf_content = ''.join([p.page_content for p in pages])
                texts = text_splitter(pdf_content)
                embedding = OpenAIEmbeddings(openai_api_key = 'OPEN_AI_API_KEY')
                document_search = FAISS.from_texts(texts, embedding)
                if question is not None:
                    chain = load_qa_chain(llm, chain_type="stuff")
                    docs = document_search.similarity_search(question)
                    answer = chain.run(input_documents=docs, question=question)

                os.remove(pdf_file_path)

    return answer



pdf_file = st.file_uploader("Uplaod a PDF file", type="pdf" )

st.subheader("What do you want to do with your document?")
#st.divider()
Options = st.radio("", ["Get Summary", "Ask a Question"])

if Options == "Get Summary":
    if pdf_file is not None:
        st.subheader("Summary")
        st.write(pdf_summarizer(pdf_file))

if Options == "Ask a Question":
    if pdf_file is not None:
        question = st.text_area("")
        if st.button("Generate Response"):
            st.write(pdf_questions(pdf_file,question))















