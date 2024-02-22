# import os and openai for configuring api key
import os
import openai

# import Abstract Syntax Trees to parse it into a Python list
import ast

# import langchain essentials
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

#import pickle to save model
import pickle

# configure openai api key
from dotenv import load_dotenv
load_dotenv()
# key : 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
openai.api_key = os.getenv("OPEN_API_KEY")

# fetching urls from links.txt file
# Read the content of the file
with open('links.txt', 'r') as file:
    content = file.read()

# Parse the string representation of the list into a Python list
try:
    urls = ast.literal_eval(content)
except (SyntaxError, ValueError):
    # Handle error if the content of the file is not a valid Python literal
    print("Error: Content of the file is not a valid Python literal.")
    urls = []

# Print or use the elements
# print("List of elements:", elements)


# sample list
# urls = ['https://www.ivanti.com/customers', 'https://www.ivanti.com/customers/aggreko', 'https://www.ivanti.com/customers/american-university', 'https://www.ivanti.com/customers/bcd-travel', 'https://www.ivanti.com/customers/bcs-automotive', 'https://www.ivanti.com/customers/bilyoner', 'https://www.ivanti.com/customers/cape-peninsula-university-of-technology', 'https://www.ivanti.com/customers/city-of-brampton', 'https://www.ivanti.com/customers/city-of-seattle', 'https://www.ivanti.com/customers/conair']

# using async html loader to scrape through list of webpages
loader = AsyncHtmlLoader(urls)
docs = loader.load()

# transforming html to text
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

# create a text splitter to split documents into chunks for feeding the llm
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# create split version of the documents loaded
texts = text_splitter.split_documents(docs_transformed)

# create embeddings
embeddings = OpenAIEmbeddings()

# create a document searcher from the split texts and generate embeddings
docsearch = FAISS.from_documents(texts, embeddings)


# create a QnA chatbot
qa = RetrievalQA.from_chain_type(
      llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
      chain_type="stuff",
      retriever=docsearch.as_retriever())


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

save_object(docsearch.as_retriever(), 'retriever.pkl')
