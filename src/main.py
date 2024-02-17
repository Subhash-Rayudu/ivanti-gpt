from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

#import openi api key
import os
import openai
from dotenv import load_dotenv
load_dotenv()
# key : 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 (kaustubh's a/c)
openai.api_key = os.getenv("OPEN_API_KEY")

#create a web based loader to load data from ivanti wiki page
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Ivanti")

#generate documents from the wiki page
documents = loader.load()

#create a text splitter to split documents into chunks for feeding the llm
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

#create split version of the documents loaded
texts = text_splitter.split_documents(documents)

#create embeddings
embeddings = OpenAIEmbeddings()

#create a document searcher from the split texts and generate embeddings
docsearch = FAISS.from_documents(texts, embeddings)

#create a QnA chatbot
qa = RetrievalQA.from_chain_type(
      llm=OpenAI(model_name="gpt-3.5-turbo"),
      chain_type="stuff",
      retriever=docsearch.as_retriever())

#print the result from user input until "end session" is entered
while(1):
  query = input("Enter your query:\n")
  if query == "end session":
    break
  print(qa.run(query))

print("Session Ended!")
