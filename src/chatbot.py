import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QMessageBox
from PyQt5.QtGui import QFont, QColor
##########
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
# key : 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 (kaustubh's a/c)
# new key: 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
openai.api_key = os.getenv("OPEN_API_KEY")
# openai.api_key = "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"
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
##########
class ChatBotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('IvantiGPT')
        self.setGeometry(100, 100, 500, 600)
        self.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF;")

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("background-color: #282828; color: #FFFFFF; border-radius: 10px; padding: 10px;")
        self.chat_history.setFont(QFont("Segoe UI", 14))  # Increase font size to 14

        self.input_box = QTextEdit()
        self.input_box.setMaximumHeight(100)
        self.input_box.setStyleSheet("background-color: #282828; color: #FFFFFF; border-radius: 10px; padding: 10px;")
        self.input_box.setFont(QFont("Segoe UI", 14))  # Increase font size to 14
        self.input_box.document().setDefaultStyleSheet("p {line-height: 1.5;}") 

        send_button = QPushButton('Send')
        send_button.setStyleSheet("background-color: #007ACC; color: #FFFFFF; border-radius: 10px; padding: 10px;")
        send_button.clicked.connect(self.send_message)

        vbox = QVBoxLayout()
        vbox.addWidget(self.chat_history)

        hbox = QHBoxLayout()
        hbox.addWidget(self.input_box)
        hbox.addWidget(send_button)

        vbox.addLayout(hbox)
        self.setLayout(vbox)

        self.chat_history.append('<b>IvantiGPT:</b> Hello! How can I help you?')

    def send_message(self):
        user_message = self.input_box.toPlainText().strip()
        if user_message == '':
            QMessageBox.warning(self, 'Warning', 'Please enter a message.')
            return

        self.chat_history.append('<b>User:</b> ' + user_message)
        # Add chatbot response here
        self.chat_history.append('<b>IvantiGPT:</b> ' + qa.run(user_message))

        self.input_box.clear()
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    chatbot_app = ChatBotApp()
    chatbot_app.show()
    sys.exit(app.exec_())
