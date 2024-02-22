# import os and openai for configuring api key
import os
import openai

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

'''
# Chatbot imports
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QColor
'''

# configure openai api key
from dotenv import load_dotenv
load_dotenv()
# key : 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
openai.api_key = os.getenv("OPEN_API_KEY")


# load model
with open('retriever.pkl', 'rb') as inp:
    big_chunks_retriever = pickle.load(inp)

qa = RetrievalQA.from_chain_type(
      llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
      chain_type="stuff",
      retriever=big_chunks_retriever)


'''
##############################----------CHATBOT_APP----------##############################
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
'''

# for terminal debugging
# print the result from user input until "end session" is entered
while(1):
  query = input("Enter your query:\n")
  if query == "end session":
    break
  print(qa.run(query))

print("Session Ended!")
