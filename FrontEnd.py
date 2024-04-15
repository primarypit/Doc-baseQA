import sys

import PyQt5
from PyQt5.QtCore import QSize
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QFontDatabase, QFont
from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QPushButton,
                             QLineEdit, QTextBrowser, QFileDialog, QStackedWidget, QVBoxLayout, QHBoxLayout)
from markdown2 import markdown
from langchain import OpenAI
from DataPreprocess import *
from VecRetrieval import *
from LLMAPI import *



class NLPInterface(QWidget):
    def __init__(self):
        super().__init__()
        QFontDatabase.addApplicationFont('resources/OpenSans-Regular.ttf')
        self.setFont(QFont('Open Sans', 14))
        self.setWindowTitle("LangChain QA")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.usingFile = True
        self.selectedFilePath = ""
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # 回答框
        self.replyBox = QTextBrowser()
        self.replyBox.setFixedHeight(600)  # 设定高度
        layout.addWidget(self.replyBox)

        # 输入组件和按钮的容器
        inputLayout = QHBoxLayout()
        self.stackedWidget = QStackedWidget()
        self.videoInput = QLineEdit()
        self.videoInput.setPlaceholderText("Input video URL here...")
        self.videoInput.setFixedHeight(34)
        self.videoInput.setFixedWidth(700)
        self.uploadButton = QPushButton("Upload PDF File")
        self.uploadButton.setFixedHeight(30)
        self.uploadButton.setFixedWidth(700)
        self.stackedWidget.addWidget(self.uploadButton)
        self.stackedWidget.addWidget(self.videoInput)
        inputLayout.addWidget(self.stackedWidget)
        self.toggleInputButton = QPushButton()
        icon1 = QIcon('resources/change.png')
        self.toggleInputButton.setIcon(icon1)
        self.toggleInputButton.setIconSize(QSize(24, 24))
        inputLayout.addWidget(self.toggleInputButton)
        layout.addLayout(inputLayout)
        # Query输入框和发送按钮
        queryLayout = QHBoxLayout()
        self.queryInput = QLineEdit()
        self.queryInput.setPlaceholderText("Question here...")
        self.queryInput.setFixedHeight(60)
        self.queryInput.setFixedWidth(700)
        queryLayout.addWidget(self.queryInput)
        icon2 = QIcon('resources/send.png')
        self.sendButton = QPushButton()
        self.sendButton.setObjectName("roundButton")
        self.sendButton.setIcon(icon2)
        self.sendButton.setIconSize(QSize(60, 60))
        self.sendButton.setFixedWidth(150)
        self.sendButton.setFixedHeight(60)
        queryLayout.addWidget(self.sendButton)

        layout.addLayout(queryLayout)

        # 连接事件
        self.uploadButton.clicked.connect(self.uploadFile)
        self.toggleInputButton.clicked.connect(self.toggleInputMethod)
        self.sendButton.clicked.connect(self.sendToLLM)

        # 设置样式
        self.applyStyles()

    def applyStyles(self):
        self.setStyleSheet("""
            * {
                font-family: 'Open Sans';
                font-size: 14px;
            }
            QPushButton {
                font: bold 14px;
                border: 2px solid #8f8f91;
                border-radius: 6px;
                background-color: #f7f7f7;
                min-width: 80px;
                min-height: 30px;
            }
            QPushButton#roundButton {
                border: 0px;
                border-radius: 40px;
                min-width: 80px;
                min-height: 80px;
                max-width: 80px;
                max-height: 80px;
            }
            QPushButton:hover {
                background-color: #e4e4e5;
            }
            QPushButton:pressed {
                background-color: #d7d6d5;
            }
            QLineEdit {
                font: 14px;
                border: 2px solid #c7c7c7;
                border-radius: 6px;
                padding: 5px;
            }
            QTextBrowser {
                border: 2px solid #c7c7c7;
                border-radius: 6px;
                padding: 5px;
                background-color: #ffffff;
            }
        """)

    def uploadFile(self):
        fname, _ = QFileDialog.getOpenFileName(self, '上传文件', '/', "PDF文件 (*.pdf)")
        if fname:  # 检查用户是否选择了文件
            self.selectedFilePath = fname  # 保存完整路径以备后用
            filename = fname.split('/')[-1]  # 从路径中提取文件名
            self.uploadButton.setText(filename)  # 设置按钮文本为文件名
        else:
            self.uploadButton.setText("Upload PDF File")  # 如果没有选择文件，则重置按钮文本


    def toggleInputMethod(self):
        self.usingFile = not self.usingFile
        self.videoInput.setVisible(not self.usingFile)
        self.uploadButton.setVisible(self.usingFile)

    def sendToLLM(self):
        if self.usingFile:
            self.sendToLLMFile()
        else:
            self.sendToLLMVideo()
        self.queryInput.setText("")

    def sendToLLMVideo(self):
        open_key = "***"
        llmai = OpenAI(temperature=0, openai_api_key=open_key)

        saving_dir = "FAISS_store/"
        url = self.videoInput.text()
        video = Video(url)
        video.summary(llmai)
        video_name = video.get_name()
        summary = load_summary(video_name)
        search_type = "similarity"
        search_kwargs = {"k": 10}

        vecs = VecStore(search_type, search_kwargs)
        vecs.build_retriever(video, True)
        vecs.load_retriever(saving_dir, video_name)

        test_query = self.queryInput.text()
        self.display("Q: " + test_query + '\n')
        related_contents = vecs.query_search(test_query)
        related_contents = sorted(related_contents, key=lambda x: x.metadata["order"])

        related_text = "\n\n".join([r.page_content for r in related_contents])
        # print(related_text)

        llmapi = LLMapi(llmai)
        ans = llmapi.get_response(test_query, summary, related_text)
        self.display("A: "+ans+'\n')

    def sendToLLMFile(self):
        open_key = "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"
        llmai = OpenAI(temperature=0, openai_api_key=open_key)

        file_name = self.selectedFilePath.split("/")[-1]
        file_folder_dir = "/".join(self.selectedFilePath.split("/")[:-1]) + "/"
        saving_dir = "FAISS_store/"

        doc = Document(file_folder_dir, file_name)
        doc.summary(llmai)
        doc_name = doc.get_name()
        summary = load_summary(doc_name)

        search_type = "similarity"
        search_kwargs = {"k": 10}

        vecs = VecStore(search_type, search_kwargs)
        vecs.build_retriever(doc, True)
        vecs.load_retriever(saving_dir, doc_name)

        test_query = self.queryInput.text()
        self.display("Q: " + test_query + '\n')
        related_contents = vecs.query_search(test_query)
        related_contents = sorted(related_contents, key=lambda x: x.metadata["order"])

        related_text = "\n\n".join([r.page_content for r in related_contents])
        # print(related_text)

        llmapi = LLMapi(llmai)
        ans = llmapi.get_response(test_query, summary, related_text)
        self.display("A: " + ans + '\n')


    def display(self, reply):
        html = markdown(reply)
        currentContent = self.replyBox.toHtml()
        newContent = currentContent + "<br>" + html
        self.replyBox.setHtml(newContent)

def main():
    app = QApplication(sys.argv)
    ex = NLPInterface()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
