from langchain.embeddings.base import Embeddings
from modelscope import AutoTokenizer, AutoModel
import torch
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = ''

DEEPSEEK_API_KEY = ''
DEEPSEEK_BASE_URL = 'https://api.deepseek.com'

# 定义 BGE 嵌入模型
class BGEM3Embeddings(Embeddings):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __embed_text(self, text: str) -> list[float]:
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.__embed_text(text) for text in texts]
    
    def embed_query(self, text: str) -> list[float]:
        return self.__embed_text(text)
    
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

model_name = "BAAI/bge-m3" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
bgem3embedding = BGEM3Embeddings(model, tokenizer)

loader = WebBaseLoader(
    web_paths=("http://www.dean.pku.edu.cn/web/rules_info.php?id=8",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("news_info")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=60)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=bgem3embedding)

retriever = vectorstore.as_retriever()

prompt_template = """
你是一个北京大学的教务人员，现在你需要根据以下内容回答问题：

内容：
{context}
问题：
{question}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

llm = ChatOpenAI(model="deepseek-chat",
                 base_url=DEEPSEEK_BASE_URL,
                 api_key=DEEPSEEK_API_KEY)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("如何做好做好试题保密工作？"))
