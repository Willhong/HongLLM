from langchain_openai import ChatOpenAI
from langchain.document_loaders.web_base import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import BSHTMLLoader
# 네이버 뉴스기사 주소
url = 'https://n.news.naver.com/article/437/0000361628?cds=news_media_pc'

# 웹 문서 크롤링
