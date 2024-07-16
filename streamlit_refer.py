import streamlit as st
import tiktoken # 토큰 개수 세기 위한 라이브러리
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# PDF, DOCs 모두에서 데이터 읽기 위해 로더들 import
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# 몇 개까지의 대화를 메모리에 넣어줄 것인지(그래야 챗봇이 대화를 기억해서) 결정하기 위한 라이브러리
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS # 임시로 벡터를 저장하는 라이브러리

# 메모리를 구현하기 위한 추가적인 라이브러리들
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    # 페이지 상세 설정
    st.set_page_config(
        page_title="DirChat",
        # streamlit에서 사전 정의된 아이콘은 이런식으로 사용 가능
        page_icon=":books:")
    # 페이지 제목
    st.title("_Private Data :red[QA Chat]_ :books:")

    # streamlit의 session에 converstation이 없으면 이걸 None으로 설정해라
    # 이후에 session_state conversation값을 쓰려면 사전에 정의가 필요해서 정의하는 거다.
    if "converstaion" not in st.session_state:
        st.session_state.conversation = None

    # streamlit의 session에 chat_history가 없으면 이걸 None으로 설정해라(과거 채팅 이력)
    # 이후에 session_state chat_history값을 쓰려면 사전에 정의가 필요해서 정의하는 거다.
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    # 사이드바 안에 필요한 구성품들을 정의해준다.
    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf','docx'],accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

    # 만약 버튼을 누르면
    if process:
        if not openai_api_key: # 만약 API키가 없으면
            st.info("Please add your OpenAI API key to continue")
            st.stop()
        files_text = get_text(uploaded_files) # 파일에서 텍스트 추출
        text_chunks = get_text_chunks(files_text) # 텍스트틀 여러 덩어리로 나누기
        vectorstore = get_vectorstore(text_chunks) # 추출된 텍스트 덩어리들 벡터화

        # vectorstore를 갖고 LLM이 답변을 생성할 수 있게 체인 생성 
        # session_state에다가 conversation이라는 변수로 데이터를 저장.
        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)

    # 대화 UI를 구성하기 위한 코드
    if 'message' not in st.session_state:
        # st.session_state.messages나 st.session_state['messages'] 똑같음.
        # messages를 초기화
        st.session_state['messages'] = [{"role":"assistance",
                                        "content":"아녕하세요! 주어진 문서에서 궁금한게 있으면 물어보세요!"}]
    # 각각의 메시지들을 with 구문으로 묶는데
    for message in st.session_state.messages:
        # 각 role에 따라 메시지의 content를 st.chat_message와 st.markdown 통해 채팅형식으로 표시
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    query = st.chat_input("질문을 입력해주세요")
    # ':='는 사용자가 입력한 텍스트를 query에 반환
    if query: # 만약 질문을 입력하면
        st.session_state.messages.append({"role":"user", "content":query})

        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistance"):
            chain = st.session_state.conversation
            with st.spinner("Thinking..."):
                result = chain({"question":query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
            st.session_state.messages.append({"role":"assistance","content":response})

# 토큰개수를 기준으로 텍스트를 나눠주는 함수
def titoken_len(text):
    tokenizer = tiktoken.get_encoding("ck100k_base") # ck100k_base라는 tokenizer활용
    tokens = tokenizer.encode(text)
    return len(tokens)

# 업로드한 파일을 모두 text로 변환하는 함수
def get_text(docs):
    doc_list = []

    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"uploaded {file_name}")
        
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extends(documents)
    
    return doc_list

# 여러 개의 덩어리로 텍스트를 쪼갬
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain



if __name__ == '__main__':
    main()
