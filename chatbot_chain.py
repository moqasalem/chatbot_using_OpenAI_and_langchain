import os
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
 
 
def get_chatbot_chain():
    #Replace sk-proj-... with your own key
    #note: in your OpenAI account you have to pay $5+ to run this chat , and each answer cost about $0.01

    os.environ["OPENAI_API_KEY"] = 'sk-proj-...'
 
    #step1 load your documentation
    loader = CSVLoader(file_path="cv_qa.csv")
    documents = loader.load()
 
    #setp 2 appy embedding
    vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
 
    #step 3 chat history
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
 
    #setep 4 call  conversation class
    chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(),
                                                  retriever=vectorstore.as_retriever(),
                                                  memory=memory)
    return chain