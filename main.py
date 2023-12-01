import os

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.globals import set_debug
from operator import itemgetter

print("Loading models...")
qa_llm_params = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MAX_NEW_TOKENS: 1000,
    GenParams.MIN_NEW_TOKENS: 5,
    GenParams.REPETITION_PENALTY: 1,
}
qa_llm = Model(
    model_id = ModelTypes.LLAMA_2_70B_CHAT, 
    params = qa_llm_params, 
    credentials = {
        "url": os.environ.get("WX_URL"),
        "apikey": os.environ.get("WX_APIKEY")
    },
    project_id = os.environ.get("WX_PROJECT_ID")).to_langchain()
genq_llm_params = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.MIN_NEW_TOKENS: 5,
    GenParams.REPETITION_PENALTY: 1,
    GenParams.STOP_SEQUENCES: ["?"]
}
genq_llm = Model(
    model_id = ModelTypes.LLAMA_2_70B_CHAT, 
    params = genq_llm_params, 
    credentials = {
        "url": os.environ.get("WX_URL"),
        "apikey": os.environ.get("WX_APIKEY")
    },
    project_id = os.environ.get("WX_PROJECT_ID")).to_langchain()

print("Loading documents...")
docs = PyPDFLoader("./Maintenance-Manual.pdf").load()

print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

print("Vectorizing documents...")
vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings())
retriever = vectorstore.as_retriever()

genq_prompt = PromptTemplate(
    input_variables=["chat_history", "question"], 
    template="""[/INST] Given the following chat history and latest user question \
        which might reference the chat history, formulate a standalone question \
        which can be understood without the chat history. Just reformulate the question if needed, otherwise return it as is.
        Chat history:
        {chat_history}
        Enf of chat history.
        Latest User question: {question} [/INST]
        Standalone Question: """
)
genq_chain = genq_prompt | genq_llm | StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
rag_prompt = PromptTemplate(
    input_variables=["context", "question"], 
    template="""[INST]Answer the question based only on the following context:
        {context}
        End of context.

        Question: {question}[/INST]
        """
)
rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question")
    }
    | rag_prompt
    | qa_llm
    | StrOutputParser()
)

def select_question(question, chat_history):
    if chat_history != "":
        return genq_chain.invoke({"chat_history": chat_history, "question": question})
    else:
        return question

#set_debug(True)
chat_history = ""
while True: # Exit by pressing Ctrl+C
    user_question = input("> ")
    question = select_question(user_question, chat_history)
    ai_msg = rag_chain.invoke({"question": question})
    chat_history += f"""
        Human: {user_question}
        Assistant: {ai_msg}
        """
    print(ai_msg)
