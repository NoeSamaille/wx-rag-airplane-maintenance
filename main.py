import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.globals import set_debug

from langchain_ibm import WatsonxLLM
from operator import itemgetter

import logging
logging.basicConfig(level=logging.INFO)

logging.info("Loading models...")
qa_llm_params = {
    "decoding_method": "greedy",
    "max_new_tokens": 1000,
    "min_new_tokens": 5,
    "repetition_penalty": 1,
}
qa_llm = WatsonxLLM(
    model_id = "meta-llama/llama-3-8b-instruct", 
    params = qa_llm_params,
    project_id = os.environ.get("WATSONX_PROJECT_ID"))
genq_llm_params = {
    "decoding_method": "greedy",
    "max_new_tokens": 100,
    "min_new_tokens": 5,
    "repetition_penalty": 1,
    "stop_sequences": ["?"]
}
genq_llm = WatsonxLLM(
    model_id = "meta-llama/llama-3-8b-instruct", 
    params = genq_llm_params,
    project_id = os.environ.get("WATSONX_PROJECT_ID"))
 
logging.info("Loading documents...")
docs = PyPDFLoader("./Maintenance-Manual.pdf").load()

logging.info("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

logging.info("Vectorizing documents...")
vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings())
retriever = vectorstore.as_retriever()

genq_prompt = PromptTemplate(
    input_variables=["chat_history", "question"], 
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful AI assistant for airplane maintenance<|eot_id|><|begin_of_text|><|start_header_id|>user<|end_header_id|>

        Given the following chat history and latest user question \
        which might reference the chat history, formulate a standalone question \
        which can be understood without the chat history. Just reformulate the question if needed, otherwise return it as is.
        Chat history:
        {chat_history}
        Enf of chat history.
        Latest User question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        Standalone Question: """
)
genq_chain = genq_prompt | genq_llm | StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
rag_prompt = PromptTemplate(
    input_variables=["context", "question"], 
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful AI assistant for airplane maintenance<|eot_id|><|begin_of_text|><|start_header_id|>user<|end_header_id|>

        Answer the question based only on the following context:
        {context}
        End of context.
        
        Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
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
    logging.info(f"Question used for Q&A with RAG: {question}")
    ai_msg = rag_chain.invoke({"question": question})
    chat_history += f"""
        Human: {user_question}
        Assistant: {ai_msg}
        """
    print(ai_msg)
