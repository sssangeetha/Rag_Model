import torch
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp


# use inbuilt cuda gpu offered by torch ML else we can use our own inbuilt cpu for processing 
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = PyPDFLoader("/Users/srisaisangeethajannapureddy/Desktop/Chat-Bot/Sangeetha-Java_developer_Resume.pdf")
data= loader.load()
text_splitter= RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap= 200)
chunks_of_text= text_splitter.split_documents(data)

#creating the model

llm_answer_gen = LlamaCpp( streaming= True, model_path= r"./mistral-7b-openorca.Q4_0.gguf", temperature=0, top_p=1, f16_kv=True, verbose= False, n_ctx= 6000, max_tokens=512)

# Created embeddings using huggingfaceembeddings for vector creation

embeddings= HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2", model_kwargs= {"device": device})

# Created the vector store for storing the vectors

vector_store= Chroma.from_documents(documents= chunks_of_text, embedding=embeddings)

#initializing the chain for generating answers

memory= ConversationBufferMemory(memory_key= "chat_history", return_messages= True)

# Creating the langchain

answer_gen_chain= ConversationalRetrievalChain.from_llm(llm=llm_answer_gen, retriever= vector_store.as_retriever(), memory=memory)


while True:
    input= input("Enter a question: ")
    if input.lower()== 'q':
        break

    answers = answer_gen_chain.invoke({"question": input, "chat_history":[]})


    print(" Answer is :", answers)












