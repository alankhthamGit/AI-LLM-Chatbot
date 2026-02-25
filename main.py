# Souce code written by Alan Tham
#
# To run type "streamlit run main.py"
# Make sure LM studio is running with meta-llama-3-8b-instruct loaded and Server is in running status.
# Make sure the following files are in the same directory as main.py
# 1) config.py
# 2) .env
# 3) human_icon.png 
# 4) engagepro_favicon.png 
# 5) engagepro_logo.png
# 6) Company_Brochure.pdf

import streamlit as st
from config import llm_local as llm
from typing import List
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import hf_embeddings
import numpy as np
import pandas as pd

# Session state setup
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state["vector_db"] = None  # Ensure the key exists

if "agent" not in st.session_state:
    st.session_state.agent = None

st.set_page_config(page_title="EngagePro Chatbot", page_icon="engagepro_favicon.png")
st.image("engagepro_logo.png")
st.title("Welcome to EngagePro.")

class TopicDetectionResult(BaseModel):
    topics: List[str]
    status: str

def check_valid_question_tool(llm, question: str):
    forbidden_topics = ["Religion"]
    all_possible_topics = ["Politics", "Technology", "Finance", "Religion", "Healthcare", "Others"]
    
    parser = JsonOutputParser(pydantic_object=TopicDetectionResult)
    
    prompt_text = """
        System: You are a professional classifier. Return ONLY valid JSON. 
        Do not include any introductory text or explanations.

        Check if the following user question is related to religion: "{question}"

        Instructions:
        - If the question mentions God, god, faith, religion, worship, or spirituality, set "status" to "forbidden"
        - Otherwise, set "status" to "valid"
        - List the "topics" found.

        JSON structure:
        {{"topics": ["..."], "status": "..."}}
        """


    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | parser
    
    result = chain.invoke({
        "question": question, 
        "topic_list": ", ".join(all_possible_topics),
        "forbidden": forbidden_topics
    })

    return result["status"]

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    
    # Precision-focused splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # Smaller size for better detail matching
        chunk_overlap=50,      # 10% overlap to maintain context
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""] # Priority splitting
    )
    
    pages = loader.load_and_split(text_splitter)
    documents_data = []
    NAIS_PICKLE = "nais.pkl"

    for idx, doc in enumerate(pages):
        print(f"Embedding {idx}")
        try:
            doc_embedding = hf_embeddings.encode(doc.page_content)
            print(doc.page_content)
            documents_data.append({
                "id": idx,
                "content": doc.page_content,
                "source": doc.metadata,
                "embedding": doc_embedding
            })
        except Exception as e:
            print(f"Error processing document {idx}: {e}")
            pd.DataFrame(documents_data).to_pickle("tmp/partial_documents_dataframe.pkl")

    df = pd.DataFrame(documents_data)

    df.to_pickle(NAIS_PICKLE)
    df = pd.read_pickle(NAIS_PICKLE)
    documents = df['content'].tolist()
    document_embeddings = np.array(df['embedding'].tolist())

    return document_embeddings, documents


def build_vectordb(chunks):
    return Chroma.from_texts(chunks, embedding_model, collection_metadata={"hnsw:space": "cosine"})

@st.cache_resource
def get_vector_db():
    """Loads PDF once and keeps it in memory for high-speed, accurate search."""
    document_embeddings, documents = extract_text_from_pdf("Company_Brochure.pdf")
    return document_embeddings, documents

def find_top_k_similar(embedding, embeddings, num_results=5):
    """
    Finds the top k most similar documents to the given embedding.
    
    :param embedding: The query embedding vector.
    :param embeddings: Numpy array of document embeddings.
    :param num_results: The number of top results to return (default is 5).
    :return: Indices of the top k similar documents and their similarities.
    """
    similarities = hf_embeddings.similarity(embedding, embeddings)[0]
    top_k_indices = np.argsort(similarities)[-num_results:]  # Get indices of top k results
    return [(index, similarities[index]) for index in reversed(top_k_indices)]  # Return in descending order of similarity

@tool
def pdf_search(question: str):
    """Search the company PDF brochure for key achievements, products, and services.
    If you cannot find the answer, say 'Results not found.'""" 
    vector_db, documents = get_vector_db()

    query_embedding = hf_embeddings.encode(question)
    top_results = find_top_k_similar(query_embedding, vector_db, num_results=5)

    valid_chunks = []
    for index, similarity in top_results:
        print(f"Document {index + 1}: {documents[index]}")
        print(f"Similarity: {similarity}\n")
        valid_chunks.append(documents[index])
    
    if not valid_chunks:
        print("TOOL RESULT: No chunks met the threshold.")
        return "The company brochure does not contain specific information regarding that request."

    return "\n\n".join(valid_chunks)

def build_agent(llm, tools):
    system_message = (
        "You are the EngagePro Official Assistant. "
        "Search the 'pdf_search' for any company-related questions. "
        "Answer the user's question using ONLY the retrieved context. "
        "If the answer is in the text, provide it clearly. "
        "If it is not there, use the wikipedia_search tool to search for the answers."
    )
    return create_react_agent(model=llm, tools=tools, prompt=system_message)


api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=10000)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
@tool
def wikipedia_search(question: str) -> str:
    """Useful for answering general knowledge questions, definitions, or 
    historical facts that are NOT found in pdf_search."""
    response = wikipedia_tool.run(question)
    print(response)
    return response

def process_question(question: str, llm):
    try:
        status = check_valid_question_tool(llm, question)
        
        if status.lower() != 'valid':
            return "I will not discuss this."
    except Exception as e:
        print(f"Safety Logic Error: {e}")

    tools = [pdf_search, wikipedia_search]

    st.session_state.agent = build_agent(llm, tools)

    history = []
    for m in st.session_state.messages:
        role = HumanMessage if m["role"] == "user" else AIMessage
        history.append(role(content=m["content"]))
    history.append(HumanMessage(content=question))

    response = st.session_state.agent.invoke({"messages": history})    
    
    for msg in reversed(response["messages"]):
        if isinstance(msg, AIMessage) and msg.content.strip():
            return msg.content

    return "The assistant was unable to find an answer. Please try rephrasing."

if question := st.chat_input("Ask me a question."):
    st.chat_message("user", avatar="human_icon.png").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    answer = process_question(question, llm)

    with st.chat_message("assistant", avatar="engagepro_favicon.png"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
