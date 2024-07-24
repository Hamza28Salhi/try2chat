import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import numpy as np

load_dotenv()
st.set_page_config(
    page_title="Chat with Google Gemini-Pro!",
    page_icon=":robot_face:",
    layout="wide",
)

# Initialize Gemini-Pro
genai.configure(api_key='AIzaSyDyci3wbUHQwpA5uKMMNtZKnVc68eYnJl4')
model = genai.GenerativeModel('gemini-pro')

# Initialize embeddings model
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize FAISS
embedding_dim = 768
index = faiss.IndexFlatL2(embedding_dim)

# Store text segments with their embeddings
texts = []
embeddings_list = []

def process_url(url):
    response = requests.get(url)
    yc_web_page = response.content

    soup = BeautifulSoup(yc_web_page, 'html.parser')

    for nav in soup.find_all('nav'):
        nav.decompose()
    for footer in soup.find_all('footer'):
        footer.decompose()

    elements = []
    for element in soup.find_all(['p', 'h3', 'ul', 'li', 'ol']):
        if element.name == 'p':
            elements.append(element.text.strip())
            next_sibling = element.find_next_sibling()
            if next_sibling and next_sibling.name == 'ol':
                list_items = [li.text.strip() for li in next_sibling.find_all('li')]
                elements.append(f"[Ordered List: {', '.join(list_items)}]")
        elif element.name == 'h3':
            elements.append(f"[Title: {element.text.strip()}]")
        elif element.name == 'ul':
            list_items = [li.text.strip() for li in element.find_all('li')]
            elements.append(f"[List: {', '.join(list_items)}]")
        elif element.name == 'ol' and element.previous_sibling and element.previous_sibling.name != 'p':
            list_items = [li.text.strip() for li in element.find_all('li')]
            elements.append(f"[Ordered List: {', '.join(list_items)}]")

    segments = []
    current_segment = []

    for element in elements:
        if element.startswith("[Title:"):
            if current_segment:
                segments.append(current_segment)
                current_segment = []
        current_segment.append(element)

    if current_segment:
        segments.append(current_segment)

    segments = [segment for segment in segments if any(segment)]

    for segment in segments:
        segment_text = " ".join(segment)
        embeddings = embeddings_model.embed_documents([segment_text])
        embeddings_list.append(embeddings[0])
        texts.append(segment_text)

process_url("https://kyra-docs.data-tricks.net/docs/documentation/actors")
process_url("https://kyra-docs.data-tricks.net/docs/documentation/tiers")
process_url("https://kyra-docs.data-tricks.net/docs/documentation/login")

# Add embeddings to FAISS index
np_embeddings = np.array(embeddings_list).astype("float32")
index.add(np_embeddings)

# Save and load FAISS index
faiss.write_index(index, "faiss_index.index")
index = faiss.read_index("faiss_index.index")

def search_index(query_embedding, top_k=5):
    distances, indices = index.search(np.array([query_embedding]).astype("float32"), top_k)
    return indices[0], distances[0]

def role_to_streamlit(role):
    if role == "model":
        return "assistant"
    else:
        return role

if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

st.title("Chat with Kyra Pro!")

for message in st.session_state.chat.history:
    with st.chat_message(role_to_streamlit(message.role)):
        st.markdown(message.parts[0].text)

if prompt := st.chat_input("I possess a well of knowledge about kyra what do you want to know?"):
    st.chat_message("user").markdown(prompt)
    
    # Generate embedding for the query
    query_embedding = embeddings_model.embed_documents([prompt])[0]
    
    # Search FAISS index
    indices, distances = search_index(query_embedding)
    
    # Retrieve and display results
    #search_results = [texts[idx] for idx in indices]
    #response_text = " ".join(search_results)
    
    # Send user entry to Gemini and read the response
    response = st.session_state.chat.send_message(prompt + "\n" )
    
    with st.chat_message("assistant"):
        st.markdown(response.text)
