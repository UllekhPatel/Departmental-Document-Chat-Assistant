import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from groq import Groq
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Mapping, Optional
import voyageai
from langchain.embeddings.base import Embeddings
import numpy as np
from htmlTemplates import css, bot_template, user_template
from pydantic import BaseModel, Field

load_dotenv()

class VoyageAIEmbeddings(Embeddings):
    def __init__(self, voyage_client, model_name="voyage-large-2-instruct", batch_size=128):
        self.client = voyage_client
        self.model_name = model_name
        self.dimension = None
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            try:
                embeddings = self.client.embed(batch, model=self.model_name, input_type="document")
                if hasattr(embeddings, 'embeddings'):
                    numpy_embeddings = [np.array(emb) for emb in embeddings.embeddings]
                elif isinstance(embeddings, list):
                    numpy_embeddings = [np.array(emb) for emb in embeddings]
                else:
                    raise ValueError(f"Unexpected embeddings type: {type(embeddings)}")
                
                all_embeddings.extend(numpy_embeddings)
                print(f"Processed batch {i//self.batch_size + 1} of size {len(batch)}")
            except Exception as e:
                print(f"Error processing batch {i//self.batch_size + 1}: {str(e)}")

        if self.dimension is None and all_embeddings:
            self.dimension = len(all_embeddings[0])
        
        return [emb.tolist() for emb in all_embeddings]  # Convert numpy arrays to lists

    def embed_query(self, text: str) -> List[float]:
        try:
            embedding = self.client.embed([text], model=self.model_name, input_type="query")
            if hasattr(embedding, 'embeddings'):
                numpy_embedding = np.array(embedding.embeddings[0])
            elif isinstance(embedding, list):
                numpy_embedding = np.array(embedding[0])
            else:
                raise ValueError(f"Unexpected query embedding type: {type(embedding)}")
            
            if self.dimension is None:
                self.dimension = len(numpy_embedding)
            return numpy_embedding.tolist()  # Convert numpy array to list
        except Exception as e:
            print(f"Error embedding query: {str(e)}")
            raise

    def embed_batch_documents(self, texts: List[str], batch_size: int = 128) -> List[List[float]]:
        """Embed a list of documents, combining results from multiple batches."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def embed_batch_queries(self, texts: List[str], batch_size: int = 128) -> List[List[float]]:
        """Embed a list of queries, combining results from multiple batches."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = [self.embed_query(text) for text in batch]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

from langchain.llms.base import LLM
from pydantic import BaseModel, Field
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import os
from groq import Groq

class GroqLLM(LLM, BaseModel):
    client: Groq = Field(default_factory=lambda: Groq(api_key=os.getenv("GROQ_API_KEY")))
    model_name: str = "llama3-70b-8192"
    department: str = "General"
    
    @property
    def system_prompt(self):
        prompts = {
            "Code":"You are an AI agent with extensive knowledge in coding. Assist the user with contextual coding queries, providing comprehensive guidance and solutions. ",
            "HR": "You are an HR assistant specializing in labor law and workplace policies. Provide comprehensive information on human resources, employee policies, and workplace regulations. Respond based on the content provided, and include additional relevant details as necessary.",
            "Finance": "You are a finance assistant specializing in financial reports, budgets, and accounting practices. Offer detailed information on financial analysis, reporting, and accounting principles. Respond based on the content provided, and include additional relevant insights as necessary.",
            "IT": "You are an IT assistant specializing in technology, software, and IT infrastructure. Provide detailed information on IT systems, software solutions, and infrastructure management. Respond based on the content provided, and include additional relevant insights as necessary.",
            "Marketing": "You are a marketing assistant specializing in marketing strategies, campaigns, and market analysis. Offer comprehensive information on digital marketing, advertising tactics, and consumer behavior analysis. Respond based on the content provided, and include additional relevant insights as necessary.",
            "Operations": "You are an operations assistant specializing in business processes, supply chain management, and operational efficiency. Provide detailed information on logistics, process optimization, and supply chain strategies. Respond based on the content provided, and include additional relevant insights as necessary.",
            "Law": "You are a legal assistant specializing in laws, regulations, contracts, and legal procedures. Your responses should not be considered as legal advice. Provide detailed information on legal frameworks, contract analysis, and procedural guidelines. Respond based on the content provided, and include additional relevant insights as necessary.",
            "General": "You are a helpful assistant dedicated to providing informative and supportive responses. Offer assistance across various topics, ensuring accuracy and clarity in your replies. Respond based on the context provided, and provide additional relevant information when beneficial."
        }
        return prompts.get(self.department, prompts["General"])
    
    class Config:
        arbitrary_types_allowed = True
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "groq"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name, "system_prompt": self.system_prompt}
def get_pdf_texts(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text
    
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len 
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks 

def get_embedding_model(department, text_content):
    if department == "Finance":
        return "voyage-finance-2"
    elif department == "Law":
        return "voyage-law-2"
    elif department == "Code":
        return "voyage-code-2"
    else:
        return "voyage-large-2-instruct"

def get_vector_store(text_chunks, department):
    vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    if isinstance(text_chunks, list):
        input_texts = text_chunks
    else:
        input_texts = [text_chunks]
    
    try:
        full_text = " ".join(input_texts)
        model_name = get_embedding_model(department, full_text)
        
        embeddings = VoyageAIEmbeddings(vo, model_name=model_name, batch_size=128)
        all_embeddings = embeddings.embed_batch_documents(input_texts)
        
        print(f"Using embedding model: {model_name}")
        print(f"Embedding dimension: {embeddings.dimension}")
        print(f"Total embeddings: {len(all_embeddings)}")
        
        # Create a list of tuples (text, embedding)
        text_embeddings = list(zip(input_texts, all_embeddings))
        
        vectorstore = FAISS.from_embeddings(text_embeddings, embeddings)
        vectorstore.embedding_function = embeddings
        return vectorstore
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        print(f"Detailed error: {e}")
        import traceback
        print(traceback.format_exc())
        return None
    
def get_conversation_chain(vectorstore, department):
    llm = GroqLLM(department=department)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    conversation_chain.retriever.vectorstore.embedding_function = vectorstore.embedding_function
    return conversation_chain

def handle_userinput(user_question, department):
    if department not in st.session_state.conversation:
        st.warning(f"No conversation chain found for {department}. Please process documents first.")
        return

    response = st.session_state.conversation[department]({'question': user_question})
    st.session_state.chat_history[department] = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history[department]):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def process_documents(pdf_docs, department):
    if pdf_docs:
        with st.spinner(f"Processing {department} documents"):
            raw_text = get_pdf_texts(pdf_docs)
            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vector_store(text_chunks, department)
                if vectorstore:
                    conversation_chain = get_conversation_chain(vectorstore, department)
                    st.session_state.conversation[department] = conversation_chain
                    return vectorstore
                else:
                    st.error("Error creating vector store. Please try again.")
            else:
                st.error("No text could be extracted from the uploaded PDFs.")
    else:
        st.error("Please upload at least one PDF.")
    return None

def main():
    load_dotenv()
    st.set_page_config(page_title="Departmental PDF Chat", page_icon=":office:")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    
    st.header("Departmental PDF Chat :office:")
    
    departments = ["General", "HR", "Finance", "IT", "Marketing", "Operations", "Law","Code"]
    selected_department = st.selectbox("Select your department:", departments)
    
    user_question = st.text_input(f"Ask a question about {'your documents' if selected_department == 'General' else f'{selected_department} documents'}:")
    if user_question:
        handle_userinput(user_question, selected_department)
    
    with st.sidebar:
        st.subheader(f"{'Your' if selected_department == 'General' else selected_department} Documents")
        pdf_docs = st.file_uploader(
            f"Upload {'PDFs' if selected_department == 'General' else f'{selected_department} PDFs'} here and click on 'Process'",
            accept_multiple_files=True
        )
        if st.button("Process"):
            vectorstore = process_documents(pdf_docs, selected_department)
            if vectorstore:
                st.success(f"Processing complete! You can now ask questions about your {selected_department} documents.")

if __name__ == '__main__':
    main()


#all the packages required 
# pip install streamlit
# pip install python-dotenv
# pip install PyPDF2
# pip install langchain-community
# pip install langchain
# pip install groq
# pip install voyageai
# pip install numpy
# pip install pydantic
