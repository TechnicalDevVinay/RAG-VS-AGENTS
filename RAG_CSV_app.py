import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from bert_score import BERTScorer
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import os
from dotenv import load_dotenv
import torch
import re

# Load environment variables
load_dotenv()

# Configuration
CHROMA_DB_PATH = os.path.normpath(r"C:\Users\vinay\OneDrive\Desktop\STREAMLIT\chroma_db_mental_health")

# Initialize session state keys with default values
if 'retrieved_info' not in st.session_state:
    st.session_state.retrieved_info = None
if 'query_input' not in st.session_state:
    st.session_state.query_input = "Please Provide A Detailed Patient Symptoms"

# Initialize ChromaDB
@st.cache_resource
def initialize_chroma():
    sqlite_file = os.path.join(CHROMA_DB_PATH, "chroma.sqlite3")
    if not os.path.exists(CHROMA_DB_PATH):
        st.error(f"ChromaDB directory not found at '{CHROMA_DB_PATH}'. Please verify the path exists.")
        raise FileNotFoundError(f"ChromaDB directory '{CHROMA_DB_PATH}' not found.")
    if not os.path.exists(sqlite_file):
        st.error(f"ChromaDB SQLite file not found at '{sqlite_file}'. Please verify or recreate the database.")
        raise FileNotFoundError(f"ChromaDB SQLite file not found at '{sqlite_file}'.")
    try:
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-t5-base")
        medical_db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embedding_function
        )
        if not medical_db._collection.count():
            st.warning("ChromaDB is empty. Please populate it with medical data using a population script.")
            raise ValueError("ChromaDB is empty.")
        return medical_db
    except Exception as e:
        st.error(f"Failed to load ChromaDB: {str(e)}. Please ensure '{CHROMA_DB_PATH}' contains valid data.")
        raise

# Initialize Groq LLM
@st.cache_resource
def init_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or len(api_key) < 10:
        raise ValueError("Invalid or missing GROQ_API_KEY in environment variables.")
    try:
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=1, api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Groq LLM: {e}")
        raise

# Initialize BERT scorers
@st.cache_resource
def init_bert_scorers():
    try:
        # BiomedBERT
        biomed_model_name = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
        biomed_tokenizer = AutoTokenizer.from_pretrained(biomed_model_name)
        biomed_model = AutoModel.from_pretrained(biomed_model_name).eval()
        biomed_scorer = BERTScorer(model_type=biomed_model_name, num_layers=12, rescale_with_baseline=False, lang='en')
        if torch.cuda.is_available():
            biomed_model = biomed_model.cuda()

        # ClinicalBERT
        clinical_model_name = 'emilyalsentzer/Bio_ClinicalBERT'
        clinical_tokenizer = AutoTokenizer.from_pretrained(clinical_model_name)
        clinical_model = AutoModel.from_pretrained(clinical_model_name).eval()
        clinical_scorer = BERTScorer(model_type=clinical_model_name, num_layers=12, rescale_with_baseline=False, lang='en')
        if torch.cuda.is_available():
            clinical_model = clinical_model.cuda()

        return biomed_scorer, biomed_tokenizer, clinical_scorer, clinical_tokenizer
    except Exception as e:
        st.error(f"Failed to load BERT models: {e}. Try reducing memory usage by commenting out one BERT model.")
        raise

# Classify query
def classify_query(query, llm):
    classification_prompt = f"""
    Classify the following user query as 'Medical' or 'General'.
    If it relates to symptoms, diagnosis, treatment, or medicine, classify it as 'Medical'.
    Otherwise, classify as 'General'.

    Query: {query}

    Response (Only output 'Medical' or 'General'):
    """
    response = llm.invoke(classification_prompt).content.strip()
    return response

# Retrieve information from ChromaDB
def retrieve_from_chroma(classified_query, chroma_db, user_query):
    if classified_query == "Medical":
        try:
            retriever = chroma_db.as_retriever(search_kwargs={"k": 3})
            results = retriever.get_relevant_documents(user_query)
            if results:
                return {"Medical": results}
            else:
                return None
        except Exception as e:
            st.error(f"Error retrieving from ChromaDB: {e}")
            return None
    else:
        return None

# Format retrieved information
def format_retrieved_info(retrieved_info, biomed_tokenizer, clinical_tokenizer, max_tokens=500):
    if not retrieved_info or "Medical" not in retrieved_info:
        return "No relevant information retrieved.", None, None
    output = "**Medical Results:**\n"
    biomed_full_text = ""
    clinical_full_text = ""
    for idx, res in enumerate(retrieved_info["Medical"], start=1):
        response = res.page_content
        cleaned_response = re.sub(r'\s*\[Mental Health Type:.*?\]', '', response).strip()
        biomed_full_text += cleaned_response + " "
        clinical_full_text += cleaned_response + " "
        source = res.metadata.get('source', 'Unknown')
        row_index = res.metadata.get('row_index', 'N/A')
        output += f"\n**Response {idx}:** {cleaned_response}\n"
        output += f"**Source:** {source}\n"
        output += f"**Row Index:** {row_index}\n"

    # Truncate
    biomed_tokens = biomed_tokenizer.tokenize(biomed_full_text)
    biomed_truncated_text = biomed_tokenizer.convert_tokens_to_string(biomed_tokens[:max_tokens])
    clinical_tokens = clinical_tokenizer.tokenize(clinical_full_text)
    clinical_truncated_text = clinical_tokenizer.convert_tokens_to_string(clinical_tokens[:max_tokens])

    return output, biomed_truncated_text, clinical_truncated_text

# Check relevancy with BERTScore
def check_relevancy(query_input, biomed_truncated_text, clinical_truncated_text, scorer, model_type):
    if not (biomed_truncated_text or clinical_truncated_text):
        return None
    truncated_text = biomed_truncated_text if model_type == "BiomedBERT" else clinical_truncated_text
    if not truncated_text:
        return None
    P, R, F1 = scorer.score([truncated_text], [query_input])
    return P.item(), R.item(), F1.item()

# Define the medical_assistant function
def medical_assistant(query_input, model_type):
    medical_db = initialize_chroma()
    llm = init_llm()
    biomed_scorer, biomed_tokenizer, clinical_scorer, clinical_tokenizer = init_bert_scorers()
    classified_query = classify_query(query_input, llm)
    retrieved_info = retrieve_from_chroma(classified_query, medical_db, query_input)
    output, biomed_truncated_text, clinical_truncated_text = format_retrieved_info(
        retrieved_info, biomed_tokenizer, clinical_tokenizer
    )

    if model_type == "BiomedBERT":
        scorer = biomed_scorer
        truncated_text = biomed_truncated_text
    else:
        scorer = clinical_scorer
        truncated_text = clinical_truncated_text

    if truncated_text:
        precision, recall, f1_score = check_relevancy(
            query_input, biomed_truncated_text, clinical_truncated_text, scorer, model_type
        )
    else:
        precision, recall, f1_score = None, None, None

    return output, precision, recall, f1_score

# MAIN STREAMLIT APP
def main():
    st.title("🧠🩺 RAG_CSV: RAG-Powered Psychiatric Support")
    st.markdown("Enter a medical query to receive a structured diagnosis, treatment, and medication response, along with relevancy scores using BiomedBERT and Bio_ClinicalBERT.")

    # Load resources
    with st.spinner("Loading ChromaDB and models (this may take a minute)..."):
        try:
            medical_db = initialize_chroma()
            llm = init_llm()
            biomed_scorer, biomed_tokenizer, clinical_scorer, clinical_tokenizer = init_bert_scorers()
        except Exception as e:
            st.error(f"Error loading resources: {e}")
            st.stop()

    # User input
    query_input = st.text_area("Enter your query:", value=st.session_state.query_input, height=100)

    st.markdown("**Models Used To Check Relevancy:** Both (BiomedBERT & ClinicalBERT)")

    st.markdown("""
    <style>
        div.stButton > button {
        background-color: #FFFFFF !important;
        border: 2px solid #000000 !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        padding: 0.5em 1em !important;
        }
    </style>
    """, unsafe_allow_html=True)

    if st.button("Process Query"):
        if not query_input.strip():
            st.warning("Please enter a query.")
        else:
            st.session_state.query_input = query_input
            with st.spinner("Classifying query..."):
                try:
                    classified_query = classify_query(query_input, llm)  # ✅ FIXED: Passed `llm`
                    st.write(f"**Query Classification:** {classified_query}")
                except Exception as e:
                    st.error(f"Error classifying query: {e}")
                    st.stop()

            if classified_query == "Medical":
                with st.spinner("Retrieving information..."):
                    try:
                        response_bio, precision_bio, recall_bio, f1_score_bio = medical_assistant(query_input, "BiomedBERT")
                        response_clin, precision_clin, recall_clin, f1_score_clin = medical_assistant(query_input, "Bio_ClinicalBERT")
                        st.session_state.retrieved_info = response_bio
                        scores_data = [
                            {"Model": "BiomedBERT", "Precision": precision_bio, "Recall": recall_bio, "F1 Score": f1_score_bio},
                            {"Model": "Bio_ClinicalBERT", "Precision": precision_clin, "Recall": recall_clin, "F1 Score": f1_score_clin}
                        ]
                    except Exception as e:
                        st.error(f"Error retrieving information: {e}")
                        st.session_state.retrieved_info = None
                        st.stop()

                if st.session_state.retrieved_info:
                    st.markdown("### 📝 Retrieved Information")
                    st.markdown(st.session_state.retrieved_info)
                else:
                    st.warning("No medical information retrieved. The ChromaDB may be empty or the query may not match any documents.")

                with st.spinner("Calculating relevancy scores..."):
                    try:
                        if scores_data:
                            st.markdown("### 📊 Relevancy Scores")
                            scores_df = pd.DataFrame(scores_data)
                            st.table(
                                scores_df.style
                                .format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1 Score": "{:.4f}"})
                                .set_table_styles([
                                    {"selector": "th, td", "props": [("border", "2px solid black"), ("border-collapse", "collapse")]},
                                    {"selector": "table", "props": [("border", "2px solid black"), ("border-collapse", "collapse")]}
                                ])
                            )
                        else:
                            st.error("Error calculating relevancy scores: No medical responses found.")
                    except Exception as e:
                        st.error(f"Error calculating relevancy scores: {e}")
            else:
                st.warning("Process stopped: Query classified as General.")
                st.session_state.retrieved_info = None

    st.markdown("<hr style='border: 1px solid black;'>", unsafe_allow_html=True)
    st.markdown("Built with Streamlit | Powered by Groq, ChromaDB, BiomedBERT, and Bio_ClinicalBERT")

if __name__ == "__main__":
    main()
