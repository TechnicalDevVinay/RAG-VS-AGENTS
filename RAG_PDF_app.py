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

def main():
    # Load environment variables
    load_dotenv()

    # Configuration
    CHROMA_DB_PATH = os.path.normpath(r"C:\Users\vinay\OneDrive\Desktop\STREAMLIT\chroma_db_medical")

    # Initialize session state
    if 'retrieved_info' not in st.session_state:
        st.session_state.retrieved_info = None
    if 'query_input' not in st.session_state:
        st.session_state.query_input = ""

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

    # Retrieve and format information
    def retrieve_from_chroma(classified_query, medical_db, query_input):
        if classified_query == "Medical":
            medical_retriever = medical_db.as_retriever()
            medical_results = medical_retriever.get_relevant_documents(query_input)[:3]
            return {"Medical": medical_results}
        else:
            return None

    def format_retrieved_info(retrieved_info, biomed_tokenizer, clinical_tokenizer, max_tokens=400):
        if not retrieved_info:
            return "No results found.", None, None
        output = ""
        biomed_full_text = ""
        clinical_full_text = ""
        for category, results in retrieved_info.items():
            output += f"**{category} Results:**\n"
            for idx, res in enumerate(results, start=1):
                response = res.page_content
                biomed_full_text += response + " "
                clinical_full_text += response + " "
                pdf_name = res.metadata.get('source', 'Unknown')
                page_number = res.metadata.get('page_number', 'Unknown')
                output += f"\n**Response {idx}:** {response}\n"
                output += f"**PDF Name:** {pdf_name}\n"
                output += f"**Page Number:** {page_number}\n"

        # Truncate for BiomedBERT
        biomed_tokens = biomed_tokenizer.tokenize(biomed_full_text)
        if len(biomed_tokens) > max_tokens:
            biomed_tokens = biomed_tokens[:max_tokens]
        biomed_truncated_text = biomed_tokenizer.convert_tokens_to_string(biomed_tokens)

        # Truncate for ClinicalBERT
        clinical_tokens = clinical_tokenizer.tokenize(clinical_full_text)
        if len(clinical_tokens) > max_tokens:
            clinical_tokens = clinical_tokens[:max_tokens]
        clinical_truncated_text = clinical_tokenizer.convert_tokens_to_string(clinical_tokens)

        return output, biomed_truncated_text, clinical_truncated_text

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

    def check_relevancy(query_input, biomed_truncated_text, clinical_truncated_text, scorer, model_type):
        if not (biomed_truncated_text or clinical_truncated_text):
            return None
        truncated_text = biomed_truncated_text if model_type == "BiomedBERT" else clinical_truncated_text
        if not truncated_text:
            return None
        P, R, F1 = scorer.score([truncated_text], [query_input])
        return P.item(), R.item(), F1.item()

    # --- Start UI ---

    # Title and description
    st.title("🧠🩺 RAG_PDF: RAG-Powered Psychiatric Support")
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
    query_input = st.text_area("Enter your query:", value="Please Provide A Detailed Patient Symptoms", height=100)

   # Display fixed relevancy method
    st.markdown("**Models Used To Check Relevancy:** Both (BiomedBERT & ClinicalBERT)")
    relevancy_method = "Both (BiomedBERT & ClinicalBERT)"

    # Inject CSS for white text on black background button
    st.markdown("""
    <style>
        div.stButton > button {
        background-color: #FFFFFF !important;  /* White background */
        border: 2px solid #000000 !important;  /* Black border */
        font-weight: bold !important;
        border-radius: 8px !important;
        padding: 0.5em 1em !important;
        }
    </style>
    """, unsafe_allow_html=True)

    def medical_assistant(query_input, model_type):
        # Retrieve relevant information
        retrieved_info = retrieve_from_chroma("Medical", medical_db, query_input)
        if not retrieved_info:
            return "No results found.", None, None, None

        # Format retrieved info and get truncated text for BERT models
        output, biomed_truncated_text, clinical_truncated_text = format_retrieved_info(
            retrieved_info, biomed_tokenizer, clinical_tokenizer
        )

        # Calculate relevancy scores
        if model_type == "BiomedBERT":
            scorer = biomed_scorer
            truncated_text = biomed_truncated_text
        else:
            scorer = clinical_scorer
            truncated_text = clinical_truncated_text

        if truncated_text:
            precision, recall, f1_score = check_relevancy(query_input, biomed_truncated_text, clinical_truncated_text, scorer, model_type)
        else:
            precision, recall, f1_score = None, None, None

        return output, precision, recall, f1_score

    if st.button("Process Query"):
        if not query_input.strip():
            st.warning("Please enter a query.")
        else:
            st.session_state.query_input = query_input
            # Classify query
            with st.spinner("Classifying query..."):
                try:
                    classified_query = classify_query(query_input, llm)
                    st.write(f"**Query Classification:** {classified_query}")
                except Exception as e:
                    st.error(f"Error classifying query: {e}")
                    st.stop()

            if classified_query == "Medical":
                # Process medical query
                with st.spinner("Retrieving information..."):
                    try:
                        # Process both models
                        response_bio, precision_bio, recall_bio, f1_score_bio = medical_assistant(query_input, "BiomedBERT")
                        response_clin, precision_clin, recall_clin, f1_score_clin = medical_assistant(query_input, "Bio_ClinicalBERT")
                        st.session_state.retrieved_info = response_bio  # Using BiomedBERT response as primary
                        scores_data = [
                            {"Model": "BiomedBERT", "Precision": precision_bio, "Recall": recall_bio, "F1 Score": f1_score_bio},
                            {"Model": "Bio_ClinicalBERT", "Precision": precision_clin, "Recall": recall_clin, "F1 Score": f1_score_clin}
                        ]
                    except Exception as e:
                        st.error(f"Error retrieving information: {e}")
                        st.session_state.retrieved_info = None
                        st.stop()

                # Display results
                if st.session_state.retrieved_info:
                    st.markdown("### 📝 Retrieved Information")
                    st.markdown(st.session_state.retrieved_info)
                else:
                    st.warning("No medical information retrieved. The ChromaDB may be empty or the query may not match any documents.")

                # Display relevancy scores
                with st.spinner("Calculating relevancy scores..."):
                    try:
                        if scores_data:
                            st.markdown("### 📊 Relevancy Scores")
                            scores_df = pd.DataFrame(scores_data)
                            st.table(
                                scores_df.style
                                .format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1 Score": "{:.4f}"})
                                .set_table_styles([
                                    {
                                        "selector": "th, td",
                                        "props": [("border", "2px solid black"), ("border-collapse", "collapse")]
                                    },
                                    {
                                        "selector": "table",
                                        "props": [("border", "2px solid black"), ("border-collapse", "collapse")]
                                    }
                                ])
                            )
                        else:
                            st.error("Error calculating relevancy scores: No medical responses found.")
                    except Exception as e:
                        st.error(f"Error calculating relevancy scores: {e}")
            else:
                st.warning("Process stopped: Query classified as General.")
                st.session_state.retrieved_info = None

    # Footer
    st.markdown(
    "<hr style='border: 1px solid black;'>",
    unsafe_allow_html=True)
    st.markdown("Built with Streamlit | Powered by Groq, ChromaDB, BiomedBERT, and Bio_ClinicalBERT")

if __name__ == "__main__":
    main()
