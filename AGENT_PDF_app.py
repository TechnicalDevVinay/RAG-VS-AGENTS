import streamlit as st
import os
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.agents import Tool, initialize_agent, AgentType
from bert_score import BERTScorer
from transformers import AutoTokenizer, AutoModel

def main():
    # Title and description
    st.title("🧠🩺 AGENT_PDF: AI-Powered Psychiatric Support")
    st.markdown("Enter a medical query to receive a structured diagnosis, treatment, and medication response, along with relevancy scores using BiomedBERT and Bio_ClinicalBERT.")

    # Check for GROQ API Key in environment
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ API Key not found in environment. Please set the GROQ_API_KEY environment variable.")
        st.stop()
    os.environ["GROQ_API_KEY"] = groq_api_key

    # Initialize LLM
    @st.cache_resource
    def initialize_llm():
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=1)

    llm = initialize_llm()

    # Initialize embeddings and ChromaDB
    @st.cache_resource
    def initialize_chroma():
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-t5-base")
        chroma_path = os.path.normpath(r"C:\Users\vinay\OneDrive\Desktop\STREAMLIT\chroma_db_medical")
        medical_db = Chroma(
            persist_directory=chroma_path,
            embedding_function=embedding_function
        )
        return medical_db

    try:
        medical_db = initialize_chroma()
    except Exception as e:
        st.error(f"Failed to load ChromaDB: {str(e)}. Please ensure the path exists and contains valid data.")
        st.stop()

    # Default user query
    input_user_query = "Please Provide A Detailed Patient Symptoms"
    query_input = input_user_query

    # Initialize BERTScorers for both models
    @st.cache_resource
    def initialize_bert_scorers():
        scorers = {}
        model_name_bio = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
        tokenizer_bio = AutoTokenizer.from_pretrained(model_name_bio)
        model_bio = AutoModel.from_pretrained(model_name_bio)
        scorers['BiomedBERT'] = BERTScorer(
            model_type=model_name_bio,
            num_layers=12,
            rescale_with_baseline=False,
            lang='en'
        )
        model_name_clin = 'emilyalsentzer/Bio_ClinicalBERT'
        tokenizer_clin = AutoTokenizer.from_pretrained(model_name_clin)
        model_clin = AutoModel.from_pretrained(model_name_clin)
        scorers['Bio_ClinicalBERT'] = BERTScorer(
            model_type=model_name_clin,
            num_layers=12,
            rescale_with_baseline=False,
            lang='en'
        )
        return scorers, tokenizer_bio

    scorers, biomed_tokenizer = initialize_bert_scorers()

    # Initialize session state
    if 'retrieved_info' not in st.session_state:
        st.session_state.retrieved_info = None
    if 'query_input' not in st.session_state:
        st.session_state.query_input = query_input

    # Function to classify query
    def classify_query(query):
        classification_prompt = f"""
        Classify the following user query as 'Medical' or 'General'.
        If it relates to symptoms, diagnosis, treatment, or medicine, classify it as 'Medical'.
        Otherwise, classify as 'General'.

        Query: {query}

        Response (Only output 'Medical' or 'General'):
        """
        response = llm.invoke(classification_prompt).content.strip()
        return response

    # Function to retrieve medical information
    def retrieve_medical(query):
        retriever = medical_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        return "\n\n".join([f"**Source:** {doc.metadata['source']} | **Page:** {doc.metadata['page_number']}\n{doc.page_content}" for doc in docs])

    # Create medical tool and agent
    @st.cache_resource
    def initialize_medical_agent(_llm):
        medical_tool = Tool(name="Medical Tool", func=retrieve_medical, description="Fetches medical information.")
        agent = initialize_agent([medical_tool], _llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)
        return agent

    medical_agent = initialize_medical_agent(llm)

    # Function to structure diagnosis output
    def cross_check_and_structure(query, medical_response):
        cross_check_prompt = f"""
        You are a psychiatrist expert doctor assistant. You have retrieved information from an AI agent.
        Your task is to:
        1. Cross-check the information for contradictions or inconsistencies.
        2. Structure a clear and concise psychiatric response in the following format:
        - For **Diagnosis**, list the 1 or 2 most probable diagnoses. For each, mention the diagnosis name and provide a brief explanation.
        - For **Treatment**, list the 1 or 2 most suitable treatments. For each, mention the treatment name and provide a brief explanation.
        - For **Medicine**, list the 1 or 2 most appropriate medicines. For each, mention the medicine name and provide a brief explanation of its purpose.

        🔹 **User Query:** {query}

        🔹 **Retrieved Information:**
        - **Diagnosis Data:** {medical_response}

        🔹 **Final Structured Response**
        (Write the final psychiatric response here. Do not include any headings such as 'Structured Response:' in your output.)
        """
        structured_response = llm.invoke(cross_check_prompt).content.strip()
        return structured_response

    # Function to check relevancy with BERTScore
    def check_relevancy(query, response, model_name):
        scorer = scorers[model_name]
        P, R, F1 = scorer.score([response], [query])
        return P.item(), R.item(), F1.item()

    # Main medical assistant function
    def medical_assistant(query, model_name):
        category = classify_query(query)
        if category != "Medical":
            return "Process stopped: Query classified as General.", None, None, None
        medical_response = medical_agent.invoke(query)
        final_response = cross_check_and_structure(query, medical_response)
        precision, recall, f1_score = check_relevancy(query, final_response, model_name)
        return final_response, precision, recall, f1_score

    # User input
    query_input = st.text_area("Enter your query:", value=st.session_state.query_input, height=100)

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

    if st.button("Process Query"):
        if not query_input.strip():
            st.warning("Please enter a query.")
        else:
            st.session_state.query_input = query_input
            # Classify query
            with st.spinner("Classifying query..."):
                try:
                    classified_query = classify_query(query_input)
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
    # st.markdown("---")
    st.markdown(
    "<hr style='border: 1px solid black;'>",
    unsafe_allow_html=True)
    st.markdown("Built with Streamlit | Powered by Groq, ChromaDB, BiomedBERT, and Bio_ClinicalBERT")

if __name__ == "__main__":
    main()
