import streamlit as st
import importlib

st.set_page_config(page_title="🧠 Psychiatric Support Hub", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #E0FFFF !important;
        color: black !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .main, .block-container {
        background-color: #E0FFFF !important;
    }
    .stApp {
        background-color: #E0FFFF !important;
    }
    header[data-testid="stHeader"] {
        background-color: #E0FFFF !important;
    }
    header[data-testid="stHeader"]::before {
        background: none !important;
    }
    .markdown-text-container, .stMarkdown, h1, h2, h3, h4, h5, h6, p, li {
        color: black !important;
    }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        color: black !important;
        border: 2px solid black;
        border-radius: 8px;
        padding: 10px;
        box-sizing: border-box;
    }
    [data-testid="stSidebar"] * {
        color: black !important;
    }
    [data-testid="stSidebar"] .stRadio > div {
        flex-direction: column;
    }
    [data-testid="stSidebar"] .stRadio > div > label {
    width: 100%;
    box-sizing: border-box;
    text-align: center;
    border: 1px solid #B3B3B3;
    background-color: #D9D9D9;
    border-radius: 8px;
    padding: 10px 15px;
    margin-bottom: 8px;
    cursor: pointer;
    color: black !important;
    font-weight: 500;
    transition: all 0.3s ease;
    }

    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background-color: #C0C0C0;
    }
    [data-testid="stSidebar"] .stRadio > div > label[data-selected="true"] {
        background-color: #B3B3B3;
        color: black !important;
        border: 1px solid #999999;
    }
    hr {
        border: none;
        border-top: 1px solid black;
        margin: 30px 0;
        width: 60%;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar layout
st.sidebar.markdown(
    "<h2>🧠 <u>Psychiatric Support Hub</u></h2>",
    unsafe_allow_html=True
)

# Navigation
navigation = ["Home", "About_Us", "AGENT_PDF_app", "AGENT_CSV_app", "RAG_PDF_app", "RAG_CSV_app"]
selected_page = st.sidebar.radio("Select Page or Application", navigation)

# App router
def run_selected_app(app_name):
    if app_name == "Home":
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<h2 style='text-align: center;'>🧠🩺 <u><strong>Psychiatric Clinical Support Hub</strong></u></h2>", unsafe_allow_html=True)

            st.markdown("""
            <div style="text-align: center; font-size: 18px; padding: 10px 0;">
                <p>Welcome to your mental wellness companion — where <strong>AI meets care</strong>.</p>
            </div>

            <div style="background-color: #f0ffff; padding: 20px; border-radius: 10px; margin-top: 20px;">
                <h4 style='text-align: center;'>🧠 <u>What is Psychiatry?</u></h4>
                <p style='text-align: justify;'>
                    Psychiatry is the branch of medicine dedicated to the <strong>understanding, diagnosis, and treatment</strong> of mental, emotional, and behavioral disorders. It helps individuals struggling with conditions like anxiety, depression, bipolar disorder, and stress-related issues. Mental health matters — and help is available.
                </p>
            </div>

            <div style="background-color: #e6f7ff; padding: 20px; border-radius: 10px; margin-top: 20px;">
                <h4 style='text-align: center;'>💡 <u>How This Hub Supports You</u></h4>
                <ul style='line-height: 1.6;'>
                    <li>✔️ <strong>Fast Insight:</strong> Clinicians can instantly analyze files for key takeaways.</li>
                    <li>✔️ <strong>Smart Assistance:</strong> AI models help discover patterns and guide care plans.</li>
                    <li>✔️ <strong>Clarity for Patients:</strong> Better communication and understanding of mental health conditions.</li>
                </ul>
            </div>

            <div style="margin-top: 30px;">
                <h4 style='text-align: center;'>🛠️ <u>Tools You Can Use</u></h4>
                <ul style="line-height: 1.6;">
                    <li>📄 <strong>AGENT_PDF_app:</strong> Instantly ask questions and extract insights from your PDF documents.</li>
                    <li>📊 <strong>AGENT_CSV_app:</strong> Analyze and query CSV data effortlessly.</li>
                    <li>📚 <strong>RAG_PDF_app:</strong> Use advanced retrieval techniques to deeply understand PDF content.</li>
                    <li>📈 <strong>RAG_CSV_app:</strong> Combine retrieval and generation for rich CSV-based insights.</li>
                </ul>
            </div>

            <hr style="margin-top: 40px;">

            <div style="text-align: center; font-size: 16px;">
                👉 <strong>Select a tool from the sidebar</strong> to get started on your journey.
            </div>

            <div style="text-align: center; margin-top: 30px;">
                <h5>⚙️ Powered by:</h5>
                <p>LangChain • Groq LLM • ChromaDB • BERTScore (BiomedBERT + Bio_ClinicalBERT)</p>
            </div>
            """, unsafe_allow_html=True)



    else:
        try:
            app_module = importlib.import_module(app_name)
            if hasattr(app_module, "main"):
                app_module.main()
            else:
                st.error(f"The app '{app_name}' does not have a main() function to run.")
        except Exception as e:
            st.error(f"Failed to load {app_name}: {e}")

# Run selected app
run_selected_app(selected_page)
