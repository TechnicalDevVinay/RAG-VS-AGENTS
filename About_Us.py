
import streamlit as st
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/gif;base64,{encoded}"

def main():
    st.markdown("""
        <style>
            .centered {
                display: flex;
                justify-content: center;
                align-items: center;
                text-align: center;
                flex-direction: column;
            }
            .gif-row {
                display: flex;
                justify-content: center;
                gap: 40px;
                margin-top: 20px;
                flex-wrap: wrap;
            }
            .gif-column {
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
                max-width: 300px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title and Intro
    st.markdown(
        "<div class='centered'><h1>🧠🩺 <u>Psychiatric Clinical Hub</u></h1><h4><i><u>Smart support for clinicians.</u></i></h4></div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown(
        "<div class='centered'><h3>🧠 <u>Empowering Mental Health Professionals</u></h3></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class='centered'>
            <p><i>This platform provides four intelligent tools designed to assist psychiatric and mental health professionals 
            in handling data and documents more efficiently.</i></p>
        </div>

        <!-- Add vertical space -->
        <div style="height: 3em;"></div>
    """,
        unsafe_allow_html=True
    )


    mh_base64 = get_base64_image("Media/MH.gif")

    # Display MH.gif aligned to the right
    st.markdown(f"""
        <div style='display: flex; justify-content: center;'>
            <img src="{mh_base64}" width="400" style="border: 5px solid black; border-radius: 8px;">
        </div>
        <div style='text-align: center; margin-right: 20px; font-style: italic;'>
            
        </div>
        
        <!-- Add vertical space -->
        <div style="height: 3em;"></div>
    """, unsafe_allow_html=True)



    # Tech stack badge
    st.markdown(
        """
        <div class='centered'>
            <div style='background-color: #e0e0e0; padding: 10px 15px; border-radius: 8px; display: inline-block; font-weight: bold; font-size: 1.1em;'>
                Supported Tech Stack<br>
                LangChain | Groq LLM | ChromaDB | BiomedBERT | Bio_ClinicalBERT
            </div>
        </div>

        <!-- Add vertical space -->
        <div style="height: 6em;"></div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
    <hr style="border: 2px solid black; width: 100%;">
    """, unsafe_allow_html=True)


    # Encode AGENT.gif, RAG.gif, CSV.gif as base64
    agent_base64 = get_base64_image("Media/AGENT.gif")
    rag_base64 = get_base64_image("Media/RAG.gif")
    csv_base64 = get_base64_image("Media/CSV.gif")

    # PDF Knowledgebase Section (using base64 for GIF)
    st.markdown("<div class='centered'><h3>📄 <u>PDF Knowledgebase</u></h3></div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-style: italic; color: #333;'>Both apps use PDF as their knowledge source.</p>", unsafe_allow_html=True)

    
    pdf_kb_base64 = get_base64_image("Media/PDF.gif")
    
    st.markdown(f"""
        <div style="display: flex; justify-content: center;">
            <img src="{pdf_kb_base64}" width="200" style="border: 3px solid black; border-radius: 8px;">
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='centered'>", unsafe_allow_html=True)




    # First row: AGENT_PDF_app and RAG_CSV_app
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; max-width: 1500px; margin: 0 auto; padding: 0 20px;">
    <div style="flex: 1; text-align: left; margin-left: 210px;">
        <img src="{agent_base64}" width="180" Height="180" style="border: 3px solid black; border-radius: 8px;">
        <p><strong><u>AGENT_PDF_app</u></strong><br><i>Instantly ask questions and get insights from your PDF documents.</i></p>
    </div>

    <div style="flex: 1; text-align: left; margin-left: 250px;">
        <img src="{rag_base64}" width="200" Height="180" style="border: 3px solid black; border-radius: 8px;">
        <p><strong><u>RAG_CSV_app</u></strong><br><i>Gain advanced insights from your CSV datasets with retrieval + generation.</i></p>
    </div>




    </div>
    <!-- Add vertical space -->
    <div style="height: 6em;"></div>
""", unsafe_allow_html=True)

    st.markdown("""
    <hr style="border: 2px solid black; width: 100%;">
    """, unsafe_allow_html=True)





    # CSV Knowledgebase Section (using base64 for GIF)
    st.markdown("<div class='centered'><h3>📄 <u>CSV Knowledgebase</u></h3></div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-style: italic; color: #333;'>Both apps use PDF as their knowledge source.</p>", unsafe_allow_html=True)

    
    pdf_kb_base64 = get_base64_image("Media/CSV.gif")
    
    st.markdown(f"""
        <div style="display: flex; justify-content: center;">
            <img src="{pdf_kb_base64}" width="220" height="200" style="border: 3px solid black; border-radius: 8px;">
        </div>

        <!-- Add vertical space -->
        <div style="height: 1em;"></div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='centered'>", unsafe_allow_html=True)


    # Second row: RAG_PDF_app and AGENT_CSV_app (using base64 again)
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; max-width: 1500px; margin: 0 auto; padding: 0 20px;">
    <div style="flex: 1; text-align: left; margin-left: 250px;">
        <img src="{rag_base64}" width="200" Height="180" style="border: 3px solid black; border-radius: 8px;">
        <p><strong><u>RAG_CSV_app</u></strong><br><i>Gain advanced insights from your CSV datasets with retrieval + generation.</i></p>
    </div>

    <div style="flex: 1; text-align: left; margin-left: 210px;">
        <img src="{agent_base64}" width="180" Height="180" style="border: 3px solid black; border-radius: 8px;">
        <p><strong><u>AGENT_PDF_app</u></strong><br><i>Instantly ask questions and get insights from your PDF documents.</i></p>
    </div>

    <!-- Add vertical space -->
    <div style="height: 10em;"></div>    
""", unsafe_allow_html=True)


    st.markdown("""
    <hr style="border: 2px solid black; width: 100%;">
    """, unsafe_allow_html=True)


    # Final section
    st.markdown(
        "<div class='centered'><h3>🔍 Why These Tools?</h3></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class='centered'>
            <p>
            These tools integrate technologies like LangChain, ChromaDB, and Groq LLM with biomedical NLP models such 
            as BiomedBERT and Bio_ClinicalBERT to offer highly responsive, domain-specific insights tailored for 
            psychiatric data analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    # st.markdown(
    #     "<div class='centered'><h3>⚙️ Powered by:</h3></div>",
    #     unsafe_allow_html=True
    # )
