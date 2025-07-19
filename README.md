# 🤖🧠 RAG vs AGENT — Mental Health AI Q&A System

Welcome to **RAG vs AGENT**, a biomedical Q&A project focused on **mental health** 🧠.  
This system enables users to compare two cutting-edge reasoning frameworks:  
- 🔗 **RAG** (Retrieval-Augmented Generation)  
- 🧠 **Agent-based LLM Reasoning**

Both utilize **LLaMA3-70B-8192** and draw from structured medical knowledge in **CSV datasets** and a renowned **psychiatric textbook PDF**.

🔬 Each generated answer is evaluated using domain-specific **BioBERT models**, scored by:
- 🎯 **Precision**
- 🔄 **Recall**
- 📊 **F1 Score**

---

## 💡 Project Highlights

- 🔄 Option Avaliable between **RAG** or **Agent** frameworks  
- 📘 Knowledge from:
  - `Kaplan and Sadock's Comprehensive Textbook of Psychiatry` (PDF)
  - Structured CSVs on **6 mental health disorders**
- 🧠 Powered by **LLaMA3-70B-8192** via Groq API  
- 🧪 Evaluate with  `BioMedBERT`, or `ClinicalBERT`
- 📊 Precision, Recall, and F1 Score analysis

---

## 🧠 Mental Health Conditions Covered

Structured data (CSV) includes detailed content on:

1. 🧠 **Bipolar Disorder**  
2. 👥 **Schizophrenia**  
3. 💔 **Borderline Personality Disorder (BPD)**  
4. 😟 **Anxiety Disorders**  
5. 😞 **Depression**  
6. 😰 **Panic Disorder**

Each disorder is organized into 5 consistent subcategories:

- 🧬 **Causes**  
- 📋 **Symptoms**  
- 💊 **Medical Interventions**  
- 🏃 **Exercises**  
- 🧠 **General Insights**

---

## 📂 Knowledge Sources

| Type     | Source |
|----------|--------|
| 📘 PDF    | *Kaplan and Sadock's Comprehensive Textbook of Psychiatry*  
| 📊 CSV    | Curated mental health content across 6 disorders and 5 subcategories  
| 🧠 Model  | `LLaMA3-70B-8192` via Groq API  
| 🗃️ Vector DB | `ChromaDB` used for semantic retrieval  

---

## 🛠️ Tech Stack

| Layer        | Tools & Frameworks                          |
|--------------|---------------------------------------------|
| 🎨 UI         | `Streamlit`, `HTML`, `CSS`                  |
| 🧠 LLM Logic | `LangChain`, `Groq`, `LLaMA3`, `RAG`, `Agents` |
| 📂 Data       | `CSV`, `PDF`, `ChromaDB`                    |
| 🧪 Evaluation | `BioMedBERT`, `ClinicalBERT`     |
| 🐍 Backend   | `Python`, `Torch`, `Transformers`           |

---

## 📊 Evaluation Metrics

Choose from the following BioBERT variants to evaluate answers:

 
- 🧬 **BioMedBERT** — Biomedical & clinical knowledge  
- 🏥 **ClinicalBERT** — Clinical health records  

Each Anser to User Query is Evaluated by Biobert Model with following units:

- 🎯 **Precision**  
- 🔄 **Recall**  
- 📊 **F1 Score**

---



