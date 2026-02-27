# VITMAS-INTERNAL-HACK-2026
CLANKER — The Freshman Survival Database
CLANKER is an AI-powered campus assistant designed to help new students quickly find essential information about university life — from medical emergencies and hostel rules to WiFi issues, mess timings, and academic procedures.
Built as a prototype for an internal hack at VIT Vellore, the system demonstrates how lightweight Natural Language Processing can deliver fast, offline-friendly student support without requiring large language models or internet connectivity.
## Problem Statement

New students often struggle to find reliable information about campus facilities, rules, and procedures. Important details are scattered across portals, PDFs, and informal sources.
CLANKER centralizes this information into a conversational interface that provides quick, context-aware answers.

##  Key Features

- Intent detection for different query types  
- Semantic search using TF-IDF and cosine similarity  
- Emergency-aware responses  
- Natural language query support  
- Dark/Light mode interface  
- Lightweight and offline-friendly  
- Streamlit web application  


##  How It Works

1. User enters a query in natural language  
2. Text is cleaned and normalized  
3. Intent is detected using keyword scoring  
4. Relevant questions are vectorized using TF-IDF  
5. Cosine similarity retrieves the best matches  
6. Answers are formatted and displayed  

Medical emergencies trigger broader information retrieval.


## 🛠️ Tech Stack

- Python  
- Streamlit  
- NLTK  
- Scikit-learn  
- TF-IDF Vectorization  
- Cosine Similarity  

##  Limitations

- Limited to predefined campus data  
- Rule-based intent detection  
- Not connected to official databases

##  Future Improvements

- Integration with official campus services  
- Multilingual support  
- Mobile application  
- LLM-based conversational capabilities

##  Disclaimer

This prototype may contain inaccuracies. For critical issues, contact official campus authorities.

##  Installation & Usage

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
streamlit run app.py
