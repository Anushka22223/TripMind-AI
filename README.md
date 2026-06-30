# ✈️ TripMind AI – Multi-Agent Travel Planning System

An intelligent AI-powered travel planning system that generates complete travel itineraries including **flights, hotels, and day-wise plans** using a multi-agent architecture built with **LangGraph, LangChain, and Groq LLMs**.

---

## Features

- Multi-Agent AI workflow powered by LangGraph
- Query Rewrite Agent for improved retrieval and search quality
- Intelligent routing between RAG, Web Search, and Hybrid Retrieval
- Retrieval-Augmented Generation (RAG) using uploaded PDF knowledge bases
- Semantic document retrieval using FAISS Vector Store and HuggingFace Embeddings
- Real-time web search using Tavily Search API
- Flight information retrieval using AviationStack API
- Hotel recommendation system using web search
- Weather-aware travel planning through a dedicated Weather Agent
- AI-generated day-wise personalized travel itineraries
- Persistent conversation memory using PostgreSQL and LangGraph PostgresSaver
- Thread-based conversation management and history restoration
- Real-time agent execution tracking with Streamlit
- Downloadable travel plans in Markdown format

---

##  System Architecture

User Query  

1. **Query Rewrite Agent** improves the user's query for better retrieval quality.
2. **Router Agent** decides whether the query should use RAG, Web Search, or Hybrid Retrieval.
3. **RAG/Web/Hybrid Agents** gather relevant travel information from PDFs, the internet, or both.
4. **Flight Agent** extracts travel routes and retrieves flight information.
5. **Hotel Agent** fetches hotel recommendations.
6. **Weather Agent** gathers destination weather forecasts and travel conditions.
7. **Itinerary Agent** generates a personalized day-wise travel plan using all collected context.
8. **Final Agent** produces a polished travel itinerary for the user.

---

## Tech Stack

### Programming Language
- Python

### AI & LLM
- LangGraph
- LangChain
- Groq
- LLaMA 3.3 70B

### Retrieval-Augmented Generation (RAG)
- FAISS Vector Store
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- PyPDFLoader
- RecursiveCharacterTextSplitter

### External APIs
- Tavily Search API
- AviationStack API

### Database & Persistence
- PostgreSQL
- LangGraph PostgresSaver
- psycopg
- psycopg_pool

### Frontend
- Streamlit

### Environment & Configuration
- Python Dotenv

### Core Concepts Implemented
- Multi-Agent Systems
- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Conditional Routing
- State Management
- Checkpointing & Memory Persistence

---

## 📂 Project Structure

```text
tripmind-ai/
│
├── main.py                 # LangGraph workflow and agent orchestration
├── frontend.py             # Streamlit user interface
│
├── tools/
│   ├── rag_tool.py         # PDF processing, embeddings, and FAISS retrieval
│   ├── tavily_tool.py      # Web search using Tavily API
│   ├── flight_tool.py      # Flight information retrieval using AviationStack
│   └── weather_tool.py     # Weather information retrieval
│
├── faiss_index/            # Generated FAISS vector database
│
├── .env                    # API keys and database credentials
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
│
└── assets/                 # Optional images/screenshots for README
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```git
git clone https://github.com/your-username/tripmind-ai.git
cd tripmind-ai
```

### 2. Create a Virtual Environment

```python
python -m venv venv
```

### 3. Activate the Environment

**Windows**

```
venv\Scripts\activate
```

**Mac/Linux**

```
source venv/bin/activate
```

### 4. Install Dependencies

```
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the project root directory:

```env
GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
AVIATIONSTACK_API_KEY=your_key
DATABASE_URL=your_postgresql_url
```

### 6. Run the Application

```
streamlit run frontend.py
```

### 7. Enable RAG (Optional)

- Upload a PDF through the Streamlit interface
- Click **Build RAG Knowledge Base**
- A FAISS vector store will be created automatically for document retrieval

### 💡 Example Query

> Plan a 7-day Japan trip from Delhi under ₹2 lakh including flights, hotels, weather forecasts, and sightseeing recommendations.



##  Future Improvements

### Real-time Integrations
- Live flight booking system integration  
- Google Maps API for route visualization  

### AI Enhancements
- AI-based price prediction for flights and hotels  
- Smarter itinerary optimization using user behavior  

### Budget-Aware Travel Planning
- Introduce budget optimization to recommend flights, hotels, and activities based on user spending limits.

###  User Authentication & Saved Trips
- Add user accounts to save, manage, and revisit travel plans across multiple sessions and devices.

## 👩‍💻 Author

### Anushka Rani

B.Tech, Electrical Engineering | MNIT Jaipur

Interested in AI, Multi-Agent Systems, Retrieval-Augmented Generation (RAG), Backend Development, and Large Language Model Applications.

Built **TripMind AI**, a multi-agent travel planning system leveraging LangGraph, FAISS, PostgreSQL, and Groq-powered LLMs.
```
