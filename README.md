# Real Estate Investment Analyzer

**A decision-support tool for evaluating Buy vs Rent outcomes in Indian real estate.**

---

## 1. Overview

### The Problem

Real estate investment decisions in India often rely on intuition rather than data. Buyers struggle to objectively compare the long-term financial outcomes of purchasing a property versus renting and investing the difference elsewhere.

### The Solution

This application provides **data-driven investment analysis** for residential properties across major Indian cities. It computes 20-year wealth projections for both buying and renting scenarios, enabling informed decision-making.

### What This Tool Supports

- Objective Buy vs Rent comparison with projected wealth outcomes
- Property-level investment metrics (ROI, rental yield, price efficiency)
- City and location-based market analysis
- Natural language queries via a RAG-powered AI assistant

---

## 2. Key Features

| Feature                 | Description                                                            |
| ----------------------- | ---------------------------------------------------------------------- |
| **Analytics Dashboard** | KPIs, price distributions, and city comparisons via interactive charts |
| **Buy vs Rent Engine**  | 20-year wealth projection comparing ownership vs renting + investing   |
| **Property Browser**    | Filterable listings by city, budget, BHK, and recommendation           |
| **Location Analysis**   | City-wise and locality-level investment insights                       |
| **AI Assistant**        | Natural language interface for querying and explaining results         |

---

## 3. System Architecture

```
┌─────────────┐      ┌─────────────────┐      ┌──────────────────┐
│   Frontend  │ ──── │  Flask Backend  │ ──── │  Static Dataset  │
│  (Browser)  │      │   (Analytics)   │      │     (CSV)        │
└─────────────┘      └────────┬────────┘      └──────────────────┘
                              │
                     ┌────────▼────────┐
                     │  RAG Assistant  │
                     │ (Explains Data) │
                     └─────────────────┘
```

**Key Design Principle:**

- All financial metrics are **precomputed by the backend** during data processing
- The AI assistant **retrieves and explains** these precomputed results
- The AI does **not** perform calculations or generate financial figures

---

## 4. Tech Stack

| Component           | Technology                        | Role                                    |
| ------------------- | --------------------------------- | --------------------------------------- |
| **Backend**         | Python, Flask                     | Web server, API, analytics engine       |
| **Data Processing** | Pandas, NumPy                     | Metric computation, data transformation |
| **AI/RAG**          | LangChain, Google Gemini, FAISS   | Query understanding, semantic retrieval |
| **Frontend**        | Tailwind CSS, Alpine.js, Chart.js | UI, interactivity, visualizations       |
| **Vector Store**    | FAISS, HuggingFace Embeddings     | Similarity search for RAG               |

---

## 5. Investment Logic

### Buy vs Rent Comparison

The system evaluates two parallel scenarios over a 20-year horizon:

**Buying Path:**

- Down payment deployed as property equity
- Monthly EMI payments over loan tenure
- Property value appreciates annually
- Final wealth = Appreciated property value − Total loan cost

**Renting Path:**

- Down payment invested in equity/mutual funds
- Monthly savings (EMI − Rent) invested continuously
- Investments grow at market returns
- Final wealth = Total investment corpus

**Outcome:** The scenario yielding higher terminal wealth determines the recommendation.

### Assumptions Used

| Parameter             | Value                 |
| --------------------- | --------------------- |
| Down Payment          | 20% of property value |
| Loan Interest         | 8.5% per annum        |
| Loan Tenure           | 20 years              |
| Property Appreciation | 5% per annum          |
| Investment Returns    | 10% per annum         |
| Rent Escalation       | 3% per annum          |

_These are heuristic assumptions for comparative analysis, not personalized financial advice._

---

## 6. AI Assistant — Scope & Guardrails

### What the AI Does

- Interprets natural language queries about properties
- Retrieves relevant data from the precomputed dataset
- Explains investment metrics and recommendations in plain language
- Provides city-level and property-level insights

### What the AI Does NOT Do

- Perform financial calculations or projections
- Generate numbers not present in the dataset
- Predict future market movements
- Provide personalized investment advice

### Grounding

All AI responses are grounded in the static dataset. The assistant uses Retrieval-Augmented Generation (RAG) to fetch relevant property records before generating explanations.

---

## 7. Dataset

| Attribute    | Details                                                       |
| ------------ | ------------------------------------------------------------- |
| **Source**   | MagicBricks (web-scraped)                                     |
| **Size**     | 895 property listings                                         |
| **Format**   | Static CSV file                                               |
| **Coverage** | Mumbai, Pune, Delhi NCR, Hyderabad, Bangalore, Kolkata        |
| **Fields**   | Price, Area, BHK, Location, Rental Yield, ROI, Recommendation |

_The dataset represents a point-in-time snapshot and is not live-updated._

---

## 8. Performance & Reliability

- **Precomputed Analytics:** All investment metrics are calculated during data processing, ensuring fast dashboard loads
- **Vector Index:** FAISS index enables sub-second semantic search for AI queries
- **Graceful Degradation:** Missing coordinates or data fields are handled with fallback displays
- **Error Boundaries:** Frontend gracefully handles API failures with user-friendly messages

---

## 9. Limitations

| Limitation         | Impact                                              |
| ------------------ | --------------------------------------------------- |
| Static dataset     | Does not reflect current market prices              |
| Fixed assumptions  | Users cannot customize financial parameters         |
| Single data source | No cross-validation with other platforms            |
| No personalization | Does not consider individual tax brackets or income |
| Limited geography  | Covers 6 major cities only                          |

---

## 10. Future Enhancements

- [ ] Database-backed storage (PostgreSQL/MongoDB)
- [ ] Periodic data refresh pipeline
- [ ] User-configurable financial assumptions
- [ ] Property comparison tool (side-by-side analysis)
- [ ] Expanded city coverage

---

## 11. How to Run

### Prerequisites

- Python 3.10+
- Google Gemini API key

### Installation

```bash
# Clone repository
git clone <repository-url>
cd RealEstateAnalyzer_Project

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure API key
echo GOOGLE_API_KEY=your_key_here > .env

# Run application
python run_app.py
```

### Access

Open **http://localhost:5000** in your browser.

---

## 12. Project Structure

```
├── run_app.py              # Flask application entry point
├── services/analysis.py    # Core analytics and metrics engine
├── src/
│   ├── Parameters/         # Financial calculation modules
│   │   ├── buy_vs_rent.py  # Buy vs rent comparison logic
│   │   ├── loan.py         # EMI and loan calculations
│   │   └── investing.py    # Investment growth projections
│   ├── rag/                # RAG engine components
│   │   ├── rag_engine.py   # Main orchestrator
│   │   ├── vector_store.py # FAISS vector operations
│   │   └── intent_classifier.py
│   └── playwright_scraper/ # Data collection scripts
├── templates/              # Jinja2 HTML templates
├── static/js/              # Frontend JavaScript (charts, chat)
└── data/outputs/           # Analyzed property dataset (CSV)
```

---

## 13. API Reference

| Endpoint      | Method | Description                              |
| ------------- | ------ | ---------------------------------------- |
| `/`           | GET    | Landing page with search                 |
| `/dashboard`  | GET    | Analytics dashboard with charts and KPIs |
| `/properties` | GET    | Property browser with filters            |
| `/api/chat`   | POST   | AI assistant query endpoint              |

---

## 14. Sample AI Queries

The AI assistant can handle queries such as:

- "What are the best properties in Mumbai under 1 crore?"
- "Compare rental yields across different cities"
- "Explain why this property is recommended for buying"
- "Show me 3BHK properties in Pune with high ROI"
- "Which city has the highest average rental yield?"

---

## 15. Acknowledgments

- **MagicBricks** — Property data source
- **Google Gemini** — Large language model for AI assistant
- **LangChain** — RAG orchestration framework
- **FAISS** — Vector similarity search
- **Tailwind CSS** — Frontend styling

---

## 16. Environment Variables

| Variable         | Required | Description                              |
| ---------------- | -------- | ---------------------------------------- |
| `GOOGLE_API_KEY` | Yes      | Google Gemini API key for RAG assistant  |
| `FLASK_DEBUG`    | No       | Enable Flask debug mode (default: False) |
| `PORT`           | No       | Server port (default: 5000)              |

---

## 17. License

This project is for **educational and demonstration purposes** only.  
Not intended for production deployment or financial advisory use.

---

_Built with Python, Flask, and Google Gemini | Real Estate Investment Analyzer_
