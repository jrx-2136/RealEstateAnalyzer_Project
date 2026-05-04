# BluePrint +

**Smarter Real Estate Investment Decisions Through Data**

---

## Project Overview

### The Problem

Real estate is one of the largest financial decisions most people make, yet the Buy vs Rent question is often answered through intuition rather than analysis. Investors lack tools to objectively compare the long-term financial outcomes of purchasing a property versus renting and investing the difference.

### The Solution

**BluePrint+** is a data-driven investment analysis tool for Indian residential real estate. It evaluates properties across major cities, computes 20-year wealth projections for both buying and renting scenarios, and provides clear recommendations based on terminal wealth comparison.

### How It Helps

- Transforms raw property data into actionable investment insights
- Removes emotional bias from the Buy vs Rent decision
- Enables city-level and property-level market comparison
- Provides natural language explanations via an AI assistant

---

## Key Features

| Feature                  | Description                                                                       |
| ------------------------ | --------------------------------------------------------------------------------- |
| **Investment Dashboard** | Interactive charts showing price distributions, city comparisons, and key metrics |
| **Property Listings**    | Filterable property browser by city, budget, BHK, and recommendation type         |
| **Buy vs Rent Analysis** | 20-year wealth projection comparing ownership vs renting + investing              |
| **City Analytics**       | Location-wise investment insights and market trends                               |
| **AI Assistant**         | RAG-powered natural language interface for querying and explaining results        |

---

## Project Structure

```
Blueprint+/
├── run_app.py              # Flask application entry point
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys)
│
├── services/
│   └── analysis.py         # Core analytics and metrics engine
│
├── src/
│   ├── Parameters/         # Financial calculation modules
│   │   ├── buy_vs_rent.py  # Buy vs rent comparison logic
│   │   ├── loan.py         # EMI and loan calculations
│   │   └── investing.py    # Investment projection models
│   │
│   ├── rag/                # RAG assistant components
│   │   ├── rag_engine.py   # Main RAG orchestrator
│   │   ├── vector_store.py # FAISS vector operations
│   │   └── intent_classifier.py
│   │
│   └── playwright_scraper/ # Data collection scripts
│
├── templates/              # Jinja2 HTML templates
├── static/js/              # Frontend JavaScript (charts, chat)
└── data/outputs/           # Analyzed property dataset (CSV)
```

---

## Tech Stack

| Layer               | Technology                        | Purpose                                   |
| ------------------- | --------------------------------- | ----------------------------------------- |
| **Frontend**        | Tailwind CSS, Alpine.js, Chart.js | UI styling, interactivity, visualizations |
| **Backend**         | Python, Flask                     | Web server, API endpoints, routing        |
| **Data Processing** | Pandas, NumPy                     | Metric computation, data transformation   |
| **AI / RAG**        | LangChain, Google Gemini          | Query interpretation, response generation |
| **Vector Store**    | FAISS, HuggingFace Embeddings     | Semantic search for relevant properties   |

---

## Investment Logic

### Buy vs Rent Comparison

The system evaluates two parallel financial scenarios over a 20-year horizon:

**Buying Scenario:**

- Deploy down payment as property equity
- Pay monthly EMI over loan tenure
- Property appreciates annually
- Terminal wealth = Appreciated property value − Total cost paid

**Renting Scenario:**

- Invest down payment in equity/mutual funds
- Invest monthly savings (EMI − Rent) continuously
- Investments grow at market returns
- Terminal wealth = Total investment corpus

**Decision:** The scenario yielding higher terminal wealth determines the recommendation.

### Assumptions

| Parameter             | Value                 |
| --------------------- | --------------------- |
| Down Payment          | 20% of property value |
| Loan Interest Rate    | 8.5% per annum        |
| Loan Tenure           | 20 years              |
| Property Appreciation | 5% per annum          |
| Investment Returns    | 10% per annum         |
| Rent Escalation       | 3% per annum          |

_These are heuristic assumptions for comparative analysis purposes._

---

## AI Assistant — Scope & Guardrails

### What the AI Does

- Interprets natural language queries about properties
- Retrieves relevant records from the precomputed dataset
- Explains investment metrics and recommendations in plain language
- Provides city-level and property-level insights

### What the AI Does NOT Do

- Perform financial calculations or projections
- Generate numbers not present in the dataset
- Predict future market movements
- Provide personalized investment advice

### Grounding

All AI responses are grounded in the static dataset. The assistant uses Retrieval-Augmented Generation (RAG) to fetch relevant property records before generating explanations. No values are hallucinated.

---

## Dataset Description

| Attribute    | Details                                                       |
| ------------ | ------------------------------------------------------------- |
| **Format**   | Static CSV file                                               |
| **Size**     | ~895 property listings                                        |
| **Source**   | MagicBricks (web-scraped)                                     |
| **Coverage** | Mumbai, Pune, Delhi NCR, Hyderabad, Bangalore, Kolkata        |
| **Fields**   | Price, Area, BHK, Location, Rental Yield, ROI, Recommendation |

_The dataset is a point-in-time snapshot and is not live-updated._

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- Google Gemini API key

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/sruthijayesh06/RealEstateAnalyzer_Project.git
cd RealEstateAnalyzer_Project

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
# Create a .env file with your API key:
GOOGLE_API_KEY=your_gemini_api_key_here

# 5. Run the application
python run_app.py
```

Open **http://localhost:5000** in your browser.

---

## Limitations

| Limitation         | Impact                                              |
| ------------------ | --------------------------------------------------- |
| Static dataset     | Does not reflect current market prices              |
| Fixed assumptions  | Financial parameters are not user-configurable      |
| Single data source | No cross-validation with other platforms            |
| No personalization | Does not consider individual tax brackets or income |
| Limited geography  | Covers 6 major Indian cities only                   |

---

## Future Enhancements

- [ ] Database-backed storage (PostgreSQL/MongoDB)
- [ ] Interactive map integration with property markers
- [ ] Property scoring and ranking system
- [ ] User-configurable financial assumptions
- [ ] Periodic data refresh pipeline

---

## License

This project is for **educational and demonstration purposes** only.

---

## Credits

- **Data Source:** MagicBricks
- **LLM:** Google Gemini
- **RAG Framework:** LangChain
- **Vector Search:** FAISS

---

_Blueprint+ — Data-driven decisions for smarter real estate investment._
