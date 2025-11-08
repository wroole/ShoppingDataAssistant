# Shopping Data Assistant

**Shopping Data Assistant** is a Python-based chatbot that allows users to interact with shopping datasets using natural language.  
It merges, cleans, and analyzes multiple CSV files, providing **data-driven answers** and **visual insights** on demand.

---

## Features

- **Chatbot Interface:** Ask questions about the sales and products data in plain English.
- **Data Visualization:** Generate charts and graphs based on your queries.
- **Data Processing:** Automatically merges and cleans multiple shopping-related CSV files.
- **SQLite Database:** Stores cleaned and merged data for fast querying.
- **Custom Prompts:** Uses prompt templates for smart and context-aware answers.

---

## Project Structure
```├── data/                  # Raw and processed datasets
├── prompts/               # Prompt templates for chatbot logic
├── static/                # Static files (e.g. images, css)
├── .env                   # Environment variables (e.g. API keys)
├── .gitignore             # Ignored files configuration
├── app.py                 # Main chatbot application
├── data_test.ipynb        # Data testing and exploration notebook
├── pandas.ipynb           # Data cleaning and preprocessing notebook
└── sales.db               # SQLite database with merged sales data
```


---

## Tech Stack

- **FastAPI** – backend framework  
- **OpenAI API** – natural language understanding  
- **SQLite3** – database engine  
- **Pandas** – data analysis  
- **Matplotlib + Seaborn** – data visualization  
- **Pydantic** – data validation  
- **dotenv** – environment variable management  

