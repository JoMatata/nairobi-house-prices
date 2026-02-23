# Nairobi House Price Predictor

A machine learning project that predicts residential property prices across Nairobi using real listing data scraped from Property24 Kenya.

Built as part of the **LTLab Data Engineering Fellowship 2026** over a 6-day intensive sprint.

---

## The Problem

Nairobi's property market lacks pricing transparency. Landlords set prices by intuition and buyers have no data-backed benchmark to validate whether a price is fair. This tool replaces guesswork with a model trained on real market data.

---

## What Was Built

| Deliverable | Description |
|---|---|
| Web scraper | Selenium + BeautifulSoup pipeline targeting Property24 Kenya |
| Clean dataset | 396 real Nairobi property listings across 15+ neighbourhoods |
| ML model | Random Forest achieving R² = 0.952 and MAE of KES 946,941 |
| Pricing app | Streamlit app — input property details, get an instant price estimate |
| Dashboard | Market intelligence dashboard with location, size and tier analysis |

---

## Model Performance

| Model | MAE (KES) | RMSE (KES) | R² |
|---|---|---|---|
| Linear Regression (Baseline) | 2,285,002 | 3,741,978 | 0.806 |
| **Random Forest (Final)** | **946,941** | **1,869,885** | **0.952** |

The Random Forest model improved prediction error by **58.6%** over the baseline. On average, predictions are off by KES 946,941 against a dataset median price of KES 12.5M — a **7.6% average error rate**.

---

## Key Market Insights

- **Size** is the single strongest price driver at 51.7% of model importance
- **Bathrooms** (23.4%) are a stronger signal than bedrooms alone — Nairobi buyers pay for finishing quality
- **Runda and Kiambu Rd** lead at KES 35M median — 4x the affordable tier
- **Westlands** is the most data-rich location with 233 listings at KES 13M median

---

## Project Structure

```
nairobi-house-prices/
│
├── data/
│   ├── raw_listings.csv          # Original scraped data (420 listings)
│   ├── clean_listings.csv        # Cleaned and feature-engineered data (396 listings)
│   ├── model.pkl                 # Trained Random Forest model
│   ├── encoders.pkl              # Label encoders for categorical features
│   ├── model_metadata.json       # Feature list, locations, model stats
│   └── model_results.csv         # Model comparison table
│
├── notebooks/
│   ├── day1_data_collection.ipynb   # Web scraper
│   ├── day2_cleaning.ipynb          # Data cleaning and feature engineering
│   └── day3_modeling.ipynb          # EDA, baseline model, Random Forest
│
├── docs/
│   ├── data_dictionary.md           # Column definitions
│   ├── eda_visuals.png              # Exploratory analysis charts
│   ├── feature_importance.png       # Model feature importance chart
│   └── baseline_model.png           # Actual vs predicted plot
│
├── app.py                        # Streamlit pricing app
├── dashboard.py                  # Streamlit market dashboard
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/JoMatata/nairobi-house-prices.git
cd nairobi-house-prices
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the pricing app**
```bash
streamlit run app.py
```

**5. Run the market dashboard**
```bash
streamlit run dashboard.py --server.port 8502
```

---

## Data Collection

Data was scraped from [Property24 Kenya](https://www.property24.co.ke) using Selenium WebDriver and BeautifulSoup.

- Selenium automated a real Chrome browser to load JavaScript-rendered listing pages
- BeautifulSoup parsed each listing tile to extract structured fields
- 3-second rate limiting was applied between requests to respect the server
- 20 pages were scraped yielding 420 raw listings, cleaned to 396

**Fields collected:** location, address, property type, bedrooms, bathrooms, parking, size (m² → sqft), price (KES), listing URL, listing date

---

## Feature Engineering

New features created during cleaning:

| Feature | Description |
|---|---|
| `price_per_sqft` | Price normalized by size for fair comparison |
| `location_tier` | High End / Mid Range / Affordable based on neighbourhood |
| `total_rooms` | Bedrooms + bathrooms combined |
| `bed_bath_ratio` | Signals luxury level and property finishing |
| `listing_month` | Extracted from date to capture seasonal trends |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| Selenium | Browser automation for scraping |
| BeautifulSoup | HTML parsing |
| Pandas | Data manipulation |
| Scikit-learn | Machine learning models |
| Matplotlib / Seaborn | Visualizations |
| Streamlit | App and dashboard deployment |
| Git / GitHub | Version control |

---

## Data Notes

Property24 Kenya skews toward premium listings — Westlands alone accounts for 56% of the dataset. The Affordable tier has only 10 listings. This is a known data bias documented honestly here. A future version would supplement with listings from BuyRentKenya and Jumia House to improve tier balance.

---

## Author

**JoMatata** — LTLab Data Engineering Fellowship 2026

Data source: [Property24 Kenya](https://www.property24.co.ke)


