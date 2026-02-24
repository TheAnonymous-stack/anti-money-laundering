# Explainable Money Laundering Bad Actor Detection

Scotiabank UofT IMI BIGDataAIHUB Competition 2025-2026

## Setup

Requires Python 3.10+. XGBoost needs the OpenMP runtime (`libomp`) on macOS.

### Option A: Conda (Recommended — handles libomp automatically)

If you don't have Conda, install [Miniconda](https://docs.anaconda.com/miniconda/) first (lightweight, ~80 MB).

```bash
conda env create -f environment.yml
conda activate imi-aml
```

### Option B: pip (no Conda required)

```bash
# macOS only — install OpenMP runtime for XGBoost:
brew install libomp

pip install -r requirements.txt
```

> **Note:** On Linux, `libomp` is typically pre-installed. On Windows, no extra step is needed.

## Run

Place all CSV data files in `data/` and run:

```bash
python run_pipeline.py
```

**Outputs** (saved to `outputs/`):
- `model_output.csv` — customer_id, predicted_label (0/1), risk_score (0-1) for all customers
- `model_output_explanations.csv` — customer_id, explanation (max 2,000 chars) for all customers

## Project Structure

```
IMI/
├── data/                           # Raw CSV data files
├── knowledge_library/
│   └── aml_red_flags.json          # 30 AML red flags with source citations
├── src/
│   ├── data_loader.py              # Centralized CSV loading with chunked reading
│   ├── feature_engineering.py      # Customer-level feature computation (~229 features)
│   ├── aml_rules.py                # 15 domain-driven AML detection rules
│   ├── knowledge_library.py        # Query interface for red flag database
│   ├── explanations.py             # SHAP-based explanation generator
│   └── models/
│       ├── anomaly_detectors.py    # Isolation Forest, ECOD, LOF, kNN
│       ├── supervised.py           # XGBoost + LightGBM with PU Learning
│       └── ensemble.py             # Weighted ensemble and validation
├── outputs/                        # Generated output CSVs
├── run_pipeline.py                 # End-to-end execution script
├── environment.yml                 # Conda environment definition
├── requirements.txt                # pip dependencies
└── README.md
```

## Model Design

### Challenge

61,410 customers (53K individuals + 8K small businesses), ~5.9M transactions across 7 channels, but only **10 positive labels** out of 1,000 labeled customers. The extreme label scarcity demands a hybrid approach.

### 4-Stage Ensemble

| Stage | Method | Weight | Purpose |
|-------|--------|--------|---------|
| 1. Unsupervised | Isolation Forest, ECOD, LOF, kNN | 20% | Detect statistical anomalies without labels |
| 2. Domain Rules | 15 AML rules from Knowledge Library | 15% | Encode FINTRAC/FATF domain knowledge |
| 3. Supervised | XGBoost + LightGBM (avg) | 50% | Learn from the 10 labeled positives |
| 4. PU Learning | Retrained XGBoost + LightGBM | 15% | Expand training set with reliable negatives |

**Final score**: `risk_score = 0.20 * unsupervised + 0.15 * rules + 0.50 * supervised + 0.15 * pu`

Customers above the 97th percentile threshold receive `predicted_label = 1`.

### Key Design Decisions

- **Shallow trees** (`max_depth=4`): prevents overfitting on only 10 positives
- **`scale_pos_weight=99`**: compensates for 990:10 class imbalance
- **Chunked processing**: card.csv (3.55M rows) is processed in 500K-row chunks with exact recombination of additive statistics
- **Sentinel values** (`-1`): individual-specific features are set to -1 for businesses and vice versa, allowing tree models to learn type-specific splits

## Feature Engineering

229 features organized into 4 categories:

### KYC Static Features
- **Individual**: age, is_minor, is_young_adult, tenure, occupation flags (student, unemployed, retired, self-employed), income, gender, marital status, province frequency
- **Business**: business age, is_newly_established, high-risk industry flags (holding, real estate, legal, financial, MSB), employee count, sales, sales_per_employee

### Per-Channel Transaction Aggregations (7 channels x ~25 features each)
- Volume: transaction count, credit/debit counts, credit ratio
- Amounts: total, mean, std, max, min, median, net flow
- AML signals: round-amount ratio, near-$10K-threshold count
- Temporal: active days, weekend ratio, night ratio, burst score
- Channel-specific: cash indicator (ABM), foreign ratio (ABM/card), e-commerce ratio (card), unique MCC count (card)

### Cross-Channel Behavioral Features
- num_active_channels, channel concentration (HHI)
- Total transaction count, amount, and net flow across all channels
- **transaction_to_income_ratio**: total activity relative to declared income
- **net_flow_to_income_ratio**: absolute net flow relative to income
- Overall credit ratio, structuring scores

## Feature-to-AML-Indicator Mapping

| Feature | AML Indicator | Source |
|---------|--------------|--------|
| `transaction_to_income_ratio` | Activity inconsistent with financial profile | FINTRAC ML/TF Indicators, Category 3 |
| `emt_credit_count` (high, for students) | Money mule receiving funds from multiple sources | FINTRAC Project Athena (2023) |
| `total_near_10k` | Structuring below $10K reporting threshold | FINTRAC Category 7; PCMLTFA |
| `industry_is_holding` + `zero_sales` | Shell company with no economic purpose | FATF Concealment of Beneficial Ownership (2018) |
| `is_newly_established` + high volume | New entity with immediate suspicious activity | FINTRAC Category 5 |
| `channel_hhi` (low) + many channels | Layering across channels to obscure trail | FINTRAC Category 6 |
| `abm_cash_count` / `abm_cash_ratio` | Cash-intensive activity at ABMs | FINTRAC Operational Alert (2018) |
| `wire_total_credit_amount` (high) | Large international fund transfers | FINTRAC Category 8 |
| `is_minor` + large transfers | Minor used as money mule | FINTRAC Project Athena (2023) |
| `abm_foreign_ratio` (high, for students) | Foreign ABM usage inconsistent with profile | FINTRAC Category 9 |

## AML Knowledge Library

30 red flags across 8 typologies in `knowledge_library/aml_red_flags.json`:

- **Money Mule**: students/young people receiving frequent transfers
- **Structuring**: splitting transactions to avoid $10K threshold
- **Shell Company**: holding companies with zero sales/employees
- **Layering**: rapid fund movements across multiple channels
- **Income Mismatch**: activity grossly exceeding declared income
- **Cash-Intensive**: frequent large ABM cash transactions
- **High-Risk Industry**: holding, real estate, legal, financial, MSB
- **Organized Crime**: combined indicators from CISC/RCMP typologies

Sources: FINTRAC operational alerts and guidance, FATF typology reports, Cullen Commission final report, CISC 2024 organized crime assessment, Canada's 2025 National Risk Assessment, RCMP case analysis.

## Explanation Strategy

Every customer receives a plain-English explanation combining:
1. **Triggered AML rules** with human-readable descriptions citing FINTRAC/FATF
2. **Top 5 SHAP features** from the XGBoost model translated to readable phrases
3. **Risk-tier templates**: high risk (>= 0.5) gets detailed evidence, moderate (>= 0.2) gets a summary, low (< 0.2) gets an all-clear statement

All explanations are capped at 2,000 characters per the competition requirement.

## Evaluation

### Validation Results
- All 10 known positives flagged with `predicted_label = 1`
- All 10 known positives rank in the **top 40** by risk_score
- 3.0% flagging rate (~1,843 customers flagged)
- Leave-One-Positive-Out (LOPO) validation ensures each positive is detectable when held out

### Metrics
- **Recall@k**: all 10 positives recovered in top 40 out of 61,410
- **Precision at threshold**: ~0.5% (expected given extreme class imbalance and unlabeled positives in the test set)
- **Rule coverage**: 7/10 positives trigger the income mismatch rule; 4/10 trigger high-risk industry; 4/10 trigger layering
