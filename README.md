
A **Streamlit** application that extracts medical concepts (diagnoses, medications, dates, person names, organisations) from clinical notes using a hybrid approach:  
- **spaCy**’s lightweight model (`en_core_web_sm`) for general named entities (PERSON, DATE, ORG, …).  
- A **built‑in dictionary** of common drugs and diseases for medical‑specific extraction.  
- **Negation detection** (via `negspacy`) to mark negated concepts (e.g., “no fever”).

## 🚀 Features
- **Patient search** by name or ID.
- **Add clinical notes** (Progress, Discharge, Consultation, Other).
- **Automatic concept extraction** – view extracted entities grouped by type.
- **Overview dashboard** showing active problems and current medications.
- **Manage patients and doctors** (register new entries).
- **Lab results** and **alerts** (acknowledgement system).

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/shifaah-nlp-app.git
   cd shifaah-nlp-app
