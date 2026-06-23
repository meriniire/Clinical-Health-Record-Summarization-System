import streamlit as st
import pandas as pd
from datetime import datetime
import os
import spacy
from negspacy.negation import Negex

st.set_page_config(page_title="General Hospital Koton Karfe, Kogi State", layout="wide")
DATA_DIR = "data"

# ------------------------------------------------------------------
# 1. Load only the simplified model (en_core_web_sm)
# ------------------------------------------------------------------
@st.cache_resource
def load_nlp_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("negex", config={"ent_types": None})  # all entities
        st.success(" NLP model loaded (en_core_web_sm).")
        return nlp
    except OSError:
        st.error("❌ Model not found. Install with:\n`python -m spacy download en_core_web_sm`")
        return None

nlp = load_nlp_model()

# ------------------------------------------------------------------
# 2. Dictionary for medical concepts (drugs and diseases)
# ------------------------------------------------------------------
DRUGS = [
    "paracetamol", "aspirin", "ibuprofen", "furosemide", "lisinopril",
    "clopidogrel", "sumatriptan", "amoxicillin", "metformin", "atenolol",
    "omeprazole", "atorvastatin", "losartan", "morphine", "diazepam"
]
DISEASES = [
    "headache", "diabetes", "hypertension", "migraine", "heart failure",
    "stroke", "pneumonia", "asthma", "cancer", "malaria", "tuberculosis",
    "depression", "anxiety", "obesity", "pneumonia", "bronchitis", "arthritis"
]

# ------------------------------------------------------------------
# 3. Data loading & helpers
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    patients = pd.read_csv(os.path.join(DATA_DIR, "patients.csv"))
    notes = pd.read_csv(os.path.join(DATA_DIR, "clinical_notes.csv"))
    labs = pd.read_csv(os.path.join(DATA_DIR, "lab_results.csv"))
    meds = pd.read_csv(os.path.join(DATA_DIR, "medications.csv"))
    concepts = pd.read_csv(os.path.join(DATA_DIR, "extracted_concepts.csv")) if os.path.exists(os.path.join(DATA_DIR, "extracted_concepts.csv")) else pd.DataFrame(columns=["concept_id", "note_id", "concept_type", "concept_value", "confidence", "negation_flag"])
    alerts = pd.read_csv(os.path.join(DATA_DIR, "alerts.csv"))
    doctors = pd.read_csv(os.path.join(DATA_DIR, "doctors.csv")) if os.path.exists(os.path.join(DATA_DIR, "doctors.csv")) else pd.DataFrame(columns=["doctor_id", "title", "first_name", "last_name", "specialization", "phone", "email"])
    return patients, notes, labs, meds, concepts, alerts, doctors

patients, notes, labs, meds, concepts, alerts, doctors = load_data()

def save_dataframe(df, filename):
    df.to_csv(os.path.join(DATA_DIR, filename), index=False)

def get_notes(patient_id):
    return notes[notes["patient_id"] == patient_id].sort_values("note_date", ascending=False)

def get_labs(patient_id):
    return labs[labs["patient_id"] == patient_id].sort_values("test_date", ascending=False)

def get_meds(patient_id):
    return meds[meds["patient_id"] == patient_id]

def get_concepts(patient_id, concept_type=None):
    patient_note_ids = notes[notes["patient_id"] == patient_id]["note_id"].tolist()
    patient_concepts = concepts[concepts["note_id"].isin(patient_note_ids)]
    if concept_type:
        patient_concepts = patient_concepts[patient_concepts["concept_type"] == concept_type]
    return patient_concepts

def get_alerts(patient_id, unacknowledged_only=True):
    patient_alerts = alerts[alerts["patient_id"] == patient_id]
    if unacknowledged_only:
        patient_alerts = patient_alerts[patient_alerts["acknowledged"] == False]
    return patient_alerts

def acknowledge_alert(alert_id):
    global alerts
    alerts.loc[alerts["alert_id"] == alert_id, "acknowledged"] = True
    alerts.loc[alerts["alert_id"] == alert_id, "acknowledged_by"] = st.session_state.get("user", "Dr. Default")
    alerts.loc[alerts["alert_id"] == alert_id, "acknowledged_date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_dataframe(alerts, "alerts.csv")
    st.cache_data.clear()
    st.rerun()

# ------------------------------------------------------------------
# 4. Extraction: spaCy entities + rule-based medical terms
# ------------------------------------------------------------------
def extract_clinical_concepts(note_text, note_id):
    if nlp is None:
        return []
    doc = nlp(note_text)
    new_concepts = []
    concepts_path = os.path.join(DATA_DIR, "extracted_concepts.csv")
    if os.path.exists(concepts_path) and os.path.getsize(concepts_path) > 0:
        existing = pd.read_csv(concepts_path)
        next_id = existing["concept_id"].max() + 1 if not existing.empty else 1
    else:
        next_id = 1

    # 4a. Extract all spaCy named entities (PERSON, DATE, ORG, etc.)
    for ent in doc.ents:
        negated = ent._.negex if hasattr(ent._, "negex") else False
        # Keep the original label (e.g., "PERSON", "DATE")
        concept_type = ent.label_
        new_concepts.append({
            "concept_id": next_id,
            "note_id": note_id,
            "concept_type": concept_type,
            "concept_value": ent.text,
            "confidence": 0.8,
            "negation_flag": negated
        })
        next_id += 1

    # 4b. Rule‑based detection of drugs and diseases
    note_lower = note_text.lower()
    # Helper to check negation within 3 words before a term
    def is_negated(term, text):
        # Find the position of the term in the text
        idx = text.find(term)
        if idx == -1:
            return False
        # Get the preceding 30 characters (approx 3 words)
        start = max(0, idx - 30)
        preceding = text[start:idx]
        # Look for negation markers
        neg_words = ["no", "not", "denies", "without", "never"]
        for word in neg_words:
            if word in preceding.split():
                return True
        return False

    # Drugs
    for drug in DRUGS:
        if drug in note_lower:
            negated = is_negated(drug, note_lower)
            new_concepts.append({
                "concept_id": next_id,
                "note_id": note_id,
                "concept_type": "Medication",
                "concept_value": drug.capitalize(),
                "confidence": 0.9,
                "negation_flag": negated
            })
            next_id += 1

    # Diseases
    for disease in DISEASES:
        if disease in note_lower:
            negated = is_negated(disease, note_lower)
            new_concepts.append({
                "concept_id": next_id,
                "note_id": note_id,
                "concept_type": "Diagnosis",
                "concept_value": disease.capitalize(),
                "confidence": 0.9,
                "negation_flag": negated
            })
            next_id += 1

    # 4c. Remove duplicates (keep first occurrence)
    seen = set()
    unique = []
    for c in new_concepts:
        key = (c["concept_type"], c["concept_value"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique

# ------------------------------------------------------------------
# 5. Add clinical note
# ------------------------------------------------------------------
def add_clinical_note(patient_id, note_type, note_text, author_id):
    global notes, concepts
    new_note_id = notes["note_id"].max() + 1 if not notes.empty else 1
    doctor_row = doctors[doctors["doctor_id"] == author_id].iloc[0]
    author_name = f"{doctor_row['title']} {doctor_row['last_name']}".strip()
    new_note = pd.DataFrame([{"note_id": new_note_id, "patient_id": patient_id, "note_type": note_type, "note_date": datetime.now().strftime("%Y-%m-%d %H:%M"), "author": author_name, "author_id": author_id, "department": doctor_row["specialization"], "note_text": note_text}])
    notes = pd.concat([notes, new_note], ignore_index=True)
    save_dataframe(notes, "clinical_notes.csv")
    extracted = extract_clinical_concepts(note_text, new_note_id)
    if extracted:
        new_concepts_df = pd.DataFrame(extracted)
        concepts_path = os.path.join(DATA_DIR, "extracted_concepts.csv")
        if os.path.exists(concepts_path) and os.path.getsize(concepts_path) > 0:
            existing = pd.read_csv(concepts_path)
            updated = pd.concat([existing, new_concepts_df], ignore_index=True)
        else:
            updated = new_concepts_df
        updated.to_csv(concepts_path, index=False)
        concepts = updated
        st.success(f"✅ Added {len(extracted)} concepts.")
    else:
        st.info("No concepts extracted.")
    st.cache_data.clear()
    st.success("Note added.")

# ------------------------------------------------------------------
# 6. Session state
# ------------------------------------------------------------------
if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "user" not in st.session_state:
    st.session_state.user = "Dr. Okafor"

st.sidebar.title("General Hospital Koton-Karfe, Kogi State ")
menu = st.sidebar.radio("Menu", ["Patient Search", "Add Clinical Note", "Register New Patient", "Manage Patients", "Manage Doctors", "About"])

# ------------------------------------------------------------------
# 7. Patient Search
# ------------------------------------------------------------------
if menu == "Patient Search":
    if st.session_state.selected_patient:
        p = st.session_state.selected_patient
        if st.button("← Back"):
            st.session_state.selected_patient = None
            st.rerun()
        st.header(f"{p['patient_name']} ({p['gender']}, {p['date_of_birth']})")
        # Alerts
        for _, alert in get_alerts(p["patient_id"]).iterrows():
            st.warning(f"{alert['alert_type']}: {alert['alert_message']}")
            if st.button("Acknowledge", key=f"ack_{alert['alert_id']}"):
                acknowledge_alert(alert["alert_id"])
        tabs = st.tabs(["Overview", "Concepts", "Labs", "Notes"])
        with tabs[0]:
            st.subheader("Overview")
            for _, note in get_notes(p["patient_id"]).head(2).iterrows():
                st.markdown(f"**{note['note_type']}** {note['note_date']}  \n{note['note_text'][:200]}...")
            # Show active problems and current medications
            diag = get_concepts(p["patient_id"], "Diagnosis")
            diag = diag[diag["negation_flag"] == False]
            meds_concepts = get_concepts(p["patient_id"], "Medication")
            meds_concepts = meds_concepts[meds_concepts["negation_flag"] == False]
            if not diag.empty:
                st.markdown("**Active Problems:** " + ", ".join(diag["concept_value"]))
            if not meds_concepts.empty:
                st.markdown("**Current Medications:** " + ", ".join(meds_concepts["concept_value"]))
        with tabs[1]:
            st.subheader("All Extracted Concepts")
            all_concepts = get_concepts(p["patient_id"])
            all_concepts = all_concepts[all_concepts["negation_flag"] == False]
            if all_concepts.empty:
                st.info("No concepts found.")
            else:
                # Group by type
                for ctype in all_concepts["concept_type"].unique():
                    st.markdown(f"**{ctype}**")
                    for _, row in all_concepts[all_concepts["concept_type"] == ctype].iterrows():
                        st.write(f"- {row['concept_value']}")
        with tabs[2]:
            st.subheader("Labs")
            lab_list = get_labs(p["patient_id"])
            if not lab_list.empty:
                st.dataframe(lab_list[["test_name", "test_value", "reference_range", "test_date"]])
        with tabs[3]:
            st.subheader("Notes")
            for _, note in get_notes(p["patient_id"]).iterrows():
                with st.expander(f"{note['note_type']} - {note['note_date']}"):
                    st.write(note['note_text'])
    else:
        st.header("Search Patient")
        search = st.text_input("ID or Name")
        if st.button("Search"):
            mask = (patients["patient_id"].astype(str).str.contains(search)) | (patients["patient_name"].str.lower().str.contains(search.lower()))
            st.session_state.search_results = patients[mask]
        if st.session_state.search_results is not None and not st.session_state.search_results.empty:
            for _, row in st.session_state.search_results.iterrows():
                if st.button(f"{row['patient_name']} (ID: {row['patient_id']})"):
                    st.session_state.selected_patient = row.to_dict()
                    st.rerun()

# ------------------------------------------------------------------
# 8. Add Clinical Note
# ------------------------------------------------------------------
elif menu == "Add Clinical Note":
    st.header("Add Note")
    if patients.empty or doctors.empty:
        st.warning("Register a patient and doctor first.")
    else:
        patient_options = patients.apply(lambda x: f"{x['patient_name']} (ID: {x['patient_id']})", axis=1).tolist()
        selected_patient_str = st.selectbox("Patient", patient_options)
        pid = int(selected_patient_str.split("ID: ")[1].replace(")", ""))
        
        doctor_options = doctors.apply(lambda x: f"{x['title']} {x['last_name']} (ID: {x['doctor_id']})", axis=1).tolist()
        selected_doc_str = st.selectbox("Doctor", doctor_options)
        did = int(selected_doc_str.split("ID: ")[1].replace(")", ""))
        
        note_type = st.selectbox("Type", ["Progress", "Discharge", "Consultation", "Other"])
        note_text = st.text_area("Note")
        if st.button("Add") and note_text:
            add_clinical_note(pid, note_type, note_text, did)
            st.rerun()

# ------------------------------------------------------------------
# 9. Register Patient
# ------------------------------------------------------------------
elif menu == "Register New Patient":
    st.header("Register Patient")
    with st.form("reg"):
        name = st.text_input("Full Name*")
        dob = st.date_input("DOB*")
        gender = st.selectbox("Gender", ["M", "F", "Other"])
        contact = st.text_input("Contact")
        if st.form_submit_button("Register") and name and dob:
            df = pd.read_csv(os.path.join(DATA_DIR, "patients.csv"))
            new_id = df["patient_id"].max() + 1 if not df.empty else 1
            new = pd.DataFrame([{"patient_id": new_id, "hospital_number": f"SH-{new_id:03d}", "patient_name": name, "date_of_birth": dob.strftime("%Y-%m-%d"), "gender": gender, "contact_info": contact, "registration_date": datetime.now().strftime("%Y-%m-%d")}])
            updated_df = pd.concat([df, new], ignore_index=True)
            updated_df.to_csv(os.path.join(DATA_DIR, "patients.csv"), index=False)
            patients = updated_df
            st.cache_data.clear()
            st.success("Registered")

# ------------------------------------------------------------------
# 10. Manage Patients
# ------------------------------------------------------------------
elif menu == "Manage Patients":
    st.header("Manage Patients")
    if not patients.empty:
        st.dataframe(patients[["patient_id", "patient_name", "gender", "date_of_birth"]])

# ------------------------------------------------------------------
# 11. Manage Doctors
# ------------------------------------------------------------------
elif menu == "Manage Doctors":
    st.header("Manage Doctors")
    if not doctors.empty:
        st.dataframe(doctors[["doctor_id", "title", "first_name", "last_name", "specialization"]])
    with st.form("add_doc"):
        title = st.selectbox("Title", ["Dr.", "Prof.", "Mr.", "Ms.", "Mrs."])
        first = st.text_input("First Name")
        last = st.text_input("Last Name")
        spec = st.text_input("Specialization", "General Medicine")
        if st.form_submit_button("Add") and first and last:
            new_id = doctors["doctor_id"].max() + 1 if not doctors.empty else 1
            new = pd.DataFrame([{"doctor_id": new_id, "title": title, "first_name": first, "last_name": last, "specialization": spec, "phone": "", "email": ""}])
            doctors = pd.concat([doctors, new], ignore_index=True)
            save_dataframe(doctors, "doctors.csv")
            st.cache_data.clear()
            st.rerun()

else:  # About
    st.info("Hybrid concept extractor: spaCy en_core_web_sm + built‑in drug/disease dictionary.")