import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Prediction App")
st.write("Enter the details below to predict the house price:")

# Input fields
area = st.number_input("Area (in sqft)", min_value=500, max_value=10000, step=50)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms]
    })
    prediction = model.predict(input_data)
    st.success(f"Estimated Price: ₹{prediction[0]:,.2f}")
# ====================================
# ChatGPT for Data Q&A — Task 3 (Growfinix)
# ====================================
# Run with:  streamlit run data_gpt_app.py
# ====================================

import os
import io
import json
import pandas as pd
import streamlit as st
from openai import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------
# UI: Title & Sidebar
# -----------------------------
st.set_page_config(page_title="ChatGPT for Data Q&A", page_icon="📊", layout="wide")
st.title("📊 ChatGPT for Data Q&A (Task 3)")
st.caption("Upload a CSV, ask questions (predefined or custom), and export a PDF report.")

with st.sidebar:
    st.header("⚙️ Settings")
    # Prefer Streamlit secrets, else env var, else user input
    api_default = os.environ.get("OPENAI_API_KEY", "")
    api_key = st.text_input(
        "OpenAI API Key",
        value=st.secrets.get("OPENAI_API_KEY", api_default),
        type="password",
        help="Key is kept in memory only for this session."
    )
    model = st.selectbox("Model", ["gpt-4o-mini"], index=0)
    max_cells = st.number_input(
        "Max cells to send to GPT (rows×cols)", min_value=200, max_value=20000, value=1200, step=200
    )
    st.markdown("---")
    st.write("📎 **Upload CSV** below on the main page.")

if not api_key:
    st.warning("Enter your OpenAI API key in the sidebar to continue.")
    st.stop()

# -----------------------------
# Helper functions
# -----------------------------
def _extract_json(text: str):
    """Try to extract a JSON object from a model reply. Falls back to raw text."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {
            "answer": text.strip(),
            "reasoning": "Model returned non-JSON. Parsed as raw text.",
            "assumptions": [],
            "confidence": "low",
            "suggested_followups": []
        }

def ask_openai(question: str, context: str, model_name: str):
    client = OpenAI(api_key=api_key)
    system = (
        "You are a careful data analyst. "
        "You are given dataset context and must respond in valid JSON with keys: "
        '["answer","reasoning","assumptions","confidence","suggested_followups"].'
    )
    user = f"""DATA CONTEXT:
{context}

QUESTION:
{question}

Return valid JSON only.
"""
    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.2,
    )
    return _extract_json(resp.output_text)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df

def build_context(df: pd.DataFrame, max_cells_threshold: int) -> str:
    """
    Requirement 9: handle large data by summarizing first.
    For small data (rows*cols <= threshold) send full table (string).
    Else, send compact context: shape, columns, dtypes, describe, head.
    """
    rows, cols = df.shape
    size = rows * cols

    basic = []
    basic.append(f"SHAPE: {rows} rows x {cols} columns")
    basic.append("COLUMNS: " + ", ".join([str(c) for c in df.columns.tolist()]))
    basic.append("DTYPES:\n" + df.dtypes.astype(str).to_string())

    if size <= max_cells_threshold:
        basic.append("\nFULL_TABLE (small dataset):\n" + df.to_string(index=False))
    else:
        basic.append("\nSUMMARY (large dataset):")
        try:
            basic.append("DESCRIBE:\n" + df.describe(include="all", datetime_is_numeric=True).fillna("").to_string())
        except Exception:
            basic.append("DESCRIBE: (skipped due to mixed types)")
        basic.append("\nHEAD(20):\n" + df.head(20).to_string(index=False))

    return "\n\n".join(basic)

def add_to_log(question: str, result: dict):
    st.session_state.qna_log.append({
        "question": question,
        "answer": result.get("answer", ""),
        "reasoning": result.get("reasoning", ""),
        "assumptions": json.dumps(result.get("assumptions", []), ensure_ascii=False),
        "confidence": result.get("confidence", ""),
        "suggested_followups": json.dumps(result.get("suggested_followups", []), ensure_ascii=False)
    })

def ensure_log():
    if "qna_log" not in st.session_state:
        st.session_state.qna_log = []

# -----------------------------
# Upload CSV (Req. 1)
# -----------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)
df = clean_data(df)

# -----------------------------
# Basic info (Req. 2)
# -----------------------------
st.success("✅ File uploaded successfully!")
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    st.subheader("Sample Data")
    st.dataframe(df.head(15), use_container_width=True)
with c2:
    st.subheader("Shape")
    st.write(df.shape)
with c3:
    st.subheader("Columns")
    st.write(list(df.columns))

with st.expander("Show dtypes"):
    st.write(df.dtypes.astype(str))

# -----------------------------
# Build context (Req. 3 & 9)
# -----------------------------
context = build_context(df, max_cells_threshold=max_cells)

# -----------------------------
# Predefined & Custom Questions (Req. 4, 5, 6, 7)
# -----------------------------
ensure_log()

st.subheader("🔹 Predefined Questions")
predefined = [
    "Which product sold most?",
    "Top student in Math?",
    "What are the key trends in this dataset?",
    "What improvement strategies would you suggest?"
]
cols = st.columns(len(predefined))
for i, q in enumerate(predefined):
    if cols[i].button(q):
        with st.spinner("Asking GPT..."):
            result = ask_openai(q, context, model)
        add_to_log(q, result)
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {result.get('answer','')}")
        with st.expander("Details (reasoning, assumptions, follow-ups)"):
            st.json(result)

st.subheader("💬 Ask a Custom Question")
custom_q = st.text_input("Type your question about the uploaded dataset")
if st.button("Ask"):
    if not custom_q.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Asking GPT..."):
            result = ask_openai(custom_q.strip(), context, model)
        add_to_log(custom_q.strip(), result)
        st.markdown(f"**Q:** {custom_q}")
        st.markdown(f"**A:** {result.get('answer','')}")
        with st.expander("Details (reasoning, assumptions, follow-ups)"):
            st.json(result)

# -----------------------------
# Log Q&A to CSV (Req. 8)
# -----------------------------
st.subheader("🗂️ Logs")
if st.session_state.qna_log:
    log_df = pd.DataFrame(st.session_state.qna_log)
    st.dataframe(log_df, use_container_width=True)
    # Save server-side
    log_path = "analysis_results.csv"
    log_df.to_csv(log_path, index=False)
    # Also offer download
    st.download_button("⬇️ Download Q&A Log (CSV)", data=log_df.to_csv(index=False), file_name="analysis_results.csv", mime="text/csv")
else:
    st.info("No Q&A yet — ask a predefined or custom question to generate logs.")

# -----------------------------
# Export Q&A as PDF (Req. 10)
# -----------------------------
st.subheader("📄 Export Q&A as PDF")
if st.button("Generate PDF"):
    if not st.session_state.qna_log:
        st.warning("No Q&A to export yet.")
    else:
        styles = getSampleStyleSheet()
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf)
        content = [Paragraph("Data Analysis Q&A Report", styles["Title"]), Spacer(1, 12)]

        # Dataset summary (concise)
        content.append(Paragraph("Dataset Summary", styles["Heading2"]))
        summary_text = f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}"
        content.append(Paragraph(summary_text, styles["Normal"]))
        content.append(Paragraph("Columns: " + ", ".join(map(str, df.columns.tolist())), styles["Normal"]))
        content.append(Spacer(1, 8))
        # Add head as monospaced block
        head_str = df.head(15).to_string(index=False)
        content.append(Preformatted(head_str, styles["Code"]))
        content.append(Spacer(1, 12))

        # Q&A
        content.append(Paragraph("AI Analysis Results", styles["Heading2"]))
        for item in st.session_state.qna_log:
            content.append(Paragraph(f"Q: {item['question']}", styles["Normal"]))
            content.append(Paragraph(f"A: {item['answer']}", styles["Normal"]))
            if item.get("reasoning"):
                content.append(Preformatted("Reasoning:\n" + item["reasoning"], styles["Code"]))
            if item.get("assumptions"):
                content.append(Preformatted("Assumptions:\n" + item["assumptions"], styles["Code"]))
            if item.get("suggested_followups"):
                content.append(Preformatted("Follow-ups:\n" + item["suggested_followups"], styles["Code"]))
            content.append(Spacer(1, 8))

        doc.build(content)
        pdf_bytes = buf.getvalue()
        buf.close()
        st.success("PDF generated.")
        st.download_button("⬇️ Download PDF report", data=pdf_bytes, file_name="analysis_report.pdf", mime="application/pdf")
