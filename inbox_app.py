import os
import json
from typing import Literal, Optional
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai

load_dotenv()

CHAT_MODEL = "gemini-flash-latest"
TOP_K = 3
PRIORITY_ORDER = {"Urgent": 0, "High": 1, "Medium": 2, "Low": 3}

st.set_page_config(page_title="SH IT Inbox Assistant", page_icon="📥", layout="wide")
st.title("📥 SH IT Inbox Assistant")
st.caption("AI reads incoming IT emails, classifies them, routes them to a team and drafts a reply for a person to approve.")
st.info("Demo version: 16 sample emails with dummy data. Nothing is sent automatically; a person reviews every draft.")

# ---------- Simple password protection (only when a password is set) ----------
APP_PASSWORD = os.getenv("APP_PASSWORD")
if APP_PASSWORD:
    entered = st.text_input("Enter the demo password", type="password")
    if entered != APP_PASSWORD:
        st.stop()


@st.cache_resource
def get_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


client = get_client()


# ======================================================================
# STEP A: TRIAGE - the AI fills in a fixed form (structured output)
# ======================================================================
class TicketTriage(BaseModel):
    category: Literal["Password & Account", "VPN & Remote Access", "Security & Phishing", "Hardware",
                      "Software", "Printer", "Network & Wi-Fi", "Not IT"]
    priority: Literal["Low", "Medium", "High", "Urgent"]
    team: Literal["Service Desk", "Network Team", "Security Team", "Hardware Team", "Apps Team", "None"]
    summary: str = Field(description="One short sentence in English describing the problem")
    employee_id: Optional[str] = Field(None, description="Employee ID if the email mentions one, otherwise null")
    device: Optional[str] = Field(None, description="Device name or ID if mentioned, otherwise null")
    deadline: Optional[str] = Field(None, description="Any time pressure mentioned, e.g. 'client call at 11', otherwise null")
    language: str = Field(description="Language the email is written in")
    needs_human: bool = Field(description="True if a person must act quickly or the request is unclear")
    reason: str = Field(description="One short sentence explaining the chosen priority")


TRIAGE_RULES = """You are the first-line triage assistant for the ABC Corp IT helpdesk.
Read the email and fill in the form. Do not answer the email.

Priority rules:
- Urgent: a security incident or risk to company data (phishing link clicked, password entered on a fake site, lost or stolen device). Needs action within minutes.
- High: the person cannot work at all, or has a deadline today.
- Medium: something is broken but there is a workaround, or it affects a group but is not blocking.
- Low: questions and requests with no deadline (new software, accessories, leaving the company).
- A request with a date in the next few weeks (for example "on Friday" or "next month") is Medium.

Team rules:
- Security Team: phishing, suspicious emails, USB exceptions, security incidents.
- Network Team: VPN, Wi-Fi and connectivity.
- Hardware Team: laptops, monitors, lost or stolen devices, returning equipment.
- Apps Team: software requests and application problems.
- Service Desk: passwords, accounts, MFA, printers and anything else IT.
- None: not an IT request.

If an email contains several problems, choose the category of the most blocking or urgent one and mention the others in the summary.
Always write the summary and reason in English, even if the email is in another language."""


def email_text(email):
    return f"From: {email['from']}\nSubject: {email['subject']}\n\n{email['body']}"


def triage(email):
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=email_text(email),
        config={"system_instruction": TRIAGE_RULES, "temperature": 0,
                "response_mime_type": "application/json", "response_schema": TicketTriage},
    )
    return response.parsed


# ======================================================================
# STEP B: RETRIEVE - reuse the PoC 3 index to find relevant policy text
# ======================================================================
@st.cache_resource
def load_index():
    with open("index.json", encoding="utf-8") as f:
        data = json.load(f)
    vectors = np.array([c["vector"] for c in data["chunks"]], dtype=np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return data, data["chunks"], vectors


index_data, chunks, chunk_vectors = load_index()


def retrieve(query, k=TOP_K):
    result = client.models.embed_content(
        model=index_data["embed_model"],
        contents=[query],
        config={"task_type": "RETRIEVAL_QUERY", "output_dimensionality": index_data["embed_dim"]},
    )
    q = np.array(result.embeddings[0].values, dtype=np.float32)
    q = q / np.linalg.norm(q)
    scores = chunk_vectors @ q
    top = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top]


# ======================================================================
# STEP C: DRAFT - write a reply grounded in the policy text
# ======================================================================
class DraftReply(BaseModel):
    reply: str = Field(description="The draft reply email, in the same language as the original email")
    found_in_policy: bool = Field(description="True only if the policy excerpts fully answer the request")


DRAFT_RULES = """You draft replies for the ABC Corp IT helpdesk. A person will review your draft before it is sent.
Rules:
1. Use ONLY the policy excerpts provided. Never invent steps, phone numbers or links.
2. Reply in the same language as the original email.
3. Be short, friendly and step-by-step. Mention the policy name the steps come from.
4. If the excerpts do not answer the request, say the right team has received the request and will contact them, and set found_in_policy to false.
5. Never say an action has already been done (for example "we have reset your password").
6. Sign off as "ABC Corp IT Helpdesk"."""


def draft_reply(email, form, results):
    excerpts = "\n\n".join(f"[{c['file']}, page {c['page']}]\n{c['text']}" for c, s in results)
    prompt = (f"Original email:\n{email_text(email)}\n\n"
              f"Triage: category={form.category}, priority={form.priority}, team={form.team}\n\n"
              f"Policy excerpts:\n{excerpts}")
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config={"system_instruction": DRAFT_RULES, "temperature": 0.2,
                "response_mime_type": "application/json", "response_schema": DraftReply},
    )
    return response.parsed


# ======================================================================
# STEP D: ROUTE - decide what should happen next (plain Python rules)
# ======================================================================
def decide_action(form, draft):
    if form.category == "Not IT":
        return "Ignore (not IT)"
    if form.priority in ("Urgent", "High") or form.needs_human or not draft.found_in_policy:
        return f"Assign to {form.team}"
    return "Reply ready to approve"


def process(email):
    form = triage(email)                                  # A: triage
    results = retrieve(form.summary)                      # B: retrieve (English summary works for any language)
    draft = draft_reply(email, form, results)             # C: draft
    return {
        "email": email,
        "form": form.model_dump(),
        "draft": draft.model_dump(),
        "action": decide_action(form, draft),             # D: route
        "sources": [{"file": c["file"], "page": c["page"], "score": s} for c, s in results],
    }


# ---------- Load the sample inbox ----------
with open("emails.json", encoding="utf-8") as f:
    inbox = json.load(f)

if "processed" not in st.session_state:
    st.session_state.processed = {}

with st.sidebar:
    st.header("Inbox")
    st.markdown(f"**{len(inbox)} emails** waiting")
    if st.button("Process inbox", type="primary"):
        progress = st.progress(0.0, text="Processing...")
        for i, email in enumerate(inbox, start=1):
            if email["id"] not in st.session_state.processed:
                try:
                    st.session_state.processed[email["id"]] = process(email)
                except Exception as e:
                    st.error(f"{email['id']}: {e}")
            progress.progress(i / len(inbox), text=f"Processed {i} of {len(inbox)}")
    if st.button("Clear results"):
        st.session_state.processed = {}
        st.rerun()
    st.divider()
    st.caption("Each email uses 3 AI calls: triage, search, draft.")

tab_inbox, tab_try = st.tabs(["Processed inbox", "Try your own email"])


def show_detail(item):
    e, form, draft = item["email"], item["form"], item["draft"]
    st.markdown(f"**From:** {e['from']}  \n**Subject:** {e['subject']}")
    st.text(e["body"])
    left, right = st.columns(2)
    with left:
        st.markdown("**AI triage form**")
        st.json(form)
    with right:
        st.markdown(f"**Suggested action:** {item['action']}")
        st.markdown("**Draft reply** (edit before sending)")
        st.text_area("Draft", draft["reply"], height=220, key=f"draft_{e['id']}", label_visibility="collapsed")
        st.caption("Policy used: " + ", ".join(f"{s['file']} p{s['page']} ({s['score']:.2f})" for s in item["sources"]))
        if st.button("Approve and send", key=f"send_{e['id']}"):
            st.success("Demo only: in a real system this would send the reply and create the ticket.")


with tab_inbox:
    items = [st.session_state.processed[e["id"]] for e in inbox if e["id"] in st.session_state.processed]
    if not items:
        st.write("Click **Process inbox** in the sidebar to start.")
    else:
        # Summary numbers
        counts = pd.Series([it["form"]["priority"] for it in items]).value_counts()
        cols = st.columns(4)
        for col, p in zip(cols, ["Urgent", "High", "Medium", "Low"]):
            col.metric(p, int(counts.get(p, 0)))

        # Accuracy against the expected labels (for the demo)
        cat_ok = sum(it["form"]["category"] == it["email"]["expected_category"] for it in items)
        pri_ok = sum(it["form"]["priority"] == it["email"]["expected_priority"] for it in items)
        st.caption(f"Accuracy vs. helpdesk labels: category {cat_ok}/{len(items)}, priority {pri_ok}/{len(items)}")

        # The inbox table, most urgent first
        rows = [{"ID": it["email"]["id"], "Subject": it["email"]["subject"], "Priority": it["form"]["priority"],
                 "Category": it["form"]["category"], "Team": it["form"]["team"], "Action": it["action"],
                 "Summary": it["form"]["summary"]} for it in items]
        df = pd.DataFrame(rows).sort_values("Priority", key=lambda s: s.map(PRIORITY_ORDER))
        st.dataframe(df, hide_index=True, width="stretch")

        st.subheader("Review emails")
        for it in sorted(items, key=lambda it: PRIORITY_ORDER[it["form"]["priority"]]):
            label = f"{it['form']['priority']} · {it['email']['id']} · {it['email']['subject']} → {it['action']}"
            with st.expander(label):
                show_detail(it)

with tab_try:
    st.write("Paste any IT support email to see how it would be handled.")
    sender = st.text_input("From", "someone@abccorp.example")
    subject = st.text_input("Subject")
    body = st.text_area("Email body", height=150)
    if st.button("Process this email") and body.strip():
        with st.spinner("Triage, search and draft..."):
            try:
                item = process({"id": "NEW", "from": sender, "subject": subject, "body": body})
                show_detail(item)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
