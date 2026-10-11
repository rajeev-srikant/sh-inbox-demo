import os
import json
import time
from typing import Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CHAT_MODEL = "gemini-flash-latest"
MAX_EMAILS = 16          # set to 5 if you are still on the free tier (about 20 requests a day)


# ---------- 1. The "form" the AI must fill in (structured output) ----------
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


# ---------- 2. The rules for filling in the form ----------
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


def triage(email):
    text = f"From: {email['from']}\nSubject: {email['subject']}\n\n{email['body']}"
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=text,
        config={
            "system_instruction": TRIAGE_RULES,
            "temperature": 0,                          # same email -> same result
            "response_mime_type": "application/json",  # reply as JSON, not free text
            "response_schema": TicketTriage,           # ...that matches our form exactly
        },
    )
    return response.parsed                             # a filled-in TicketTriage form


# ---------- 3. Run it on the inbox and measure accuracy ----------
with open("emails.json", encoding="utf-8") as f:
    emails = json.load(f)[:MAX_EMAILS]

results = []
category_correct = 0
priority_correct = 0

for email in emails:
    form = triage(email)
    cat_ok = form.category == email["expected_category"]
    pri_ok = form.priority == email["expected_priority"]
    category_correct += cat_ok
    priority_correct += pri_ok
    print(f"{email['id']} | {form.category:<20} {'OK ' if cat_ok else 'X  '}"
          f"| {form.priority:<7} {'OK ' if pri_ok else 'X  '}| {form.team:<14}| {form.summary}")
    results.append({"id": email["id"], **form.model_dump()})
    time.sleep(2)                                      # be gentle with the rate limit

n = len(emails)
print(f"\nCategory correct: {category_correct}/{n}    Priority correct: {priority_correct}/{n}")

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Saved results.json")
