import os
import json
import time
import docx
from google import genai
from google.genai import types
from google.genai.errors import ServerError

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 1. Load your original, untagged resume
doc = docx.Document("resume.docx")

# Extract non-empty paragraphs
raw_paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
full_resume_text = "\n".join(raw_paragraphs)

job_description = """PASTE_JOB_DESCRIPTION_HERE"""

SYSTEM_INSTRUCTION = """
You are an expert ATS resume writer. 
You will receive the full raw text of a user's resume and a target Job Description.
Your task:
1. Return a JSON array of strings called 'tailored_paragraphs'. This array MUST match the exact number and sequence of text sections/bullets provided in the input resume text, reworded and optimized for the target job description.
2. Return a cover letter object with paragraphs tailored for the role.

JSON Output Format:
{
  "tailored_paragraphs": [
    "Reworded Summary...",
    "Reworded Job 1 Bullet 1...",
    "..."
  ],
  "cover_letter": [
    "Dear Hiring Manager...",
    "Body paragraph...",
    "Sincerely..."
  ]
}
"""

def generate_with_retry(prompt, retries=5, delay=3):
    for attempt in range(retries):
        try:
            return client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json"
                )
            )
        except ServerError as e:
            if attempt == retries - 1:
                raise e
            print(f"API busy (503). Retrying attempt {attempt + 1}/{retries} in {delay}s...")
            time.sleep(delay)
            delay *= 2

print("Analyzing resume structure and tailoring content...")
response = generate_with_retry(f"Resume Content:\n{full_resume_text}\n\nJob Description:\n{job_description}")

data = json.loads(response.text)
tailored_list = data.get("tailored_paragraphs", [])

# 2. Overwrite text in-place while retaining original styling runs
tailored_idx = 0
for p in doc.paragraphs:
    if p.text.strip():
        if tailored_idx < len(tailored_list):
            new_text = tailored_list[tailored_idx]
            if p.runs:
                p.runs[0].text = new_text
                for run in p.runs[1:]:
                    run.text = ""
            else:
                p.text = new_text
            tailored_idx += 1

doc.save("tailored_resume.docx")
print("Saved tailored_resume.docx successfully.")

# 3. Generate Cover Letter
cv_doc = docx.Document()
cv_doc.add_heading("Cover Letter", level=1)
for paragraph in data.get("cover_letter", []):
    cv_doc.add_paragraph(paragraph)

cv_doc.save("cover_letter.docx")
print("Saved cover_letter.docx successfully.")
