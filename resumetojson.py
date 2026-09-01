import json, os
from pypdf import PdfReader
from google import genai
from google.genai import types

# 1. Read raw text from your actual resume PDF
reader = PdfReader("resume.pdf")
raw_text = "".join([page.extract_text() for page in reader.pages])

# 2. Ask Gemini 3.6 Flash to structure it into JSON
client = genai.Client(api_key=os.getenv("insert_your_own_api_key_here"))

prompt = f"Convert this raw resume text into structured JSON with full_name, contact, work_experience (company, role, dates, bullets), education, and skills:\n\n{raw_text}"

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt,
    config=types.GenerateContentConfig(response_mime_type="application/json")
)

# 3. Save JSON automatically
with open("master_resume.json", "w") as f:
    f.write(response.text)

print("Master resume JSON generated successfully!")
