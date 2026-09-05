# ATS Resume & Cover Letter Studio

A local Flask app that takes a master resume and a target job description, then uses Gemini to generate an ATS-optimized resume, matching skills, and a tailored cover letter. It renders both as PDFs and saves recent outputs locally for download.

<img width="2940" height="1912" alt="Screenshot 2026-09-05 at 11 18 03" src="https://github.com/user-attachments/assets/3b767356-341d-43fe-b81b-2091f13fe95f" />


## Features

- Paste a target job description
- Load a structured master resume JSON
- Ask Gemini to tailor content for the role
- Keep skills, experience, and projects grounded in the original resume
- Generate a resume PDF and cover letter PDF
- Save recent job applications locally in a history file
- Download previous generated PDFs from the browser UI

## Tech Stack

- Python
- Flask
- Google GenAI SDK
- Playwright
- Jinja2 templates
- JSON-based resume data model

## Project Structure

```text
local-cv-builder/
├── app.py                 # Main Flask app
├── resumetojson.py        # Converts a resume PDF into structured JSON
├── direct_edit.py         # Earlier draft for DOCX-based editing
├── run_app.sh             # Convenience script to launch the app
├── outputs/               # Generated PDFs/history files
├── master_resume.json     # Your source resume data
├── resume.pdf             # Input resume PDF (if used)
├── README.md              # Project documentation
└── venv/                  # Local virtual environment
```

## Setup

1. Open the project folder.
2. Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install flask google-genai playwright
playwright install chromium
```

4. Set your Gemini API key:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

If you use the helper script, make sure the API key is set there too.

## Run the App

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

You can also run:

```bash
bash run_app.sh
```

## How It Works

1. The app loads your master resume JSON.
2. You paste a job description.
3. Gemini receives both the resume and JD with a strict system prompt.
4. The model returns tailored summary, skill categories, experience bullets, projects, and cover letter paragraphs.
5. The app sanitizes output to remove formatting artifacts and weird LaTeX leftovers.
6. Flask renders HTML templates for the resume and cover letter.
7. Playwright exports both to PDF.
8. The PDFs are saved under the outputs directory and listed in the app history.

## Important Notes

- The master resume is treated as the single source of truth.
- The app avoids inventing facts, but the model can still produce messy output, so sanitization and careful prompt design are important.
- The app is strongest when your master resume JSON is accurate and complete.

## LLM-Assisted Development

This app was built in a vibe-coding workflow:

- LLMs were used to generate the Flask app structure and route logic.
- They helped draft the ATS prompt and JSON output schema.
- They generated the resume and cover letter HTML templates.
- They helped debug issues around JSON parsing, Jinja rendering, PDF export, and file cleanup.
- They were also used to improve the prompt until the output behaved more consistently.

This dramatically sped up development, but the final output still needed human review and validation.

## Problems Faced While Building It

The main issues were:

- Gemini sometimes returned invalid JSON or weird formatting.
- Model output occasionally included LaTeX-like fragments such as `$...$`.
- The app had to be strict about not inventing skills, tools, or achievements.
- PDF rendering needed tuning for margins and page layout.
- History and output file cleanup had to be designed carefully to avoid stale files.
- Prompt tuning was necessary to keep the model truthful and ATS-aware without over-claiming.

## Best Practice

LLMs were extremely useful for scaffolding and iteration, but the app still needed:

- strong prompt guardrails
- output validation
- careful sanitization
- manual checking against the real resume data

That combination is what makes the project practical rather than just flashy.

## Future Improvements

- Add resume upload in the browser instead of relying on a local JSON file
- Support DOCX/PDF input for resume conversion automatically
- Add better validation for generated JSON structure
- Improve PDF layout consistency across job types
- Add a settings panel for default output folder and API configuration

## License

This project is intended for personal and portfolio use. Add your preferred license if you plan to share it publicly.
