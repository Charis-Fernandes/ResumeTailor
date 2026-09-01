import os
import json
import time
import uuid
import signal
import threading
import re
from flask import Flask, render_template_string, request, send_from_directory, jsonify
from google import genai
from google.genai import types
from playwright.sync_api import sync_playwright

app = Flask(__name__)

# Initialize Google GenAI Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(OUTPUT_DIR, "history.json")

def format_skill_title(key):
    """Converts keys like 'web_and_backend' to 'Web & Backend'"""
    words = key.replace('_', ' ').split()
    formatted = []
    for w in words:
        if w.lower() == 'and':
            formatted.append('&')
        elif w.lower() in ['ai', 'it', 'api', 'apis', 'sql', 'iot', 'aws']:
            formatted.append(w.upper())
        else:
            formatted.append(w.capitalize())
    return ' '.join(formatted)

def sanitize_ai_text(data):
    """Strips residual LaTeX syntax and formatting artifacts."""
    if isinstance(data, str):
        s = data.replace(r'\$', '$').replace(r'\&', '&').replace(r'\%', '%')
        s = s.replace('\$ \$', '|').replace('$|$', '|').replace('$$', '|')
        s = re.sub(r'\$([A-Za-z0-9_]+)_{[a-zA-Z0-9]}\$', r'\1', s)
        s = re.sub(r'\$([^$]+)\$', r'\1', s)
        s = re.sub(r'\bAl\b', 'AI', s)
        s = re.sub(r'\bAl\s+', 'AI ', s)
        return s.strip()
    elif isinstance(data, list):
        return [sanitize_ai_text(item) for item in data]
    elif isinstance(data, dict):
        return {k: sanitize_ai_text(v) for k, v in data.items()}
    return data

app.jinja_env.filters['format_title'] = format_skill_title

def cleanup_old_files():
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    except Exception:
        return {}
    
    cutoff = time.time() - 86400
    updated_history = {}
    
    for job_id, data in history.items():
        if data.get("timestamp", 0) > cutoff:
            updated_history[job_id] = data
        else:
            for fname in [data.get("resume_file"), data.get("cv_file")]:
                if fname:
                    fpath = os.path.join(OUTPUT_DIR, fname)
                    if os.path.exists(fpath):
                        os.remove(fpath)
                        
    with open(HISTORY_FILE, "w") as f:
        json.dump(updated_history, f, indent=2)
        
    return updated_history

def save_history_entry(job_id, entry):
    history = cleanup_old_files()
    history[job_id] = entry
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def generate_pdf_from_html(html_content, output_path):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(html_content)
        page.pdf(
            path=output_path,
            format="Letter",
            print_background=True,
            # Increased margins to prevent edge-to-edge bleeding
            margin={"top": "0.5in", "bottom": "0.5in", "left": "0.6in", "right": "0.6in"}
        )
        browser.close()


SYSTEM_INSTRUCTION = """
<role>
You are a Senior ATS Recruiter, Resume Architect, and Technical Hiring Specialist across engineering, software, IT, data, and adjacent professional roles.
Your mission: parse the candidate's Master Resume JSON and a target Job Description (JD), then produce the strongest truthful, ATS-optimized, JD-tailored resume and cover letter, output as JSON only.
</role>

<definitions>
- "High-impact bullet": a bullet describing a concrete action with a specific technical method and, where the Master Resume supports it, a measurable result (a number, percentage, scale, or comparison) — not a duty description.
- "Evidence strength" for a JD requirement: STRONG = the Master Resume shows direct, hands-on use of it. PARTIAL = adjacent or indirect exposure. NONE = no support at all.
- "Recency": how recently the supporting experience occurred relative to the candidate's other work — more recent experience is weighted higher when choosing what to keep.
- "Umbrella term": a category label (e.g., "Microsoft Office," "Google Workspace," "cloud tools") that groups several named tools together. Avoid these whenever the JD names the underlying tools individually and the Master Resume supports them individually — see Rule 4.
</definitions>

<core_rules>

RULE 1 — MASTER RESUME IS THE ONLY SOURCE OF TRUTH
Never invent, assume, exaggerate, or fabricate anything. Every factual claim must be directly supported by the Master Resume.
Never invent: technologies, tools, metrics, responsibilities, achievements, companies, titles, dates, certifications, team sizes, users/customers, or performance improvements.
You may rewrite, combine, shorten, reorder, and reframe existing information — never create new facts.

RULE 2 — ANALYZE THE JD
Identify and prioritize: must-have requirements, preferred requirements, named tools/software/frameworks/platforms, domain knowledge, responsibilities, education/certifications, and key terminology.
Weight requirements by explicit importance, repetition, and relevance to the actual day-to-day responsibilities — do not treat every mention equally.

RULE 3 — MAP JD REQUIREMENTS TO CANDIDATE EVIDENCE
For each important JD requirement, classify evidence as STRONG / PARTIAL / NONE (see definitions).
Never add a JD requirement rated NONE to the resume's skills, experience, or projects.

RULE 4 — EXACT MIRRORING OF NAMED TOOLS AND TERMS
This applies to every named tool, platform, or piece of software in the JD — not just programming languages or cloud technologies. Examples: Microsoft Excel, Microsoft Word, Google Drive, Salesforce, AutoCAD, Adobe Photoshop, Jira, SAP.
If the JD names a tool individually and the Master Resume supports the candidate having used that specific tool, list it by its exact name — do not fold it into an umbrella term (see definitions) even if that would look tidier.
Also mirror the JD's exact phrasing for methodologies and technical terms (e.g., "Amazon Web Services" rather than silently substituting "AWS," if that's the JD's own phrasing).
Never mirror a name the candidate has no evidence for — that violates Rule 1.

RULE 5 — SELECT AND REMOVE
Prioritize content by: Relevance × Evidence Strength × Impact × Recency.
Remove content that contributes little to the target role. Don't delete strong, highly relevant evidence purely to hit a length target.

RULE 6 — EXPERIENCE BULLETS
- Cap at 3 high-impact bullets per role (especially for early-career candidates). Give quantitative metrics where the Master Resume supports them. If no metric is available, focus on a strong action + technical method + context.
- Every bullet opens with a strong, specific action verb (e.g., Architected, Engineered, Deployed, Optimized, Reduced, Automated) — never a weak or passive opener.
- Structure: Action + Technical Method/Context + Result. Use a full metric-based bullet when the Master Resume genuinely supports a number — never force a fabricated-sounding metric into a bullet that doesn't have one in the source data.

RULE 7 — PROJECTS
Select only the 2–3 projects most relevant to the JD; omit the rest completely.
Max 2 bullets per selected project. Prefer projects that provide evidence for important JD requirements not already covered by experience.

RULE 8 — SKILLS
Group into up to 4 categories. Every listed skill or tool must be supported by the Master Resume — never add something just because the JD mentions it.
Within a category, list named tools individually per Rule 4 rather than bundled under an umbrella term, whenever the JD calls them out individually.
Avoid keyword stuffing.

RULE 9 — SUMMARY
2–3 lines, targeted to the role. May synthesize existing facts but must not introduce new technologies, seniority, specialization, or experience.

RULE 10 — COVER LETTER (3 paragraphs, all grounded in the Master Resume)
1. Hook — direct, specific enthusiasm for [Company] and [Role]. No generic fluff; reference something concrete from the JD or company description.
2. Proof — map 1–2 verified achievements from the Master Resume directly to the JD's biggest pain point or core requirement.
3. Close — concise, confident, emphasizes readiness for next steps.
If the JD centers on a specific technology, method, or tool the candidate doesn't yet have evidence for, it's fine to name it explicitly as something the candidate is eager to learn or apply — framed clearly as a growth interest, never as a possessed skill. This shows close reading of the JD without violating Rule 1.
Never invent motivations, achievements, company knowledge, or experience not present in the Master Resume.

RULE 11 — FILL THE PAGE, DON'T JUST SHRINK IT
The caps in Rules 6–8 are ceilings, not fixed targets. If applying them would leave the page visibly sparse because the Master Resume has more genuinely relevant material, use it: add a bullet, keep a third project, or include a brief additional section (e.g., relevant coursework, achievements, activities) — but only with content that exists in the Master Resume and is relevant to the target role. A well-filled, professional one-pager beats a rigidly minimal one with dead space. Conversely, never pad with filler, restated bullets, or irrelevant detail just to fill space.

RULE 12 — LAYOUT & FORMATTING
- Project tech stacks stay on the left, under the project title; dates/location stay on the right.
- Preserve company names, job titles, dates, degree titles, and institution names exactly as given.
- Always write "AI" in uppercase (never "Al" or "al").
- No LaTeX syntax or backslashes.
- Return pure JSON only — no commentary, no markdown fences, no text outside the JSON object.

</core_rules>

<match_score_and_keywords>
MATCH SCORE: base it on actual high-priority JD coverage, technical/responsibility/domain alignment, and evidence strength. Do not inflate it from keyword overlap alone.
MISSING KEYWORDS: list only important, unsupported JD requirements (skip trivial phrasing). For each, provide: keyword, type, importance.
</match_score_and_keywords>

<output_schema>
Return pure JSON only, matching this schema exactly:
{
  "match_score": 90,
  "missing_keywords": [
    { "keyword": "AWS", "type": "technical", "importance": "high" }
  ],
  "tailored_summary": "...",
  "tailored_skills": {
    "Category 1": [],
    "Category 2": [],
    "Category 3": [],
    "Category 4": []
  },
  "tailored_experience": [
    { "company": "", "role": "", "location": "", "dates": "", "bullets": [] }
  ],
  "tailored_projects": [
    { "name": "", "role": "", "tech_stack": "", "dates": "", "bullets": [] }
  ],
  "company_name": "",
  "role_title": "",
  "cover_letter_paragraphs": []
}
</output_schema>

<verification_checklist>
Before returning the JSON, verify:
- Every claim traces directly to the Master Resume — no invented fact.
- No JD requirement rated NONE was added to skills, experience, or projects.
- Every bullet opens with a strong action verb.
- Named tools and JD terminology are mirrored exactly, individually, only where genuinely earned — no umbrella-term bundling of tools the JD lists separately.
- High-priority JD requirements are addressed wherever evidence exists.
- Irrelevant content was removed; important accomplishments weren't needlessly cut, and the page isn't left visibly sparse if more genuine content was available.
- No section is repetitive or keyword-stuffed.
- Dates, titles, companies, degrees, and institutions are unchanged from the Master Resume.
- JSON is valid, matches the schema exactly, and contains nothing else.
</verification_checklist>
"""

UI_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ATS Resume & CV Studio</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 40px auto; max-width: 900px; background: #f8fafc; color: #0f172a; line-height: 1.5; }
        .card { background: white; padding: 32px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px; border: 1px solid #e2e8f0; }
        h1 { margin-top: 0; font-size: 22px; color: #0f172a; border-bottom: 2px solid #0f172a; padding-bottom: 8px; }
        textarea { width: 100%; height: 160px; padding: 12px; font-size: 14px; border: 1px solid #cbd5e1; border-radius: 6px; box-sizing: border-box; font-family: inherit; outline: none; }
        textarea:focus { border-color: #0f172a; }
        button { background: #0f172a; color: white; padding: 10px 20px; border: none; border-radius: 6px; font-size: 14px; cursor: pointer; font-weight: 600; }
        button:hover { background: #1e293b; }
        button:disabled { background: #94a3b8; cursor: not-allowed; }
        .btn-download { display: inline-flex; align-items: center; background: #059669; color: white; text-decoration: none; padding: 8px 16px; border-radius: 4px; font-weight: 600; margin-right: 10px; font-size: 13px; }
        .btn-download:hover { background: #047857; }
        .score-badge { font-size: 28px; font-weight: 800; color: #2563eb; }
        .tag { display: inline-block; background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; padding: 3px 10px; border-radius: 4px; font-size: 12px; margin-right: 6px; margin-bottom: 6px; font-weight: 600; }
        
        .progress-container { display: none; margin-top: 20px; }
        .progress-bar-bg { width: 100%; background: #e2e8f0; height: 8px; border-radius: 4px; overflow: hidden; margin-bottom: 10px; }
        .progress-bar-fill { width: 0%; height: 100%; background: #0f172a; transition: width 0.4s ease; }
        .thinking-box { background: #0f172a; padding: 14px; border-radius: 6px; font-family: "Courier New", monospace; font-size: 12px; color: #38bdf8; max-height: 100px; overflow-y: auto; }

        .history-item { border-bottom: 1px solid #e2e8f0; padding: 12px 0; display: flex; justify-content: space-between; align-items: center; }
        .history-item:last-child { border-bottom: none; }
    </style>
</head>
<body>
    <div class="card">
        <h1>ATS Technical Resume & Cover Letter Studio</h1>
        <form id="processForm">
            <label><b>Target Job Description:</b></label><br><br>
            <textarea id="jd" name="job_description" placeholder="Paste target job description text here..." required></textarea><br><br>
            <button type="submit" id="submitBtn">Generate Tailored Resume & CV</button>
        </form>

        <div id="progressContainer" class="progress-container">
            <div class="progress-bar-bg">
                <div id="progressBar" class="progress-bar-fill"></div>
            </div>
            <div id="thinkingBox" class="thinking-box">> Initializing process...</div>
        </div>
    </div>

    <div id="resultsCard" class="card" style="display: none;">
        <h2>Analysis Results</h2>
        <p><b>Target Role:</b> <span id="resRole" style="font-weight: bold;"></span> (<span id="resCompany"></span>)</p>
        <p><b>ATS Match Score:</b> <span id="resScore" class="score-badge">0%</span></p>
        <p><b>Missing Keywords:</b></p>
        <div id="resKeywords"></div>
        <hr style="margin: 20px 0; border: none; border-top: 1px solid #e2e8f0;">
        <h3>Generated Documents</h3>
        <a id="dlResume" href="#" class="btn-download" download>📄 Download Resume (PDF)</a>
        <a id="dlCv" href="#" class="btn-download" download>📝 Download Cover Letter (PDF)</a>
    </div>

    <div class="card">
        <h2>Saved History (Last 24 Hours)</h2>
        <div id="historyList">Loading history...</div>
    </div>
    <!-- Add this button in your UI_TEMPLATE -->
<button onclick="killServer()" style="background: #dc2626; color: white; border: none; padding: 8px 14px; border-radius: 6px; font-weight: 600; cursor: pointer; float: right;">🛑 Power Off Studio</button>

<script>
async function killServer() {
    if (!confirm("Shut down the app server completely?")) return;
    
    try {
        await fetch('/shutdown', { method: 'POST' });
    } catch (e) {
        // Ignored — server dies mid-flight
    }
    
    document.body.innerHTML = `
        <div style="display:flex; justify-content:center; align-items:center; height:100vh; background:#0f172a; color:white; font-family:sans-serif;">
            <div style="text-align:center;">
                <h1 style="color:#f87171; margin-bottom:8px;">🛑 Studio Off</h1>
                <p style="color:#94a3b8;">Port 5000 freed. You can close this tab now.</p>
            </div>
        </div>
    `;
}
</script>



    <script>
        function logThinking(msg) {
            const box = document.getElementById("thinkingBox");
            box.innerHTML += "<br>> " + msg;
            box.scrollTop = box.scrollHeight;
        }

        function setProgress(percent) {
            document.getElementById("progressBar").style.width = percent + "%";
        }

        async function loadHistory() {
            const res = await fetch('/history');
            const data = await res.json();
            const container = document.getElementById("historyList");
            
            if (Object.keys(data).length === 0) {
                container.innerHTML = "<p style='color: #64748b; font-size: 14px;'>No saved files in the last 24 hours.</p>";
                return;
            }

            let html = "";
            for (const [jobId, item] of Object.entries(data)) {
                const dateStr = new Date(item.timestamp * 1000).toLocaleString();
                html += `
                    <div class="history-item">
                        <div>
                            <b style="color: #0f172a;">${item.role_title}</b> @ ${item.company_name} <br>
                            <small style="color: #64748b;">${dateStr} | Match Score: ${item.match_score}%</small>
                        </div>
                        <div>
                            <a href="/download/${item.resume_file}" class="btn-download" download>Resume PDF</a>
                            <a href="/download/${item.cv_file}" class="btn-download" download>CV PDF</a>
                        </div>
                    </div>
                `;
            }
            container.innerHTML = html;
        }

        document.getElementById("processForm").addEventListener("submit", async function(e) {
            e.preventDefault();
            const btn = document.getElementById("submitBtn");
            const progress = document.getElementById("progressContainer");
            const resultsCard = document.getElementById("resultsCard");
            const jdText = document.getElementById("jd").value;

            btn.disabled = true;
            progress.style.display = "block";
            resultsCard.style.display = "none";
            document.getElementById("thinkingBox").innerHTML = "> Initializing request...";

            setProgress(25);
            logThinking("Loading Master Resume data...");
            
            setTimeout(() => {
                setProgress(55);
                logThinking("Pruning & Optimizing via Gemini 3.6 Flash Engine...");
            }, 800);

            try {
                const response = await fetch("/process", {
                    method: "POST",
                    headers: { "Content-Type": "application/x-www-form-urlencoded" },
                    body: new URLSearchParams({ job_description: jdText })
                });

                setProgress(85);
                logThinking("Rendering strict 1-page layout via Playwright...");

                const result = await response.json();
                
                setProgress(100);
                logThinking("Success! Tailored documents generated.");

                document.getElementById("resRole").innerText = result.role_title;
                document.getElementById("resCompany").innerText = result.company_name;
                document.getElementById("resScore").innerText = result.match_score + "%";
                
                const kwContainer = document.getElementById("resKeywords");
                kwContainer.innerHTML = "";
                (result.missing_keywords || []).forEach(kw => {
                    kwContainer.innerHTML += `<span class="tag">${kw}</span>`;
                });

                document.getElementById("dlResume").href = "/download/" + result.resume_file;
                document.getElementById("dlCv").href = "/download/" + result.cv_file;
                
                resultsCard.style.display = "block";
                await loadHistory();

            } catch (err) {
                logThinking("Error: Processing failed.");
            } finally {
                btn.disabled = false;
            }
        });

        loadHistory();
    </script>
    <button onclick="clearHistory()" style="background: #dc2626; margin-left: 10px;">Clear History & Files</button>

<script>
async function clearHistory() {
    if (!confirm("Are you sure you want to clear all history and generated PDFs?")) return;
    await fetch('/clear-history', { method: 'POST' });
    loadHistory();
    alert("History cleared successfully!");
}
</script>

</body>
</html>
"""

RESUME_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @page { 
        size: letter; 
        margin: 0; 
    }
    * { 
        box-sizing: border-box; 
    }
    body {
        font-family: 'Roboto', -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
        color: #000000;
        background: #ffffff;
        font-size: 9pt;
        line-height: 1.3;
        margin: 0;
        /* Add explicit padding to the HTML container to guarantee clear spacing */
        padding: 0.5in 0.6in;
        width: 100%;
    }

    .header { text-align: center; margin-bottom: 8px; }
    .header .name { font-size: 18pt; font-weight: 700; color: #000000; margin-bottom: 2px; line-height: 1; }
    .header .contact { font-size: 8.5pt; color: #000000; }
    .header .contact a { color: #000000; text-decoration: none; }

    .section-title {
        font-size: 10pt;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #000000;
        border-bottom: 1px solid #000000;
        margin-top: 8px;
        margin-bottom: 4px;
        padding-bottom: 1px;
    }

    .subheading-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 3px;
        margin-bottom: 1px;
        table-layout: fixed;
    }
    .subheading-table td { padding: 0; vertical-align: top; }
    .sub-left-top { font-weight: 700; font-size: 9pt; text-align: left; width: 70%; color: #000000; }
    .sub-right-top { font-weight: 700; font-size: 9pt; text-align: right; width: 30%; color: #000000; }
    .sub-left-bottom { font-style: italic; font-size: 8.5pt; text-align: left; width: 100%; color: #222222; }

    ul { margin: 2px 0 4px 0; padding-left: 15px; }
    li { margin-bottom: 2px; font-size: 8.5pt; color: #000000; text-align: justify; }

    .summary-text { font-size: 8.5pt; color: #000000; margin-bottom: 4px; text-align: justify; }
    .skills-list { margin: 2px 0 4px 0; padding-left: 0; list-style: none; }
    .skills-list li { margin-bottom: 2px; font-size: 8.5pt; text-align: left; }
</style>
</head>
<body>
    <div class="header">
        <div class="name">{{ master.full_name }}</div>
        <div class="contact">
            {% if master.contact.phone %}{{ master.contact.phone }} | {% endif %}
            <a href="mailto:{{ master.contact.email }}">{{ master.contact.email }}</a>
            {% if master.contact.linkedin %} | <a href="https://{{ master.contact.linkedin }}">{{ master.contact.linkedin }}</a>{% endif %}
            {% if master.contact.github %} | <a href="https://{{ master.contact.github }}">{{ master.contact.github }}</a>{% endif %}
        </div>
    </div>

    <div class="section-title">Summary</div>
    <div class="summary-text">{{ ai.tailored_summary }}</div>

    <div class="section-title">Technical Skills</div>
    <ul class="skills-list">
        {% set skills_dict = ai.tailored_skills if ai.tailored_skills else master.skills %}
        {% for cat, skills_list in skills_dict.items() %}
        <li>
            <strong>{{ cat | format_title }}:</strong> 
            {{ skills_list | join(', ') if skills_list is iterable and skills_list is not string else skills_list }}
        </li>
        {% endfor %}
    </ul>

    <div class="section-title">Experience</div>
    {% set experiences = ai.tailored_experience if ai.tailored_experience else master.work_experience %}
    {% for job in experiences %}
        <table class="subheading-table">
            <tr>
                <td class="sub-left-top">{{ job.role }}</td>
                <td class="sub-right-top">{{ job.dates }}</td>
            </tr>
            <tr>
                <td class="sub-left-bottom" colspan="2">{{ job.company }}{% if job.location %} — {{ job.location }}{% endif %}</td>
            </tr>
        </table>
        <ul>
        {% for bullet in job.bullets %}
            <li>{{ bullet }}</li>
        {% endfor %}
        </ul>
    {% endfor %}

    {% if ai.tailored_projects %}
    <div class="section-title">Key Projects</div>
    {% for proj in ai.tailored_projects %}
        <table class="subheading-table">
            <tr>
                <td class="sub-left-top">{{ proj.name }}</td>
                <td class="sub-right-top">{{ proj.dates if proj.dates else '' }}</td>
            </tr>
            <tr>
                <td class="sub-left-bottom" colspan="2">
                    <em>{{ proj.role if proj.role else 'Technical Project' }}</em>
                    {% if proj.tech_stack %} | <strong>Tech Stack:</strong> {{ proj.tech_stack }}{% endif %}
                </td>
            </tr>
        </table>
        <ul>
        {% for bullet in proj.bullets %}
            <li>{{ bullet }}</li>
        {% endfor %}
        </ul>
    {% endfor %}
    {% endif %}

    {% if master.education %}
    <div class="section-title">Education</div>
    {% for edu in master.education %}
        <table class="subheading-table">
            <tr>
                <td class="sub-left-top">{{ edu.institution }}</td>
                <td class="sub-right-top">{{ edu.year if edu.year else '' }}</td>
            </tr>
            <tr>
                <td class="sub-left-bottom" colspan="2">{{ edu.degree }}</td>
            </tr>
        </table>
    {% endfor %}
    {% endif %}

    {% if master.certifications %}
    <div class="section-title">Certifications</div>
    <ul>
    {% for cert in master.certifications %}
        <li>
            {% if cert is mapping %}
                <strong>{{ cert.name }}</strong> - {{ cert.details }} ({{ cert.date }})
            {% else %}
                {{ cert }}
            {% endif %}
        </li>
    {% endfor %}
    </ul>
    {% endif %}
</body>
</html>
"""

CV_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @page { size: letter; margin: 0; }
    * { box-sizing: border-box; }
    body {
        font-family: 'Roboto', -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
        color: #000000;
        background: #ffffff;
        font-size: 9.5pt;
        line-height: 1.4;
        margin: 0;
        padding: 0.4in 0.5in;
        width: 100%;
    }
    .header { text-align: center; margin-bottom: 20px; border-bottom: 1px solid #000; padding-bottom: 6px; }
    .header .name { font-size: 18pt; font-weight: 700; color: #000; margin-bottom: 2px; }
    .header .contact { font-size: 8.5pt; color: #000; }
    .recipient { margin-bottom: 16px; font-size: 9.5pt; }
    p { margin-bottom: 10px; text-align: justify; }
    .signature { margin-top: 24px; }
</style>
</head>
<body>
    <div class="header">
        <div class="name">{{ master.full_name }}</div>
        <div class="contact">
            {% if master.contact.phone %}{{ master.contact.phone }} | {% endif %}
            <a href="mailto:{{ master.contact.email }}">{{ master.contact.email }}</a>
            {% if master.contact.linkedin %} | <a href="https://{{ master.contact.linkedin }}">{{ master.contact.linkedin }}</a>{% endif %}
            {% if master.contact.github %} | <a href="https://{{ master.contact.github }}">{{ master.contact.github }}</a>{% endif %}
        </div>
    </div>

    <div class="recipient">
        <strong>Hiring Manager</strong><br>
        {{ ai.company_name }}<br>
        Re: <strong>{{ ai.role_title }} Application</strong>
    </div>

    <p>Dear Hiring Manager,</p>

    {% for paragraph in ai.cover_letter_paragraphs %}
        <p>{{ paragraph }}</p>
    {% endfor %}

    <div class="signature">
        Sincerely,<br><br>
        <strong>{{ master.full_name }}</strong>
    </div>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(UI_TEMPLATE)

@app.route("/shutdown", methods=["POST"])
def shutdown():
    def hard_kill():
        time.sleep(0.3)  # Brief pause to let Flask send the 200 response
        os._exit(0)      # Hard exit bypasses Werkzeug reloader completely

    threading.Thread(target=hard_kill, daemon=True).start()
    return jsonify({"status": "shutting down"})

@app.route("/history")
def get_history():
    history = cleanup_old_files()
    return jsonify(history)

@app.route("/clear-history", methods=["POST"])
def clear_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
            for data in history.values():
                for fname in [data.get("resume_file"), data.get("cv_file")]:
                    if fname:
                        fpath = os.path.join(OUTPUT_DIR, fname)
                        if os.path.exists(fpath):
                            os.remove(fpath)
            os.remove(HISTORY_FILE)
        except Exception:
            pass
    return jsonify({"status": "success", "message": "History and files cleared."})


@app.route("/process", methods=["POST"])
def process():
    jd = request.form["job_description"]
    
    master_path = os.path.join(os.path.dirname(__file__), "master_resume.json")
    with open(master_path, "r") as f:
        master_resume = json.load(f)
    
    prompt = f"Master Resume:\n{json.dumps(master_resume)}\n\nTarget Job Description:\n{jd}"
    
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json"
        )
    )
    
    raw_json = json.loads(response.text)
    ai_data = sanitize_ai_text(raw_json)
    
    job_id = uuid.uuid4().hex[:8]
    resume_filename = f"resume_{job_id}.pdf"
    cv_filename = f"cv_{job_id}.pdf"
    
    rendered_resume = render_template_string(RESUME_HTML_TEMPLATE, master=master_resume, ai=ai_data)
    resume_pdf_path = os.path.join(OUTPUT_DIR, resume_filename)
    generate_pdf_from_html(rendered_resume, resume_pdf_path)
    
    rendered_cv = render_template_string(CV_HTML_TEMPLATE, master=master_resume, ai=ai_data)
    cv_pdf_path = os.path.join(OUTPUT_DIR, cv_filename)
    generate_pdf_from_html(rendered_cv, cv_pdf_path)
    
    entry = {
        "timestamp": time.time(),
        "company_name": ai_data.get("company_name", "Target Company"),
        "role_title": ai_data.get("role_title", "Target Role"),
        "match_score": ai_data.get("match_score", 0),
        "resume_file": resume_filename,
        "cv_file": cv_filename
    }
    save_history_entry(job_id, entry)
    
    ai_data["resume_file"] = resume_filename
    ai_data["cv_file"] = cv_filename
    
    return jsonify(ai_data)

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
