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
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>ATS Resume & CV Studio</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        /* ── Reset & Base ── */
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
            --bg-deep: #06060f;
            --bg-card: rgba(255, 255, 255, 0.04);
            --bg-card-hover: rgba(255, 255, 255, 0.07);
            --border-card: rgba(255, 255, 255, 0.08);
            --border-card-hover: rgba(255, 255, 255, 0.15);
            --text-primary: #f0f0f5;
            --text-secondary: #8b8b9e;
            --text-muted: #55556a;
            --accent: #818cf8;
            --accent-glow: rgba(129, 140, 248, 0.35);
            --accent-2: #c084fc;
            --accent-green: #34d399;
            --accent-green-glow: rgba(52, 211, 153, 0.3);
            --accent-red: #f87171;
            --radius: 16px;
            --radius-sm: 10px;
            --radius-xs: 6px;
            --spring: cubic-bezier(0.34, 1.56, 0.64, 1);
            --smooth: cubic-bezier(0.4, 0, 0.2, 1);
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: var(--bg-deep);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* ── Animated Background ── */
        .bg-scene {
            position: fixed; inset: 0; z-index: 0; overflow: hidden; pointer-events: none;
        }
        .bg-orb {
            position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.35;
            animation: orbFloat 20s ease-in-out infinite alternate;
        }
        .bg-orb:nth-child(1) { width: 500px; height: 500px; background: #6366f1; top: -10%; left: -5%; animation-duration: 22s; }
        .bg-orb:nth-child(2) { width: 400px; height: 400px; background: #a855f7; bottom: -10%; right: -5%; animation-duration: 18s; animation-delay: -5s; }
        .bg-orb:nth-child(3) { width: 300px; height: 300px; background: #3b82f6; top: 50%; left: 40%; animation-duration: 25s; animation-delay: -10s; }
        @keyframes orbFloat {
            0% { transform: translate(0, 0) scale(1); }
            33% { transform: translate(40px, -30px) scale(1.1); }
            66% { transform: translate(-20px, 40px) scale(0.95); }
            100% { transform: translate(30px, -20px) scale(1.05); }
        }

        /* grain overlay */
        .bg-scene::after {
            content: ''; position: absolute; inset: 0;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
            opacity: 0.5;
        }

        /* ── Layout ── */
        .container {
            position: relative; z-index: 1;
            max-width: 780px; margin: 0 auto;
            padding: 40px 24px 60px;
        }

        /* ── Header ── */
        .hero {
            text-align: center; margin-bottom: 40px;
            animation: fadeSlideUp 0.8s var(--smooth) both;
        }
        .hero-badge {
            display: inline-flex; align-items: center; gap: 6px;
            background: rgba(129, 140, 248, 0.12); border: 1px solid rgba(129, 140, 248, 0.2);
            padding: 6px 14px; border-radius: 999px; font-size: 12px; font-weight: 600;
            color: var(--accent); letter-spacing: 0.5px; text-transform: uppercase;
            margin-bottom: 16px;
        }
        .hero-badge .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent-green); animation: pulse 2s ease-in-out infinite; }
        .hero h1 {
            font-size: 36px; font-weight: 800; letter-spacing: -1px; line-height: 1.15;
            background: linear-gradient(135deg, #f0f0f5 0%, #818cf8 50%, #c084fc 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero p { color: var(--text-secondary); font-size: 15px; margin-top: 8px; }

        /* ── Cards ── */
        .card {
            background: var(--bg-card);
            backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-card);
            border-radius: var(--radius);
            padding: 32px;
            margin-bottom: 20px;
            transition: border-color 0.3s ease, box-shadow 0.3s ease, transform 0.3s var(--spring);
            animation: fadeSlideUp 0.7s var(--smooth) both;
        }
        .card:nth-child(2) { animation-delay: 0.1s; }
        .card:nth-child(3) { animation-delay: 0.2s; }
        .card:nth-child(4) { animation-delay: 0.3s; }
        .card:hover {
            border-color: var(--border-card-hover);
            box-shadow: 0 8px 40px rgba(0,0,0,0.25), 0 0 0 1px rgba(255,255,255,0.05);
        }

        .card-title {
            font-size: 13px; font-weight: 700; text-transform: uppercase;
            letter-spacing: 1.2px; color: var(--text-secondary);
            margin-bottom: 20px; display: flex; align-items: center; gap: 8px;
        }
        .card-title .icon { font-size: 16px; }

        /* ── Form ── */
        .form-label {
            display: block; font-size: 14px; font-weight: 600;
            color: var(--text-primary); margin-bottom: 8px;
        }
        textarea {
            width: 100%; height: 180px; padding: 16px;
            font-size: 14px; font-family: 'Inter', sans-serif;
            background: rgba(255,255,255,0.03); color: var(--text-primary);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: var(--radius-sm);
            outline: none; resize: vertical;
            transition: border-color 0.3s ease, box-shadow 0.3s ease, background 0.3s ease;
        }
        textarea::placeholder { color: var(--text-muted); }
        textarea:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
            background: rgba(255,255,255,0.05);
        }

        /* ── Buttons ── */
        .btn {
            display: inline-flex; align-items: center; justify-content: center; gap: 8px;
            padding: 12px 28px; border: none; border-radius: var(--radius-sm);
            font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 700;
            cursor: pointer; position: relative; overflow: hidden;
            transition: transform 0.25s var(--spring), box-shadow 0.3s ease;
        }
        .btn:active { transform: scale(0.96) !important; }

        .btn-primary {
            background: linear-gradient(135deg, #818cf8, #6366f1);
            color: white;
            box-shadow: 0 4px 20px var(--accent-glow), inset 0 1px 0 rgba(255,255,255,0.15);
        }
        .btn-primary:hover {
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 8px 30px var(--accent-glow), inset 0 1px 0 rgba(255,255,255,0.2);
        }
        .btn-primary:disabled {
            opacity: 0.4; cursor: not-allowed;
            transform: none !important; box-shadow: none !important;
        }
        /* shimmer on hover */
        .btn-primary::after {
            content: ''; position: absolute; inset: 0;
            background: linear-gradient(105deg, transparent 40%, rgba(255,255,255,0.15) 50%, transparent 60%);
            transform: translateX(-100%);
            transition: transform 0.6s ease;
        }
        .btn-primary:hover::after { transform: translateX(100%); }

        .btn-download {
            background: rgba(52, 211, 153, 0.1);
            border: 1px solid rgba(52, 211, 153, 0.25);
            color: var(--accent-green); text-decoration: none;
            padding: 10px 20px; border-radius: var(--radius-sm);
            font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 600;
            display: inline-flex; align-items: center; gap: 8px;
            transition: all 0.25s var(--spring);
        }
        .btn-download:hover {
            background: rgba(52, 211, 153, 0.18);
            border-color: rgba(52, 211, 153, 0.4);
            transform: translateY(-2px) scale(1.03);
            box-shadow: 0 4px 20px var(--accent-green-glow);
        }

        .btn-danger {
            background: rgba(248, 113, 113, 0.08);
            border: 1px solid rgba(248, 113, 113, 0.2);
            color: var(--accent-red); padding: 8px 16px;
            border-radius: var(--radius-xs); font-size: 12px; font-weight: 600;
            cursor: pointer; font-family: 'Inter', sans-serif;
            transition: all 0.25s var(--spring);
        }
        .btn-danger:hover {
            background: rgba(248, 113, 113, 0.15);
            border-color: rgba(248, 113, 113, 0.35);
            transform: translateY(-1px);
        }

        /* ── Progress ── */
        .progress-container { display: none; margin-top: 24px; animation: fadeSlideUp 0.4s var(--smooth); }
        .progress-bar-bg {
            width: 100%; height: 4px; border-radius: 999px; overflow: hidden;
            background: rgba(255,255,255,0.06);
        }
        .progress-bar-fill {
            width: 0%; height: 100%; border-radius: 999px;
            background: linear-gradient(90deg, #818cf8, #c084fc, #818cf8);
            background-size: 200% 100%;
            animation: shimmerBar 2s linear infinite;
            transition: width 0.6s var(--smooth);
        }
        @keyframes shimmerBar { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }

        .thinking-box {
            margin-top: 14px; padding: 16px;
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: var(--radius-sm);
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            font-size: 12px; color: var(--accent);
            max-height: 120px; overflow-y: auto;
            line-height: 1.8;
        }
        .thinking-box .line { opacity: 0; animation: typeLine 0.3s var(--smooth) forwards; }
        .thinking-box .line::before { content: '› '; color: var(--text-muted); }

        /* ── Results ── */
        .results-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; flex-wrap: wrap; }
        .result-meta { flex: 1; min-width: 200px; }
        .result-role { font-size: 20px; font-weight: 700; color: var(--text-primary); margin-bottom: 2px; }
        .result-company { font-size: 14px; color: var(--text-secondary); }

        /* Animated Score Ring */
        .score-ring-container { position: relative; width: 100px; height: 100px; flex-shrink: 0; }
        .score-ring { transform: rotate(-90deg); }
        .score-ring-bg { fill: none; stroke: rgba(255,255,255,0.06); stroke-width: 6; }
        .score-ring-fill {
            fill: none; stroke: url(#scoreGradient); stroke-width: 6;
            stroke-linecap: round; stroke-dasharray: 251.2; stroke-dashoffset: 251.2;
            transition: stroke-dashoffset 1.5s var(--smooth);
        }
        .score-value {
            position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
            font-size: 24px; font-weight: 800; color: var(--text-primary);
        }
        .score-label { font-size: 10px; color: var(--text-secondary); text-align: center; margin-top: 4px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }

        .divider {
            border: none; border-top: 1px solid rgba(255,255,255,0.06);
            margin: 24px 0;
        }
        .section-label {
            font-size: 12px; font-weight: 700; text-transform: uppercase;
            letter-spacing: 1px; color: var(--text-muted); margin-bottom: 14px;
        }

        .keywords-container { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
        .tag {
            display: inline-flex; align-items: center; gap: 4px;
            padding: 5px 12px; border-radius: 999px;
            font-size: 12px; font-weight: 600;
            animation: popIn 0.35s var(--spring) both;
            transition: transform 0.2s var(--spring);
        }
        .tag:hover { transform: scale(1.08); }
        .tag-high { background: rgba(248, 113, 113, 0.12); color: #fca5a5; border: 1px solid rgba(248,113,113,0.2); }
        .tag-medium { background: rgba(251, 191, 36, 0.1); color: #fcd34d; border: 1px solid rgba(251,191,36,0.2); }
        .tag-low { background: rgba(148, 163, 184, 0.1); color: #94a3b8; border: 1px solid rgba(148,163,184,0.15); }

        .download-group { display: flex; gap: 12px; flex-wrap: wrap; }

        /* ── History ── */
        .history-item {
            display: flex; justify-content: space-between; align-items: center;
            padding: 16px; margin-bottom: 8px;
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.04);
            border-radius: var(--radius-sm);
            transition: all 0.25s var(--spring);
            animation: fadeSlideUp 0.4s var(--smooth) both;
        }
        .history-item:hover {
            background: rgba(255,255,255,0.05);
            border-color: rgba(255,255,255,0.1);
            transform: translateX(4px);
        }
        .history-role { font-weight: 600; color: var(--text-primary); font-size: 14px; }
        .history-meta { font-size: 12px; color: var(--text-muted); margin-top: 3px; display: flex; align-items: center; gap: 8px; }
        .history-score { color: var(--accent); font-weight: 700; }
        .history-actions { display: flex; gap: 8px; flex-shrink: 0; }
        .history-actions .btn-download { padding: 7px 14px; font-size: 12px; }

        .empty-state {
            text-align: center; padding: 32px; color: var(--text-muted); font-size: 14px;
        }
        .empty-state .empty-icon { font-size: 32px; margin-bottom: 8px; opacity: 0.5; }

        /* ── Footer bar ── */
        .footer-bar {
            display: flex; justify-content: space-between; align-items: center;
            margin-top: 12px; padding: 0 4px;
            animation: fadeSlideUp 0.7s var(--smooth) both;
            animation-delay: 0.4s;
        }
        .footer-info { font-size: 12px; color: var(--text-muted); }

        /* ── Animations ── */
        @keyframes fadeSlideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes popIn {
            from { opacity: 0; transform: scale(0.6); }
            to { opacity: 1; transform: scale(1); }
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(0.85); }
        }
        @keyframes typeLine {
            from { opacity: 0; transform: translateX(-6px); }
            to { opacity: 1; transform: translateX(0); }
        }
        @keyframes scoreCount {
            from { opacity: 0; transform: scale(0.5); }
            to { opacity: 1; transform: scale(1); }
        }
        @keyframes cardReveal {
            from { opacity: 0; transform: translateY(30px) scale(0.97); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

        /* ── Responsive ── */
        @media (max-width: 600px) {
            .container { padding: 20px 16px 40px; }
            .card { padding: 24px; }
            .hero h1 { font-size: 26px; }
            .results-header { flex-direction: column; align-items: center; text-align: center; }
            .history-item { flex-direction: column; gap: 12px; align-items: flex-start; }
        }
    </style>
</head>
<body>
    <!-- Animated background -->
    <div class="bg-scene">
        <div class="bg-orb"></div>
        <div class="bg-orb"></div>
        <div class="bg-orb"></div>
    </div>

    <!-- SVG gradient def for score ring -->
    <svg width="0" height="0" style="position:absolute">
        <defs>
            <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#818cf8"/>
                <stop offset="100%" stop-color="#c084fc"/>
            </linearGradient>
        </defs>
    </svg>

    <div class="container">
        <!-- Hero -->
        <div class="hero">
            <div class="hero-badge"><span class="dot"></span> Gemini 3.6 Flash Powered</div>
            <h1>ATS Resume & Cover Letter Studio</h1>
            <p>Paste a job description. Get a perfectly tailored resume & cover letter in seconds.</p>
        </div>

        <!-- Input Card -->
        <div class="card">
            <div class="card-title"><span class="icon">✦</span> Job Description</div>
            <form id="processForm">
                <textarea id="jd" name="job_description" placeholder="Paste the target job description here…" required></textarea>
                <div style="margin-top: 16px;">
                    <button type="submit" id="submitBtn" class="btn btn-primary">
                        <span id="btnText">Generate Resume & Cover Letter</span>
                        <span id="btnSpinner" style="display:none;">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M12 2v4m0 12v4m-7.07-15.07l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"><animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite"/></path></svg>
                        </span>
                    </button>
                </div>
            </form>

            <div id="progressContainer" class="progress-container">
                <div class="progress-bar-bg">
                    <div id="progressBar" class="progress-bar-fill"></div>
                </div>
                <div id="thinkingBox" class="thinking-box"></div>
            </div>
        </div>

        <!-- Results Card -->
        <div id="resultsCard" class="card" style="display: none;">
            <div class="card-title"><span class="icon">◆</span> Analysis Results</div>
            <div class="results-header">
                <div class="result-meta">
                    <div class="result-role" id="resRole"></div>
                    <div class="result-company" id="resCompany"></div>
                </div>
                <div>
                    <div class="score-ring-container">
                        <svg class="score-ring" width="100" height="100" viewBox="0 0 100 100">
                            <circle class="score-ring-bg" cx="50" cy="50" r="40"/>
                            <circle class="score-ring-fill" id="scoreCircle" cx="50" cy="50" r="40"/>
                        </svg>
                        <div class="score-value" id="resScore">0%</div>
                    </div>
                    <div class="score-label">ATS Match</div>
                </div>
            </div>

            <hr class="divider">
            <div class="section-label">Missing Keywords</div>
            <div id="resKeywords" class="keywords-container"></div>

            <hr class="divider">
            <div class="section-label">Download Documents</div>
            <div class="download-group">
                <a id="dlResume" href="#" class="btn-download" download>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
                    Resume PDF
                </a>
                <a id="dlCv" href="#" class="btn-download" download>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="12" y1="18" x2="12" y2="12"/><polyline points="9 15 12 18 15 15"/></svg>
                    Cover Letter PDF
                </a>
            </div>
        </div>

        <!-- History Card -->
        <div class="card">
            <div class="card-title"><span class="icon">◷</span> Recent History</div>
            <div id="historyList"></div>
        </div>

        <!-- Footer -->
        <div class="footer-bar">
            <span class="footer-info">Files auto-expire after 24 hours</span>
            <div style="display: flex; gap: 8px;">
                <button onclick="clearHistory()" class="btn-danger">Clear History</button>
                <button onclick="killServer()" class="btn-danger" style="border-color: rgba(248,113,113,0.35);">⏻ Shutdown</button>
            </div>
        </div>
    </div>

<script>
// ── Thinking Box ──
let thinkingLineCount = 0;
function logThinking(msg) {
    const box = document.getElementById("thinkingBox");
    const line = document.createElement("div");
    line.className = "line";
    line.textContent = msg;
    line.style.animationDelay = (thinkingLineCount * 0.05) + "s";
    box.appendChild(line);
    box.scrollTop = box.scrollHeight;
    thinkingLineCount++;
}

function setProgress(percent) {
    document.getElementById("progressBar").style.width = percent + "%";
}

// ── Score Animation ──
function animateScore(score) {
    const circle = document.getElementById("scoreCircle");
    const display = document.getElementById("resScore");
    const circumference = 2 * Math.PI * 40; // r=40
    const offset = circumference - (score / 100) * circumference;
    circle.style.strokeDashoffset = offset;

    // Count up animation
    let current = 0;
    const step = Math.ceil(score / 40);
    const counter = setInterval(() => {
        current = Math.min(current + step, score);
        display.textContent = current + "%";
        if (current >= score) clearInterval(counter);
    }, 30);
}

// ── History ──
async function loadHistory() {
    const res = await fetch('/history');
    const data = await res.json();
    const container = document.getElementById("historyList");

    if (Object.keys(data).length === 0) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">📋</div>No history yet — generate your first resume!</div>';
        return;
    }

    let html = "";
    let i = 0;
    for (const [jobId, item] of Object.entries(data)) {
        const dateStr = new Date(item.timestamp * 1000).toLocaleString();
        html += `
            <div class="history-item" style="animation-delay: ${i * 0.05}s">
                <div>
                    <div class="history-role">${item.role_title} <span style="color: var(--text-muted); font-weight: 400;">@</span> ${item.company_name}</div>
                    <div class="history-meta">
                        <span>${dateStr}</span>
                        <span>·</span>
                        <span class="history-score">${item.match_score}% match</span>
                    </div>
                </div>
                <div class="history-actions">
                    <a href="/download/${item.resume_file}" class="btn-download" download>Resume</a>
                    <a href="/download/${item.cv_file}" class="btn-download" download>CV</a>
                </div>
            </div>
        `;
        i++;
    }
    container.innerHTML = html;
}

// ── Form Submit ──
document.getElementById("processForm").addEventListener("submit", async function(e) {
    e.preventDefault();
    const btn = document.getElementById("submitBtn");
    const btnText = document.getElementById("btnText");
    const btnSpinner = document.getElementById("btnSpinner");
    const progress = document.getElementById("progressContainer");
    const resultsCard = document.getElementById("resultsCard");
    const jdText = document.getElementById("jd").value;

    btn.disabled = true;
    btnText.textContent = "Processing…";
    btnSpinner.style.display = "inline";
    progress.style.display = "block";
    resultsCard.style.display = "none";
    thinkingLineCount = 0;
    document.getElementById("thinkingBox").innerHTML = "";
    document.getElementById("scoreCircle").style.strokeDashoffset = 251.2;

    setProgress(15);
    logThinking("Initializing request…");

    setTimeout(() => { setProgress(30); logThinking("Loading Master Resume data…"); }, 500);
    setTimeout(() => { setProgress(50); logThinking("Analyzing JD requirements & mapping skills…"); }, 1500);
    setTimeout(() => { setProgress(60); logThinking("Pruning & optimizing via Gemini 3.6 Flash…"); }, 3000);

    try {
        const response = await fetch("/process", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({ job_description: jdText })
        });

        setProgress(85);
        logThinking("Rendering strict 1-page layout via Playwright…");

        const result = await response.json();

        setProgress(100);
        logThinking("✓ Tailored documents generated successfully!");

        // Populate results
        document.getElementById("resRole").innerText = result.role_title;
        document.getElementById("resCompany").innerText = result.company_name;

        // Keywords with importance coloring
        const kwContainer = document.getElementById("resKeywords");
        kwContainer.innerHTML = "";
        (result.missing_keywords || []).forEach((kw, idx) => {
            const keyword = typeof kw === 'object' ? kw.keyword : kw;
            const importance = typeof kw === 'object' ? (kw.importance || 'low') : 'low';
            const tagClass = importance === 'high' ? 'tag-high' : importance === 'medium' ? 'tag-medium' : 'tag-low';
            const el = document.createElement("span");
            el.className = "tag " + tagClass;
            el.textContent = keyword;
            el.style.animationDelay = (idx * 0.06) + "s";
            kwContainer.appendChild(el);
        });

        document.getElementById("dlResume").href = "/download/" + result.resume_file;
        document.getElementById("dlCv").href = "/download/" + result.cv_file;

        // Show results with animation
        resultsCard.style.display = "block";
        resultsCard.style.animation = "cardReveal 0.6s var(--spring) both";
        animateScore(result.match_score);

        await loadHistory();

    } catch (err) {
        logThinking("✗ Error: Processing failed. Check console for details.");
        console.error(err);
    } finally {
        btn.disabled = false;
        btnText.textContent = "Generate Resume & Cover Letter";
        btnSpinner.style.display = "none";
    }
});

// ── Server Controls ──
async function killServer() {
    if (!confirm("Shut down the app server completely?")) return;
    try { await fetch('/shutdown', { method: 'POST' }); } catch (e) {}
    document.body.innerHTML = `
        <div style="display:flex; justify-content:center; align-items:center; height:100vh; font-family:'Inter',sans-serif;">
            <div style="text-align:center;">
                <div style="font-size:48px; margin-bottom:16px; opacity:0.6;">⏻</div>
                <h1 style="color:#f87171; font-size:24px; font-weight:800; margin-bottom:8px;">Studio Offline</h1>
                <p style="color:#55556a; font-size:14px;">Port 5000 freed. You can close this tab.</p>
            </div>
        </div>
    `;
}

async function clearHistory() {
    if (!confirm("Clear all history and generated PDFs?")) return;
    await fetch('/clear-history', { method: 'POST' });
    loadHistory();
}

// ── Init ──
loadHistory();
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
