"""Builds the two files Google Classroom needs, from SUBMISSION.md:
    dist/<roll>.zip              (single top-level folder <roll>/, junk and secrets excluded)
    dist/<roll>_submission.pdf   (same details, clickable links)
It refuses to build while any <<FILL ...>> placeholder, localhost URL or malformed hash remains.
Usage (from the project folder):  python scripts/build_submission.py
Requires:  pip install reportlab   (see requirements-dev.txt)"""
import re
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
fields: list[tuple[str, str]] = []
for line in (ROOT / 'SUBMISSION.md').read_text(encoding='utf-8').splitlines():
    m = re.match(r'^([A-Za-z][A-Za-z /()\-]+):\s*(.+)$', line.strip())
    if m:
        fields.append((m.group(1).strip(), m.group(2).strip()))
data = dict(fields)

problems = []
for k, v in fields:
    if '<<' in v:
        problems.append(f'"{k}" still has a placeholder: {v}')
roll = data.get('Roll number', '')
if not re.fullmatch(r'i\d{6}', roll):
    problems.append(f'Roll number "{roll}" must look like i230824 (lowercase, no spaces/hyphens)')
if not re.fullmatch(r'[0-9a-f]{40}', data.get('Final commit hash', '')):
    problems.append('Final commit hash must be the full 40-character hash from `git rev-parse HEAD`')
for k in ('Working agent interface', 'Health endpoint (GET)', 'Arena endpoint (POST)', 'Manifest endpoint (GET)', 'API documentation'):
    v = data.get(k, '')
    if not v.startswith('https://') or re.search(r'localhost|127\.0\.0\.1', v):
        problems.append(f'"{k}" must be a public https URL (not localhost): {v}')
if not data.get('GitHub repository URL', '').startswith('https://github.com/'):
    problems.append('GitHub repository URL must start with https://github.com/')
if problems:
    print('Cannot build yet:\n - ' + '\n - '.join(problems))
    sys.exit(1)

# ---- secret scan --------------------------------------------------------------------------------
EXCLUDE_DIRS = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.pytest_cache', 'dist'}
EXCLUDE_FILES = {'.env'}


def included(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    return path.is_file() and not (set(rel.parts) & EXCLUDE_DIRS) and rel.name not in EXCLUDE_FILES \
        and not rel.name.endswith(('.pyc', '.zip', '.log')) and not (rel.name.startswith('.env.') and rel.name != '.env.example')


key_re = re.compile(r'(AIza[0-9A-Za-z_\-]{30,}|AQ\.[0-9A-Za-z_\-]{30,}|sk-ant-[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9]{32,})')
for f in filter(included, ROOT.rglob('*')):
    if f.suffix in {'.py', '.md', '.json', '.yaml', '.yml', '.txt', '.example', '.js', '.html', '.toml', ''}:
        if key_re.search(f.read_text(encoding='utf-8', errors='ignore')):
            print(f'Looks like a real API key inside {f.relative_to(ROOT)} - remove it first.')
            sys.exit(1)

out = ROOT / 'dist'
out.mkdir(exist_ok=True)

# ---- ZIP ------------------------------------------------------------------------------------------
zpath = out / f'{roll}.zip'
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(filter(included, ROOT.rglob('*'))):
        z.write(f, Path(roll) / f.relative_to(ROOT))
print('wrote', zpath, f'({zpath.stat().st_size // 1024} KB)')

# ---- PDF -------------------------------------------------------------------------------------------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
except ImportError:
    print('reportlab is missing: run  pip install reportlab  and re-run.')
    sys.exit(1)

styles = getSampleStyleSheet()
pdf = out / f'{roll}_submission.pdf'
doc = SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                        title=f'{roll} Agent Arena submission')
story = [Paragraph('Agentic AI - Assignment 1 - Agent Arena: Submission summary', styles['Title']), Spacer(1, 10)]
for k, v in fields:
    value = f'<link href="{escape(v)}" color="blue"><u>{escape(v)}</u></link>' if v.startswith('https://') else escape(v)
    story.append(Paragraph(f'<b>{escape(k)}:</b> {value}', styles['BodyText']))
    story.append(Spacer(1, 4))
doc.build(story)
print('wrote', pdf)
print(f'\nNext: extract {zpath.name} into a different folder and follow README to confirm it runs; then upload BOTH files to Classroom and click Turn in.')
