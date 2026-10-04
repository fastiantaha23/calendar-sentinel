# Submission summary

Full name: <<FILL: your full name as on university records>>
Roll number: i230824
Class / section: <<FILL: e.g. BS-AI 5A>>
University email: <<FILL: your university email>>
GitHub username: fastiantaha23
Agent name: Calendar Sentinel
Domain: Deadline / Calendar Sentinel (domain 1) - course outlines and task notes to sandbox calendar entries and reminders

GitHub repository URL: <<FILL: https://github.com/fastiantaha23/REPO-NAME>>
Final commit hash: <<FILL: output of git rev-parse HEAD (40 characters)>>
Working agent interface: <<FILL: https://YOUR-APP.onrender.com/>>
Health endpoint (GET): <<FILL: https://YOUR-APP.onrender.com/health>>
Arena endpoint (POST): <<FILL: https://YOUR-APP.onrender.com/arena/run>>
Manifest endpoint (GET): <<FILL: https://YOUR-APP.onrender.com/arena/manifest>>
API documentation: <<FILL: https://YOUR-APP.onrender.com/docs>>

Hosting provider: Render (free web service, 1 worker)
Default model / provider: <<FILL: the model you deployed after the comparison, e.g. gemini-2.5-flash-lite / Google Gemini API>>
Other available models: <<FILL: e.g. gemini-2.5-flash, or state none>>
Example input: Add 'Project proposal' due on 2026-12-01 to my calendar and remind me 2 days before.
Expected result: status completed; one sandbox calendar entry for 2026-12-01 and a reminder 2 days before; tool trace shows create_entries and add_reminders succeeding.
Cold-start / restart limitations: Free Render instances sleep when idle (first request may take 30-60 s) and restart without notice; chat history is in memory only and is lost on restart; each run uses an isolated in-memory sandbox calendar.
Repository access: <<FILL: Instructor invited (accepted / pending) - GitHub username of instructor: ...>>
Public test results: evaluation/results/public_results.json
