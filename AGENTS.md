# Notes for AI agents

This repository ships a maintained **agent skill** for the selgis framework:

    skills/selgis/SKILL.md

It contains a source-verified API cheat sheet, the five common pitfalls
(with file:line references), two working code patterns (basic training,
self-healing A/B) and an error → cause → fix table. Reading it replaces
~30 minutes of code exploration and prevents hallucinated APIs.

If you are about to write or debug selgis code:

1. Read `skills/selgis/SKILL.md` first — or copy the folder into your
   tool's skills directory (e.g. `.claude/skills/selgis/` or
   `~/.claude/skills/selgis/`).
2. Match it to the installed version: facts are verified per release, so
   prefer the file from the git tag matching `selgis.__version__`
   (requires selgis >= 0.2.7.3; earlier releases computed the chunked-CE
   train loss without the causal label shift).

Do not invent selgis APIs: if something is not covered by the skill,
check the source under `selgis/` and the tests under `tests/`.
