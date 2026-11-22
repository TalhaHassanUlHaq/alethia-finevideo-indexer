# Submission Checklist

This checklist follows Section 7 of `AI_Engineering_TakeHome_Brief.pdf`.
Complete every pending item only with evidence from the real full 50-shard run.

## Required root deliverables

- [x] `run.py` — required entry point; default invocation remains supported.
- [x] `requirements.txt` — pinned runtime dependencies.
- [x] `README.md` — installation, preparation, default run, optimized run, and
  validation instructions.
- [x] `results.jsonl.gz` — generated from the actual 50-shard output, validated,
  and ready to commit at the repository root.
- [x] `HARDWARE.md` — records the distributed Kaggle T4 environment, observed
  run window, validation counts, and output checksums.
- [ ] `WRITEUP.md` — candidate-owned file is present; verify independently that
  all six required answers are complete and within their word limits. AI agents
  must not read, review, or edit it.
- [ ] `ai_chats/` — add complete exported/shareable Codex and Gemini histories
  to `ai_chats/links.txt` or as native exports; remove only secrets.

## Final manual checks

- [ ] Replace both `PENDING` entries in `ai_chats/links.txt` with the complete
  Codex and Gemini share links or native export filenames.
- [ ] Open each shared link in a private browser window and confirm it works.
- [ ] Confirm no password or API token appears in any tracked file; replace a
  credential in chat history only with `[REMOVED]`.
- [ ] Confirm the GitHub repository is public and the final commit was pushed
  before the assignment deadline.
