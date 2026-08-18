@echo off
REM lmm -- activate the proof-before-ship guard after a fresh clone.
REM (core.hooksPath is repo-local, so a clone needs this one step on Windows.)
git config core.hooksPath .githooks
if exist .githooks\pre-push ( attrib +x .githooks\pre-push >nul 2>&1 )
echo [lmm] guard hooks active -- every push runs the embedded selftest.
