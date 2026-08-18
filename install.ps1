# lmm installer (Windows, PowerShell).
#
# lmm is three files that import each other — lmm.py (entry), backend.py
# (engine) and frontend.py (CLI + GUI) — so they are installed together into a
# library directory and an `lmm` function is added to your PowerShell $PROFILE.
# Copying only lmm.py, as this script used to, produced an `lmm` that could not
# start: the import of backend.py failed immediately.
$ErrorActionPreference = "Stop"

$files = @("lmm.py", "backend.py", "frontend.py")
foreach ($f in $files) {
    if (-not (Test-Path (Join-Path $PSScriptRoot $f))) {
        Write-Error "install: $f not found next to this script - incomplete checkout?"
    }
}

$libDir = Join-Path $env:USERPROFILE ".local\share\lmm"
New-Item -ItemType Directory -Force -Path $libDir | Out-Null
foreach ($f in $files) {
    Copy-Item (Join-Path $PSScriptRoot $f) (Join-Path $libDir $f) -Force
}
$dest = Join-Path $libDir "lmm.py"
Write-Host "installed -> $libDir"

# A function, not Set-Alias: an alias to a .py file only resolves when PATHEXT
# and the python file association line up, which they usually do not. A
# function always works and forwards arguments faithfully.
$marker = "# LMM - Local/remote Model Manager"
$fnLine = "function lmm { python `"$dest`" @args }"
if (-not (Test-Path $PROFILE)) {
    # First-time PowerShell users have no profile; without creating one they
    # got no command at all.
    New-Item -ItemType File -Force -Path $PROFILE | Out-Null
}
# The guard must match what we actually write. The old script wrote
# "Set-Alias lmm '<path>'" but grepped for "Set-Alias lmm=" (bash-style '='),
# which can never match — so every re-run appended another copy to $PROFILE.
if (-not (Select-String -Path $PROFILE -Pattern $marker -SimpleMatch -Quiet)) {
    Add-Content -Path $PROFILE -Value "`n$marker`n$fnLine"
    Write-Host "added 'lmm' to $PROFILE (restart terminal or: . `$PROFILE)"
} else {
    Write-Host "'lmm' already present in $PROFILE"
}
Write-Host "done. Try: lmm discover"
