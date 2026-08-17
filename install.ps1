# lmm installer (Windows, PowerShell). Copies lmm.py to ~\.local\bin and adds
# an `lmm` function to your PowerShell $PROFILE if not already present.
$ErrorActionPreference = "Stop"

$src = Join-Path $PSScriptRoot "lmm.py"
$destDir = Join-Path $env:USERPROFILE ".local\bin"
$dest = Join-Path $destDir "lmm.py"
New-Item -ItemType Directory -Force -Path $destDir | Out-Null
Copy-Item $src $dest -Force
Write-Host "installed -> $dest"

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
