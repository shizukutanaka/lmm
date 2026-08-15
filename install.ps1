# lmm installer (Windows, PowerShell). Copies lmm.py to ~\.local\bin and adds
# an alias to your PowerShell $PROFILE if not already present.
$ErrorActionPreference = "Stop"

$src = Join-Path $PSScriptRoot "lmm.py"
$destDir = Join-Path $env:USERPROFILE ".local\bin"
$dest = Join-Path $destDir "lmm.py"
New-Item -ItemType Directory -Force -Path $destDir | Out-Null
Copy-Item $src $dest -Force
Write-Host "installed -> $dest"

$aliasLine = "Set-Alias lmm '$dest'"
if (Test-Path $PROFILE) {
    if (-not (Select-String -Path $PROFILE -Pattern "Set-Alias lmm=" -Quiet)) {
        Add-Content -Path $PROFILE -Value "`n# LMM - Local/remote Model Manager`n$aliasLine"
        Write-Host "added alias to $PROFILE (restart terminal or: . $PROFILE)"
    }
} else {
    Write-Host "no PowerShell profile found; run 'lmm' via: python $dest <cmd>"
}
Write-Host "done. Try: python $dest discover"
