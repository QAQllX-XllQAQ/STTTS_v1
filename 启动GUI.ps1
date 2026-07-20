# STTTS GUI 启动脚本 (PowerShell)
# 需要管理员权限（全局键盘钩子）

# 自动提权
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Requesting administrator privileges..." -ForegroundColor Yellow
    Start-Process -FilePath "powershell.exe" -ArgumentList "-ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs -WorkingDirectory "$PSScriptRoot"
    exit
}

Set-Location "$PSScriptRoot"
Write-Host "Running STTTS GUI (as admin)..." -ForegroundColor Green

try {
    python gui.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Exited normally" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Python script crashed with code $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
