Write-Host "====================================================" -ForegroundColor Cyan
Write-Host " PaperMind AI - Research Intelligence Engine" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "Initializing server and launching browser..." -ForegroundColor Yellow

# 1. Clean up any stale process occupying port 8501
$staleConns = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
if ($staleConns) {
    Write-Host "Clearing previous session on port 8501..." -ForegroundColor DarkGray
    foreach ($c in $staleConns) {
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 600
}

# 2. Trigger browser to open http://localhost:8501 directly
Start-Job -ScriptBlock {
    Start-Sleep -Seconds 1
    Start-Process "http://localhost:8501"
} | Out-Null

# 3. Launch Streamlit
if (Get-Command streamlit -ErrorAction SilentlyContinue) {
    streamlit run app.py --server.port 8501 --server.headless false
} elseif (Test-Path "C:\Users\YASH\anaconda3\python.exe") {
    & "C:\Users\YASH\anaconda3\python.exe" -m streamlit run app.py --server.port 8501 --server.headless false
} else {
    python -m streamlit run app.py --server.port 8501 --server.headless false
}
