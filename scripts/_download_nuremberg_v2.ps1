# Download Nuremberg dashboard data year-by-year using the latest terrapulse binary.
# Outputs to data/cities/nuremberg_dashboard_v2/raw/
# Run from project root: powershell -File scripts/_download_nuremberg_v2.ps1

$ErrorActionPreference = "Continue"
$binary = ".\terrapulse\target\release\terrapulse.exe"
$anchor = "data\cities\nuremberg_dashboard\anchor_nuremberg_dashboard.tif"
$rawDir = "data\cities\nuremberg_dashboard_v2\raw"

$years = @(2019, 2020, 2021, 2022, 2023, 2024, 2025)

foreach ($year in $years) {
    Write-Host ""
    Write-Host ("=" * 60)
    Write-Host "  Downloading year $year"
    Write-Host ("=" * 60)

    & $binary download `
        --bbox 10.96 49.31 11.30 49.56 `
        --epsg 32632 `
        --years "$year" `
        --region nuremberg_dashboard `
        --raw-dir $rawDir `
        --anchor-ref $anchor

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Year $year failed with exit code $LASTEXITCODE"
    } else {
        Write-Host "  Year $year completed successfully"
    }
}

Write-Host ""
Write-Host ("=" * 60)
Write-Host "  All downloads complete!"
Write-Host ("=" * 60)
