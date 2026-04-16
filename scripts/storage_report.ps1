param(
    [string]$Root = ".",
    [int]$TopDirs = 15,
    [int]$TopFiles = 25
)

$ErrorActionPreference = "SilentlyContinue"
$rootPath = (Resolve-Path $Root).Path

Write-Host "Storage Report: $rootPath"
Write-Host ""

Write-Host "Top Directories:"
Get-ChildItem -Path $rootPath -Force | ForEach-Object {
    if ($_.PSIsContainer) {
        $size = (Get-ChildItem -Path $_.FullName -Recurse -Force -File | Measure-Object -Property Length -Sum).Sum
        [PSCustomObject]@{
            Path = $_.FullName
            SizeGB = [math]::Round($size / 1GB, 2)
        }
    }
} | Sort-Object SizeGB -Descending | Select-Object -First $TopDirs | Format-Table -AutoSize

Write-Host ""
Write-Host "Top Large Files:"
Get-ChildItem -Path $rootPath -Recurse -Force -File |
    Sort-Object Length -Descending |
    Select-Object -First $TopFiles |
    Select-Object @{Name = "SizeGB"; Expression = { [math]::Round($_.Length / 1GB, 3) } }, FullName |
    Format-Table -AutoSize

Write-Host ""
Write-Host "Special Targets:"
$special = @(
    ".git",
    ".git\lfs",
    ".git\lfs\tmp",
    "cache",
    "cache\huggingface",
    "output"
)

foreach ($item in $special) {
    $path = Join-Path $rootPath $item
    if (Test-Path $path) {
        $size = (Get-ChildItem -Path $path -Recurse -Force -File | Measure-Object -Property Length -Sum).Sum
        $sizeGb = [math]::Round($size / 1GB, 2)
        Write-Host ("{0,-30} {1,8} GB" -f $item, $sizeGb)
    }
}
