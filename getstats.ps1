# get-download-stats.ps1
$releases = Invoke-RestMethod "https://api.github.com/repos/schlammis/k2p2viewer/releases"
$releases |
  ForEach-Object { $_.assets } |
  Select-Object name, download_count |
  Sort-Object download_count -Descending |
  Format-Table -AutoSize