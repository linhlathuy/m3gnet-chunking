$names = @{
    "0" = "Tl2SeO4"
    "2" = "K3NbO8"
    "3" = "Nb3RhSe6"
    "4" = "Sc2Te3"
    "5" = "UF5"
    "6" = "Cs2Sb"
    "7" = "Ho2TeO13"
    "8" = "ScMnGe2"
    "9" = "EuIn4"
}

Get-ChildItem -Directory -Filter "m3gnet*" | ForEach-Object {
    $f = $_.FullName
    Write-Host "== $f =="

    Get-ChildItem -Directory -Path $f -Filter "traj*" | ForEach-Object {
        $s = $_.FullName
        Write-Host "-- $s"

        foreach ($idx in $names.Keys) {
            $src = Join-Path $s "$idx.cif"
            $dst = Join-Path $s ($names[$idx] + ".cif")

            if (Test-Path $src) {
                Copy-Item $src $dst
                Write-Host "copied $src -> $dst"
            }
        }
    }
}