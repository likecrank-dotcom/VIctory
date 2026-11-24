param([Parameter(Mandatory)][string]$RunPath, [int]$MaxImages=24)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
$ring=Join-Path $RunPath 'frames\preview-ring'
$items=@(Get-ChildItem -LiteralPath $ring -Filter '*.json' | ForEach-Object {
    $meta=Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
    [pscustomobject]@{meta=$meta;png=[IO.Path]::ChangeExtension($_.FullName,'.png')}
} | Sort-Object {$_.meta.sequence} | Select-Object -Last $MaxImages)
if($items.Count -eq 0){throw 'No archived previews'}
$columns=4; $tileWidth=384; $imageHeight=216; $tileHeight=248
$canvas=[Drawing.Bitmap]::new($columns*$tileWidth, [int]([Math]::Ceiling($items.Count/$columns)*$tileHeight))
$graphics=[Drawing.Graphics]::FromImage($canvas)
$font=[Drawing.Font]::new('Segoe UI',9)
try {
    $graphics.Clear([Drawing.Color]::FromArgb(24,24,24))
    for($i=0;$i -lt $items.Count;$i++){
        $x=($i%$columns)*$tileWidth; $y=[int][Math]::Floor($i/$columns)*$tileHeight
        $source=[Drawing.Image]::FromFile($items[$i].png)
        try {$graphics.DrawImage($source,[Drawing.Rectangle]::new($x,$y,$tileWidth,$imageHeight))}
        finally {$source.Dispose()}
        $m=$items[$i].meta
        $label="seq $($m.sequence) | src $($m.source_frame) | $($m.kind) f$($m.format)"
        $graphics.DrawString($label,$font,[Drawing.Brushes]::White,[single]($x+4),[single]($y+$imageHeight+4))
    }
    $output=Join-Path $RunPath 'preview-contact.png'
    $canvas.Save($output,[Drawing.Imaging.ImageFormat]::Png)
    $output
} finally {$font.Dispose();$graphics.Dispose();$canvas.Dispose()}
