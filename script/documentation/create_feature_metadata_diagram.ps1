Add-Type -AssemblyName System.Drawing

$width = 1672
$height = 941
$outDir = Join-Path (Split-Path $PSScriptRoot -Parent) 'deliverables'
$outPath = Join-Path $outDir 'image_features_metadata_diagram_hd.png'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$bmp = New-Object System.Drawing.Bitmap($width, $height)
$bmp.SetResolution(144, 144)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit

function Brush([string]$hex) { return [System.Drawing.SolidBrush]::new([System.Drawing.ColorTranslator]::FromHtml($hex)) }
function Pen([string]$hex, [float]$size) {
    $p = [System.Drawing.Pen]::new([System.Drawing.ColorTranslator]::FromHtml($hex), $size)
    $p.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
    $p.EndCap = [System.Drawing.Drawing2D.LineCap]::Round
    return $p
}
function Pts([object[]]$pairs) {
    $list = [System.Collections.Generic.List[System.Drawing.PointF]]::new()
    foreach ($pair in $pairs) { $list.Add([System.Drawing.PointF]::new([float]$pair[0], [float]$pair[1])) }
    return $list.ToArray()
}
function Poly([object[]]$pairs, [string]$fill, [string]$stroke, [float]$line = 4) {
    $points = Pts $pairs
    $b = Brush $fill
    $p = Pen $stroke $line
    $g.FillPolygon($b, $points)
    $g.DrawPolygon($p, $points)
    $b.Dispose()
    $p.Dispose()
}
function Line([object[]]$pairs, [string]$color, [float]$size = 4) {
    $p = Pen $color $size
    $g.DrawLines($p, (Pts $pairs))
    $p.Dispose()
}
function Txt([string]$text, [float]$x, [float]$y, [float]$size, [string]$color, [string]$weight = 'Regular') {
    $style = if ($weight -eq 'Bold') { [System.Drawing.FontStyle]::Bold } else { [System.Drawing.FontStyle]::Regular }
    $font = [System.Drawing.Font]::new('Segoe UI', $size, $style, [System.Drawing.GraphicsUnit]::Pixel)
    $b = Brush $color
    $g.DrawString($text, $font, $b, $x, $y)
    $font.Dispose()
    $b.Dispose()
}
function ArrowHead([float]$x, [float]$y, [string]$direction, [string]$color) {
    if ($direction -eq 'left') { Poly @(@($x,$y),@(([float]$x + 15),([float]$y - 8)),@(([float]$x + 15),([float]$y + 8))) $color $color 1 }
    if ($direction -eq 'right') { Poly @(@($x,$y),@(([float]$x - 15),([float]$y - 8)),@(([float]$x - 15),([float]$y + 8))) $color $color 1 }
    if ($direction -eq 'up') { Poly @(@($x,$y),@(([float]$x - 8),([float]$y + 15)),@(([float]$x + 8),([float]$y + 15))) $color $color 1 }
    if ($direction -eq 'down') { Poly @(@($x,$y),@(([float]$x - 8),([float]$y - 15)),@(([float]$x + 8),([float]$y - 15))) $color $color 1 }
}

# Background
$g.Clear([System.Drawing.Color]::FromArgb(250, 252, 255))
$bg = Brush '#f1f6fb'
$path = [System.Drawing.Drawing2D.GraphicsPath]::new()
$r = 34
$path.AddArc(18, 18, $r, $r, 180, 90)
$path.AddArc($width-52, 18, $r, $r, 270, 90)
$path.AddArc($width-52, $height-52, $r, $r, 0, 90)
$path.AddArc(18, $height-52, $r, $r, 90, 90)
$path.CloseFigure()
$g.FillPath($bg, $path)
$bg.Dispose()
$path.Dispose()

$navy = '#0e3155'
$dark = '#082543'
$mid = '#284d70'
$edge = '#102f4e'
$pale = '#edf4fa'
$orange = '#f6ab20'
$orangeDark = '#a85b0c'
$dim = '#435668'

# Main three-dimensional feature block
Poly @(@(262,98),@(540,154),@(540,689),@(262,635)) $dark $edge 8
Poly @(@(262,98),@(1082,36),@(1301,101),@(540,154)) '#355877' $edge 8
Poly @(@(540,154),@(1301,101),@(1301,638),@(540,689)) '#f8fbff' $edge 8

# Subtle left face technical plates and gears
Poly @(@(285,215),@(475,251),@(475,543),@(285,506)) '#173c5e' '#476786' 5
Poly @(@(295,235),@(464,267),@(464,310),@(295,278)) '#254a6b' '#476786' 3
$gearPen = Pen '#486883' 6
$gearFill = Brush '#173957'
$g.FillEllipse($gearFill, 335, 319, 110, 110)
$g.DrawEllipse($gearPen, 335, 319, 110, 110)
$g.DrawEllipse($gearPen, 365, 349, 50, 50)
$g.FillEllipse($gearFill, 372, 356, 36, 36)
$g.DrawEllipse($gearPen, 372, 356, 36, 36)
$gearPen.Dispose()
$gearFill.Dispose()
Line @(@(278,540),@(355,555),@(355,623),@(470,645)) '#466582' 5

# Front label on the large feature block
Txt 'Image' 667 220 92 $navy 'Bold'
Txt 'Features' 632 308 92 $navy 'Bold'
Txt '(1000+ dimensions)' 632 413 43 $navy

# Cracks around the compressed metadata region
Line @(@(554,574),@(600,549),@(622,505),@(648,548),@(679,566),@(699,522)) $edge 7
Line @(@(694,574),@(736,537),@(753,477),@(774,520),@(810,548)) $edge 7
Line @(@(811,552),@(850,519),@(871,473),@(892,522),@(937,543),@(958,500)) $edge 7
Line @(@(548,621),@(590,607),@(611,577)) $edge 6
Line @(@(939,578),@(995,563),@(1020,538)) $edge 6

# Metadata cube projecting from the model
Poly @(@(736,611),@(837,571),@(930,613),@(828,659)) '#ffc23d' $orangeDark 6
Poly @(@(736,611),@(828,659),@(828,809),@(736,757)) '#dc8414' $orangeDark 6
Poly @(@(828,659),@(930,613),@(930,760),@(828,809)) $orange $orangeDark 6
Line @(@(777,596),@(868,637),@(868,786)) '#bb6911' 3
Line @(@(772,632),@(828,604),@(885,631)) '#bb6911' 3

# Small fragments below the opening
Poly @(@(680,818),@(704,807),@(728,820),@(712,835),@(684,832)) '#59748c' $edge 3
Poly @(@(616,845),@(639,836),@(650,850),@(632,859)) '#99afc3' $edge 3
Poly @(@(752,851),@(765,844),@(776,855),@(760,861)) $orange $orangeDark 2

# Metadata callout
Line @(@(936,692),@(1093,692)) $orangeDark 5
ArrowHead 1089 692 'right' $orangeDark
Txt 'Metadata' 1113 651 51 '#5c3c18' 'Bold'
Txt '(10 dimensions)' 1113 712 35 '#5c3c18'

# Width dimension
Line @(@(547,116),@(547,79),@(1160,45)) $dim 3
Line @(@(568,80),@(1137,48)) '#8fa0ae' 3
ArrowHead 568 80 'left' '#8fa0ae'
ArrowHead 1137 48 'right' '#8fa0ae'
Txt 'W' 826 39 38 $dim 'Bold'

# Height dimension
Line @(@(235,102),@(205,102),@(205,633),@(237,633)) $dim 3
Line @(@(205,122),@(205,613)) '#8fa0ae' 3
ArrowHead 205 122 'up' '#8fa0ae'
ArrowHead 205 613 'down' '#8fa0ae'
Txt 'H' 159 348 39 $dim 'Bold'

# Depth and compressed dimensions at right
Line @(@(1325,102),@(1363,102),@(1363,636),@(1324,636)) $dim 3
Line @(@(1363,122),@(1363,616)) '#8fa0ae' 3
ArrowHead 1363 122 'up' '#8fa0ae'
ArrowHead 1363 616 'down' '#8fa0ae'
Txt 'D' 1381 338 39 $dim 'Bold'
Line @(@(1328,638),@(1363,638),@(1363,724),@(1329,724)) $dim 3
Line @(@(1363,654),@(1363,708)) '#8fa0ae' 3
ArrowHead 1363 654 'up' '#8fa0ae'
ArrowHead 1363 708 'down' '#8fa0ae'
Txt 'C' 1381 665 34 $dim 'Bold'

$bmp.Save($outPath, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose()
$bmp.Dispose()
Write-Output $outPath
