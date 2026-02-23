# 🛰️ Download Satellite Imagery

Download Sentinel-2 satellite imagery for **any location on Earth** with one command.

## Prerequisites

1. **Python 3.7+** — [python.org](https://python.org)
2. **Docker** — [docker.com/get-started](https://docker.com/get-started) (Docker Desktop on Windows/Mac)

## Step 1: Get the script

Download [`download.py`](https://raw.githubusercontent.com/IvanYachUkr/TerraPulse/main/download.py) — it's a single file, no pip install needed.

**Linux / macOS:**
```bash
curl -O https://raw.githubusercontent.com/IvanYachUkr/TerraPulse/main/download.py
```

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/IvanYachUkr/TerraPulse/main/download.py" -OutFile "download.py"
```

Or just click the link above and save the file.

## Step 2: Download imagery

```bash
python download.py --bbox 10.95 49.38 11.20 49.52 --output ./satellite_data
```

That's it. The script:
- Pulls the TerraPulse Docker image (first time only, ~400 MB)
- Downloads Sentinel-2 + Sentinel-1 seasonal composites
- Saves GeoTIFF files to the folder you specified

## Arguments

| Argument | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `--bbox` | ✅ | — | Bounding box: `west south east north` (WGS84 degrees) |
| `--output` | ✅ | — | Folder to save files (created automatically) |
| `--years` | | `2021` | Year(s) to download |
| `--region` | | `region` | Name used in filenames |

## Examples

```bash
# Nuremberg, Germany (2021)
python download.py --bbox 10.95 49.38 11.20 49.52 --output ./nuremberg

# Paris, France (2023)
python download.py --bbox 2.25 48.81 2.42 48.90 --output ./paris --region paris --years 2023

# Multiple years
python download.py --bbox 10.95 49.38 11.20 49.52 --output ./data --years 2021 2022 2023

# New York City (note: negative longitude)
python download.py --bbox -74.02 40.70 -73.93 40.78 --output ./nyc --region nyc
```

## Finding coordinates

1. Open [Google Maps](https://maps.google.com)
2. Right-click any point → coordinates are copied (lat, lon)
3. The bbox format is `west south east north`:
   - **west** = left edge longitude
   - **south** = bottom edge latitude
   - **east** = right edge longitude
   - **north** = top edge latitude

## Output

The script creates seasonal GeoTIFF composites (spring, summer, autumn):

```
satellite_data/
├── sentinel2_region_2021_spring.tif   # 10-band optical (Spring: Mar–May)
├── sentinel2_region_2021_summer.tif   # 10-band optical (Summer: Jun–Aug)
├── sentinel2_region_2021_autumn.tif   # 10-band optical (Autumn: Sep–Nov)
├── sentinel1_region_2021_spring.tif   # SAR radar (Spring)
├── sentinel1_region_2021_summer.tif   # SAR radar (Summer)
└── sentinel1_region_2021_autumn.tif   # SAR radar (Autumn)
```

Each optical GeoTIFF contains 10 Sentinel-2 bands (B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12) at 10 m resolution, cloud-masked and composited.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Docker is not installed` | Install [Docker Desktop](https://docker.com/get-started) |
| `Docker is not running` | Start Docker Desktop, wait for it to initialize |
| Download is slow | First run downloads the ~400 MB image; subsequent runs are instant |
| No output files | Check the bbox — coordinates might be swapped (west < east, south < north) |
| Permission denied | On Linux, you may need `sudo` or add your user to the `docker` group |
