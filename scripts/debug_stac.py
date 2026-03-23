import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pystac_client
import planetary_computer
from config import AOI_BBOX

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

results = catalog.search(collections=["esa-worldcover"], bbox=AOI_BBOX, datetime="2020")
items = list(results.get_items())

if items:
    print(f"Found {len(items)} items")
    for key in items[0].assets.keys():
        print(f"- {key}")
else:
    print("No items found")
