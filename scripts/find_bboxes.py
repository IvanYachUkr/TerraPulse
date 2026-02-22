import math

def get_epsg(lon, lat):
    zone = math.floor((lon + 180) / 6) + 1
    return 32600 + zone

locations = [
    ("tabernas_desert", -2.65, 37.05, "Extreme Bare: True desert in Spain, massive bare soil & badlands."),
    ("bardenas_reales", -1.45, 42.20, "Extreme Bare: Arid badlands in northern Spain, distinct from southern deserts."),
    ("iceland_highlands", -19.35, 64.20, "Extreme Bare: Volcanic rock, glaciers, sparse moss. Totally unique European barren land."),
    ("swiss_alps_high", 7.75, 46.00, "Extreme Bare: High altitude alpine rock and snow above the tree line (near Zermatt)."),
    ("skane_agriculture", 13.30, 55.45, "North Crop: The 'Granary of Sweden'. Massive intensive cold-climate agriculture."),
    ("zemgale_latvia", 23.65, 56.45, "North Crop: Massive flat grain fields in the Baltics, very different soil/climate from Germany."),
    ("jutland_farmland", 8.95, 56.25, "North Crop: Intensive dairy/pig pasture and cropland in Denmark."),
    ("jaen_olives", -3.80, 37.85, "South Crop: The 'Sea of Olives' in Andalusia. Represents tree-crops (olives/almonds) which models often confuse with forest/shrub."),
    ("thessaly_plain", 22.35, 39.45, "South Crop: Massive arid summer agriculture in Greece (cotton, wheat)."),
    ("foggia_wheat", 15.55, 41.45, "South Crop: The breadbasket of Southern Italy (Apulia), endless durum wheat under intense sun."),
    ("danube_delta", 29.45, 45.05, "Unique: Massive wetland, reedbeds, and water mixing. Crucial for Water/Shrub/Grassland distinction."),
    ("lapland_tundra", 27.20, 68.35, "Unique: Subarctic tundra, sparse stunted boreal forest, very short growing season."),
    ("hortobagy_puszta", 21.15, 47.60, "Unique: The Puszta. Massive continuous saline steppe/grassland in Eastern Europe, no trees."),
    ("les_landes_forest", -0.80, 44.15, "Unique: Largest monolithic managed pine forest in Europe on sandy soil.")
]

print("=== NEW TRAINING REGIONS TO ADD ===\\n")
for name, clon, clat, desc in locations:
    # Approx 25x15 km bounding box
    lon_span = 0.25
    lat_span = 0.14
    
    west = round(clon - lon_span/2, 2)
    east = round(clon + lon_span/2, 2)
    south = round(clat - lat_span/2, 2)
    north = round(clat + lat_span/2, 2)
    
    epsg = get_epsg(clon, clat)
    
    # Generate WorldCover tile name
    # Format: N/S lat (floor to 3 deg), E/W lon (floor to 3 deg)
    # Wait, WorldCover tiles are 3x3 degrees.
    # N lat is floor(lat/3)*3. E lon is floor(lon/3)*3.
    lat_tile = math.floor(clat / 3) * 3
    lon_tile = math.floor(clon / 3) * 3
    
    ns = f"N{lat_tile:02d}" if lat_tile >= 0 else f"S{abs(lat_tile):02d}"
    ew = f"E{lon_tile:03d}" if lon_tile >= 0 else f"W{abs(lon_tile):03d}"
    wc_tile = f"{ns}{ew}"
    
    bbox_str = f"[{west:.2f}, {south:.2f}, {east:.2f}, {north:.2f}]"
    
    print(f"CityConfig(\"{name}\", {bbox_str}, {epsg}, \"{wc_tile}\"),")
    print(f"# WHY: {desc}\\n")
