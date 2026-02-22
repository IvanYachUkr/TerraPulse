import math

def get_epsg(lon, lat):
    zone = math.floor((lon + 180) / 6) + 1
    return 32600 + zone

locations = [
    ("paris_dense_center", 2.3522, 48.8566, "Extreme Built-Up: The sprawling, dense Haussmann-style urban center of Paris, very little green space."),
    ("athens_med_urban", 23.7275, 37.9838, "Extreme Built-Up: Massive sprawling Mediterranean white-roof concrete jungle, highly reflective."),
    ("el_ejido_greenhouses", -2.8104, 36.7731, "Extreme Artificial: The 'Sea of Plastic' in Almeria, world's largest dense concentration of greenhouses. Looks like a solid white block from space, critical for differentiating artificial structures from agriculture.")
]

print("=== NEW BBOXES FROM BROWSER EXPLORATION ===\\n")
for name, clon, clat, desc in locations:
    # Approx 25x15 km bounding box
    lon_span = 0.25
    lat_span = 0.14
    
    west = round(clon - lon_span/2, 2)
    east = round(clon + lon_span/2, 2)
    south = round(clat - lat_span/2, 2)
    north = round(clat + lat_span/2, 2)
    
    epsg = get_epsg(clon, clat)
    
    lat_tile = math.floor(clat / 3) * 3
    lon_tile = math.floor(clon / 3) * 3
    
    ns = f"N{lat_tile:02d}" if lat_tile >= 0 else f"S{abs(lat_tile):02d}"
    ew = f"E{lon_tile:03d}" if lon_tile >= 0 else f"W{abs(lon_tile):03d}"
    wc_tile = f"{ns}{ew}"
    
    bbox_str = f"[{west:.2f}, {south:.2f}, {east:.2f}, {north:.2f}]"
    
    print(f"CityConfig(\"{name}\", {bbox_str}, {epsg}, \"{wc_tile}\"),")
    print(f"# WHY: {desc}\\n")
