"""
Generates realistic procedural orbital/satellite imagery for instant testing.
- Coastal Harbour (water, coastline, docks, ships)
- Agricultural Swaths (cropland grids, circular pivot irrigation, field boundaries)
- Urban Airport (runways, taxiways, hangars, road grid)
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate_orbital_samples(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Agricultural Swaths (Pivot irrigation + rectangular plots)
    w, h = 512, 512
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    # Background terrain (sand / dry earth)
    arr[:, :] = [165, 148, 120]

    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    # Grid fields
    colors = [
        (45, 95, 40),   # Deep green crop
        (70, 130, 50),  # Bright green crop
        (120, 140, 60), # Olive field
        (170, 145, 95), # Fallow ground
        (85, 115, 65),  # Medium green
    ]

    for y in range(0, h, 64):
        for x in range(0, w, 64):
            c = colors[(x * 7 + y * 13) % len(colors)]
            draw.rectangle([x + 2, y + 2, x + 62, y + 62], fill=c)

    # Circular pivot irrigation fields
    draw.ellipse([60, 60, 220, 220], fill=(30, 80, 35), outline=(140, 125, 95), width=2)
    draw.ellipse([260, 180, 440, 360], fill=(50, 110, 45), outline=(150, 135, 105), width=2)
    draw.ellipse([80, 300, 240, 460], fill=(80, 125, 55), outline=(130, 120, 90), width=2)

    # Roads / Canals
    draw.line([(0, 256), (w, 256)], fill=(195, 185, 160), width=4)
    draw.line([(256, 0), (256, h)], fill=(195, 185, 160), width=4)
    draw.line([(0, 128), (w, 128)], fill=(60, 95, 130), width=3) # Canal

    img = img.filter(ImageFilter.GaussianBlur(0.6))
    agri_path = os.path.join(output_dir, "sample_agricultural.jpg")
    img.save(agri_path, "JPEG", quality=90)

    # 2. Coastal Harbour (Ocean, coast, docks, boats)
    arr2 = np.zeros((h, w, 3), dtype=np.uint8)
    # Ocean gradient
    for y in range(h):
        blue_val = int(70 + 40 * (y / h))
        arr2[y, :] = [20, 50, blue_val]

    img2 = Image.fromarray(arr2)
    draw2 = ImageDraw.Draw(img2)

    # Landmass
    draw2.polygon([(0, 0), (280, 0), (220, 180), (320, 340), (180, h), (0, h)], fill=(110, 105, 95))

    # Port pier / breakwater
    draw2.rectangle([210, 150, 380, 180], fill=(80, 85, 90))
    draw2.rectangle([360, 150, 380, 290], fill=(75, 80, 85))

    # Ships / vessels
    ships = [(270, 200), (310, 220), (330, 250), (410, 120), (430, 320)]
    for sx, sy in ships:
        draw2.polygon([(sx, sy), (sx + 24, sy + 6), (sx, sy + 12)], fill=(220, 225, 230))
        draw2.line([(sx - 8, sy + 6), (sx - 20, sy + 6)], fill=(150, 190, 210), width=2) # Wake

    # Urban grid near coast
    for uy in range(20, 260, 30):
        for ux in range(20, 160, 30):
            draw2.rectangle([ux, uy, ux + 22, uy + 22], fill=(135, 130, 120))

    img2 = img2.filter(ImageFilter.GaussianBlur(0.5))
    coastal_path = os.path.join(output_dir, "sample_coastal.jpg")
    img2.save(coastal_path, "JPEG", quality=90)

    # 3. Urban Airport (Airfield runways, taxiways, terminals)
    arr3 = np.zeros((h, w, 3), dtype=np.uint8)
    arr3[:, :] = [95, 115, 85] # Grass airfield
    img3 = Image.fromarray(arr3)
    draw3 = ImageDraw.Draw(img3)

    # Main Runway 1
    draw3.line([(40, 80), (470, 430)], fill=(45, 48, 52), width=24)
    # Runway markings (center line dashes)
    for t in range(70, 440, 25):
        ratio = (t - 40) / (470 - 40)
        px = int(40 + ratio * 430)
        py = int(80 + ratio * 350)
        draw3.line([(px - 4, py - 3), (px + 4, py + 3)], fill=(230, 230, 230), width=2)

    # Cross Runway 2
    draw3.line([(60, 420), (450, 120)], fill=(50, 52, 56), width=18)

    # Taxiways
    draw3.line([(100, 70), (480, 380)], fill=(65, 68, 70), width=8)

    # Terminal building and apron
    draw3.polygon([(340, 30), (480, 30), (480, 140), (380, 140)], fill=(160, 165, 170))
    draw3.rectangle([370, 60, 460, 110], fill=(90, 95, 105)) # Terminal roof

    img3 = img3.filter(ImageFilter.GaussianBlur(0.5))
    airport_path = os.path.join(output_dir, "sample_airport.jpg")
    img3.save(airport_path, "JPEG", quality=90)

    print(f"[Samples] Generated orbital test samples in {output_dir}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public", "samples")
    generate_orbital_samples(out)
