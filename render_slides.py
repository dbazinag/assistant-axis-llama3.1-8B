"""Render each .slide in slides.html to a 2560x1440 PNG in ./presentation/.
Run: python3 render_slides.py
"""
import pathlib, os
from playwright.sync_api import sync_playwright

OUT = "presentation"
os.makedirs(OUT, exist_ok=True)
url = pathlib.Path("slides.html").resolve().as_uri()

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=2)
    page.goto(url, wait_until="networkidle")
    page.wait_for_timeout(400)  # let webfont settle
    slides = page.query_selector_all(".slide")
    for s in slides:
        name = s.get_attribute("data-name")
        s.screenshot(path=f"{OUT}/{name}.png")
        print("wrote", f"{OUT}/{name}.png")
    browser.close()
