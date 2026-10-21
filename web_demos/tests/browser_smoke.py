"""Optional Playwright UI check against a running demo service.

Usage: python tests/browser_smoke.py http://127.0.0.1:7860
Set CHROME_EXECUTABLE when using an existing Chromium installation.
"""
import os
from pathlib import Path
import sys
import tempfile

from playwright.sync_api import sync_playwright


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"
    with tempfile.TemporaryDirectory() as directory, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=os.environ.get("CHROME_EXECUTABLE"), args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url + "/wapr")
        page.wait_for_selector("#backend")
        page.locator("#backend").select_option("trt")
        page.locator("#fx").fill("600")
        page.screenshot(path=str(Path(directory) / "pose-demo-wapr.png"), full_page=True)
        assert page.title() == "WAPR · Pose Lab"
        assert page.locator("#backend").input_value() == "trt"
        assert page.locator("#fx").input_value() == "600"
        page.locator('[name="region"][value="mask"]').check()
        assert page.locator("#w").is_disabled()
        page.locator('[name="region"][value="box"]').check()
        assert page.locator("#w").is_enabled()
        sample = Path(__file__).resolve().parents[1] / "runtime/cache/wapr/cache/samples/pose_lmo/rgb.png"
        if sample.is_file():
            page.locator('[data-file="rgb"]').set_input_files(sample)
            page.locator("#rgb-canvas").wait_for(state="visible")
            bounds = page.locator("#rgb-canvas").bounding_box()
            page.mouse.move(bounds["x"] + .2*bounds["width"], bounds["y"] + .2*bounds["height"])
            page.mouse.down()
            page.mouse.move(bounds["x"] + .7*bounds["width"], bounds["y"] + .7*bounds["height"])
            page.mouse.up()
            width = page.locator("#rgb-canvas").evaluate("canvas => canvas.width")
            assert abs(float(page.locator("#w").input_value()) - .5*width) <= 2
        page.locator("#reset").click()
        assert page.locator("#backend").input_value() == "torch"
        if sample.is_file():
            page.locator("#example").click()
            page.wait_for_function("document.getElementById('notice').textContent.includes('官方样例已载入')")
            assert page.locator('[name="region"][value="mask"]').is_checked()
            assert page.locator("#depth_scale").input_value() == "1"
            assert page.locator("#rgb-canvas").is_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(Path(directory) / "pose-demo-mobile.png"), full_page=True)
        assert not errors, errors
        browser.close()
        print("Browser smoke checks passed: navigation, forms, validation, reset, mobile layout, no JS errors.")


if __name__ == "__main__":
    main()
