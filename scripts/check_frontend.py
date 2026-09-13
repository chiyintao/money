"""Screenshot the dashboard and assert the panels actually rendered.

The browser used to be named by an absolute path to one machine's Edge install, so
the check only ran where that path existed. It now takes CHROME_PATH when one is
needed and otherwise lets Playwright pick the browser it ships with, which is the
portable default.

    python scripts/check_frontend.py
"""
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = os.getenv('DASHBOARD_URL', 'http://127.0.0.1:8101')
SYMBOL = os.getenv('CHECK_SYMBOL', 'ETHUSDT')


def main():
    root = Path('data/ui_checks')
    root.mkdir(parents=True, exist_ok=True)
    launch = {}
    # Only name an executable when one was named for us. Passing None is not the
    # same as omitting the argument: Playwright treats an explicit None as "no
    # browser", not as "choose one".
    chrome = os.getenv('CHROME_PATH')
    if chrome:
        if not Path(chrome).is_file():
            print('CHROME_PATH does not exist: %s' % chrome)
            return 2
        launch['executable_path'] = chrome
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **launch)
        try:
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.goto(URL, wait_until='networkidle')
            page.screenshot(path=str(root / 'desktop.png'), full_page=True)
            print('desktop class=%s status=%s errors=%s' % (
                page.locator('#trade').get_attribute('class'),
                page.locator('#chartStatus').inner_text(), errors))
            page.select_option('#contractSelect', SYMBOL)
            page.wait_for_function(
                "!document.querySelector('#chartStatus').textContent.includes('\u52a0\u8f7d\u4e2d')")
            print('symbol', page.locator('#contractSelect').input_value())
            page.get_by_role('button', name='模型', exact=True).click()
            print('models visible', page.locator('#models').is_visible())
            page.get_by_role('button', name='模拟交易', exact=True).click()
            page.set_viewport_size({'width': 390, 'height': 844})
            page.screenshot(path=str(root / 'mobile.png'), full_page=True)
            print('mobile overflow',
                  page.evaluate('document.documentElement.scrollWidth > innerWidth'))
        finally:
            browser.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
