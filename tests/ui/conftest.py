"""Fixtures for the Selenium UI tests.

A real uvicorn server is started once per test run on a free port with a
throw-away SQLite database, and each test gets fresh headless browser windows.
Each browser has its own profile, so two browsers act as two group members
(the same as two browser profiles / normal + incognito when testing by hand).

Browser choice: Chrome by default; set UI_BROWSER=edge or UI_BROWSER=firefox to
switch. Set UI_HEADLESS=0 to watch the tests run. If Selenium or the browser is
not available, the UI tests are skipped rather than failing.
"""
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

webdriver = pytest.importorskip("selenium.webdriver")
from selenium.common.exceptions import WebDriverException  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402
from selenium.webdriver.support import expected_conditions as EC  # noqa: E402
from selenium.webdriver.support.ui import WebDriverWait  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WAIT = 15


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_up(url, proc, timeout=60):
    import httpx

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"uvicorn exited early with code {proc.returncode}")
        try:
            if httpx.get(url + "/healthz", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise RuntimeError("uvicorn did not start in time")


@pytest.fixture(scope="session")
def base_url(tmp_path_factory):
    port = _free_port()
    env = {**os.environ, "DB_PATH": str(tmp_path_factory.mktemp("ui") / "ui.db")}
    log = open(tmp_path_factory.getbasetemp() / "uvicorn.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_until_up(url, proc)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def _new_driver():
    kind = os.environ.get("UI_BROWSER", "chrome").lower()
    headless = os.environ.get("UI_HEADLESS", "1") != "0"
    if kind == "firefox":
        opts = webdriver.FirefoxOptions()
        if headless:
            opts.add_argument("-headless")
        driver = webdriver.Firefox(options=opts)
    else:
        opts = webdriver.EdgeOptions() if kind == "edge" else webdriver.ChromeOptions()
        if headless:
            opts.add_argument("--headless=new")
        for arg in ("--no-sandbox", "--disable-dev-shm-usage", "--window-size=1280,1000"):
            opts.add_argument(arg)
        driver = webdriver.Edge(options=opts) if kind == "edge" else webdriver.Chrome(options=opts)
    driver.set_window_size(1280, 1000)
    return driver


class Page:
    """Thin wrapper over a WebDriver with helpers that match index.html's element ids."""

    def __init__(self, driver, base_url):
        self.d = driver
        self.base_url = base_url
        self.wait = WebDriverWait(driver, WAIT)

    def open(self, path="/"):
        self.d.get(self.base_url + path)

    def el(self, id_):
        return self.d.find_element(By.ID, id_)

    def visible(self, id_):
        return self.wait.until(EC.visibility_of_element_located((By.ID, id_)))

    def is_hidden(self, id_):
        return "hidden" in self.el(id_).get_attribute("class").split()

    def type(self, id_, text):
        box = self.visible(id_)
        box.clear()
        box.send_keys(text)

    def click(self, id_):
        self.wait.until(EC.element_to_be_clickable((By.ID, id_))).click()

    def message(self, contains):
        """Wait for the flash message (#msg) to contain the given text and return it."""
        self.wait.until(EC.text_to_be_present_in_element((By.ID, "msg"), contains))
        return self.el("msg").text

    def message_class(self):
        return self.d.find_element(By.CSS_SELECTOR, "#msg .msg").get_attribute("class")

    # ---- multi-step flows -------------------------------------------------
    def register(self, username, password="password123"):
        self.open()
        self.type("u", username)
        self.type("p", password)
        self.click("registerBtn")
        self.visible("groups")

    def create_group(self, name, display_name):
        self.type("gname", name)
        self.type("gdn", display_name)
        self.click("createBtn")
        self.visible("group")
        self.wait.until(lambda d: self.el("gcode").text.strip())
        return self.el("gcode").text.strip()

    def join_group(self, code, display_name):
        self.type("jcode", code)
        self.type("jdn", display_name)
        self.click("joinBtn")
        self.visible("group")

    def vote_buttons(self):
        self.visible("vote")
        self.wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "#vlist .movie")) > 0)
        return self.d.find_elements(By.CSS_SELECTOR, "#vlist .movie")

    def vote_all(self, interested=lambda i: i % 2 == 0):
        """Click Interested/Not interested on every movie. The list re-renders after each
        click, so the buttons are re-located every time."""
        n = len(self.vote_buttons())
        for i in range(n):
            css = ".yes" if interested(i) else ".no"
            self.d.find_elements(By.CSS_SELECTOR, f"#vlist .movie .choices button{css}")[i].click()
        return n


@pytest.fixture()
def browser(base_url):
    """Factory: browser() returns a new Page in its own browser (its own profile/localStorage)."""
    drivers = []

    def make():
        try:
            d = _new_driver()
        except WebDriverException as e:
            pytest.skip(f"Could not start a browser for UI tests: {e.msg or e}")
        drivers.append(d)
        return Page(d, base_url)

    yield make
    for d in drivers:
        d.quit()


@pytest.fixture()
def uname():
    """Unique username factory, so tests sharing one server never collide."""
    return lambda prefix="user": f"{prefix}_{uuid.uuid4().hex[:8]}"
