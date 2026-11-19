"""End-to-end UI tests (Selenium) for app/static/index.html.

Each test drives a real browser against a live uvicorn server (see conftest.py).
Requirement ids in the docstrings refer to docs/backlog.csv.

Run only these:   pytest -m ui -v
Skip these:       pytest -m "not ui"
"""
import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

pytestmark = pytest.mark.ui


# --- FR-01: register and log in -------------------------------------------------

def test_register_logs_in_and_shows_groups(browser, uname):
    page = browser()
    name = uname()
    page.register(name)
    assert "Account created" in page.message("Account created")
    assert name in page.el("who").text
    assert page.is_hidden("auth")
    assert "No groups yet" in page.el("glist").text


def test_login_after_logout_and_session_survives_reload(browser, uname):
    page = browser()
    name = uname()
    page.register(name)

    page.d.refresh()                      # token is kept in localStorage
    page.visible("groups")
    assert name in page.el("who").text

    page.d.find_element(By.CSS_SELECTOR, "#who button").click()   # Log out
    page.visible("auth")
    assert page.is_hidden("groups")

    page.type("u", name)
    page.type("p", "password123")
    page.click("loginBtn")
    page.visible("groups")
    assert name in page.el("who").text


# --- FR-15: input validation and error messages ---------------------------------

def test_wrong_password_shows_error(browser, uname):
    page = browser()
    name = uname()
    page.register(name)
    page.d.find_element(By.CSS_SELECTOR, "#who button").click()
    page.visible("auth")

    page.type("u", name)
    page.type("p", "wrong-password")
    page.click("loginBtn")
    assert "Invalid username or password" in page.message("Invalid username or password")
    assert "err" in page.message_class()
    assert not page.is_hidden("auth")


def test_short_password_is_rejected_on_register(browser, uname):
    page = browser()
    page.open()
    page.type("u", uname())
    page.type("p", "short")
    page.click("registerBtn")
    assert "password" in page.message("Invalid input").lower()
    assert page.is_hidden("groups")


def test_duplicate_username_is_rejected(browser, uname):
    name = uname()
    browser().register(name)
    other = browser()
    other.open()
    other.type("u", name)
    other.type("p", "password123")
    other.click("registerBtn")
    assert "already taken" in other.message("already taken")


def test_invalid_join_code_shows_error(browser, uname):
    page = browser()
    page.register(uname())
    page.type("jcode", "ZZZZZZ")
    page.type("jdn", "Me")
    page.click("joinBtn")
    assert "Invalid joining code" in page.message("Invalid joining code")
    assert page.is_hidden("group")


# --- FR-02 / FR-03: groups, join by code or link, unique display names ----------

def test_create_and_join_group_by_code(browser, uname):
    alice, bob = browser(), browser()
    alice.register(uname("alice"))
    code = alice.create_group("Friday crew", "Alice")
    assert alice.el("gtitle").text == "Friday crew"
    assert code in alice.el("glink").text and "?join=" in alice.el("glink").text

    bob.register(uname("bob"))
    bob.join_group(code, "Bob")
    assert "Joined Friday crew" in bob.message("Joined")
    assert "Alice" in bob.el("gmembers").text and "Bob" in bob.el("gmembers").text

    # Group shows up in each member's list with their own display name.
    bob.click("backBtn")
    bob.visible("groups")
    assert "You are Bob" in bob.el("glist").text


def test_join_link_prefills_code(browser, uname):
    alice, bob = browser(), browser()
    alice.register(uname("alice"))
    code = alice.create_group("Link night", "Alice")

    bob.register(uname("bob"))
    bob.open(f"/?join={code}")
    bob.visible("groups")
    bob.wait.until(lambda d: bob.el("jcode").get_attribute("value") == code)
    bob.type("jdn", "Bob")
    bob.click("joinBtn")
    bob.visible("group")
    assert bob.el("gtitle").text == "Link night"


def test_duplicate_display_name_in_group_is_rejected(browser, uname):
    alice, bob = browser(), browser()
    alice.register(uname("alice"))
    code = alice.create_group("Dupes", "Sam")

    bob.register(uname("bob"))
    bob.type("jcode", code)
    bob.type("jdn", "Sam")
    bob.click("joinBtn")
    assert "display name is already used" in bob.message("display name")
    assert bob.is_hidden("group")


# --- FR-04 / FR-05 / FR-06 / FR-07: voting and locking --------------------------

def test_voting_screen_requires_every_movie_before_submit(browser, uname):
    page = browser()
    page.register(uname())
    page.create_group("Solo", "Me")
    page.click("startBtn")

    movies = page.vote_buttons()
    assert len(movies) == 10                                     # FR-04
    genres = {p.text for p in page.d.find_elements(By.CSS_SELECTOR, "#vlist .pill")}
    assert len(genres) >= 5, f"expected a genre-diverse selection, got {genres}"

    submit = page.el("submitVotes")
    assert not submit.is_enabled()
    page.vote_all()
    assert page.el("submitVotes").is_enabled()

    # Toggling a vote keeps exactly one choice highlighted for that movie (FR-05).
    first_no = page.d.find_elements(By.CSS_SELECTOR, "#vlist .movie .choices button.no")[0]
    first_no.click()
    first = page.d.find_elements(By.CSS_SELECTOR, "#vlist .movie")[0]
    on = first.find_elements(By.CSS_SELECTOR, ".choices button.on")
    assert [b.text for b in on] == ["Not interested"]


def test_partial_votes_keep_submit_disabled(browser, uname):
    page = browser()
    page.register(uname())
    page.create_group("Partial", "Me")
    page.click("startBtn")
    n = len(page.vote_buttons())
    for i in range(n - 1):
        page.d.find_elements(By.CSS_SELECTOR, "#vlist .movie .choices button.yes")[i].click()
    assert not page.el("submitVotes").is_enabled()


def test_submitted_votes_lock_and_wait_for_other_members(browser, uname):
    alice, bob = browser(), browser()
    alice.register(uname("alice"))
    code = alice.create_group("Waiting room", "Alice")
    bob.register(uname("bob"))
    bob.join_group(code, "Bob")

    alice.click("startBtn")
    alice.vote_all()
    alice.click("submitVotes")
    alice.visible("wait")
    assert "Waiting for: Bob" in alice.el("waittext").text       # FR-07

    # Re-opening the session keeps Alice on the locked/wait screen, not the voting screen.
    alice.click("refreshBtn")
    alice.visible("wait")
    assert alice.is_hidden("vote")


# --- FR-11 / FR-12 / FR-13: results, choosing the watched movie, rating ---------

def test_full_movie_night_two_members(browser, uname):
    alice, bob = browser(), browser()
    alice.register(uname("alice"))
    code = alice.create_group("Movie night", "Alice")
    bob.register(uname("bob"))
    bob.join_group(code, "Bob")

    alice.click("startBtn")
    alice.vote_all(lambda i: i < 6)
    alice.click("submitVotes")
    alice.visible("wait")

    bob.click("startBtn")                 # joins the same open session
    bob.vote_all(lambda i: i % 3 != 0)
    bob.click("submitVotes")

    # Last member to submit goes straight to the results (FR-11).
    bob.visible("results")
    rows = bob.d.find_elements(By.CSS_SELECTOR, "#rlist .movie")
    assert len(rows) == 10
    ranks = [r.find_element(By.CSS_SELECTOR, ".rank").text for r in rows]
    assert ranks == [f"#{i}" for i in range(1, 11)]
    for r in rows:
        assert r.find_element(By.TAG_NAME, "b").text.strip()                 # title
        assert "group match" in r.text                                       # rating/match line
        assert r.find_element(By.TAG_NAME, "p").text.strip()                 # description
    assert bob.el("rnote").text

    # Alice refreshes from the wait screen and sees the same ranked list.
    alice.click("refreshBtn")
    alice.visible("results")
    alice_titles = [e.text for e in alice.d.find_elements(By.CSS_SELECTOR, "#rlist .movie b")]
    bob_titles = [e.text for e in bob.d.find_elements(By.CSS_SELECTOR, "#rlist .movie b")]
    assert alice_titles == bob_titles

    # FR-12: the group picks the movie it watched.
    chosen = bob_titles[2]
    bob.d.find_elements(By.XPATH, "//div[@id='rlist']//button[normalize-space()='We watched this']")[2].click()
    bob.visible("rate")
    assert bob.el("rtitle").text == chosen
    assert "No ratings yet" in bob.el("rsummary").text
    assert len(bob.d.find_elements(By.XPATH, "//div[@id='rlist']//button[normalize-space()='We watched this']")) == 0

    # FR-13: each member rates 1-5 and the group mean is shown.
    bob.d.find_element(By.XPATH, "//div[@id='stars']/button[normalize-space()='4 ★']").click()
    bob.wait.until(EC.text_to_be_present_in_element(("id", "rsummary"), "from 1 rating"))
    assert "Group average: 4" in bob.el("rsummary").text

    alice.d.refresh()
    alice.visible("groups")
    alice.d.find_element(By.XPATH, "//div[@id='glist']//button[normalize-space()='Open']").click()
    alice.visible("group")
    alice.d.find_element(By.XPATH, "//div[@id='gsessions']//button[normalize-space()='View']").click()
    alice.visible("rate")
    assert alice.el("rtitle").text == chosen
    alice.d.find_element(By.XPATH, "//div[@id='stars']/button[normalize-space()='1 ★']").click()
    alice.wait.until(EC.text_to_be_present_in_element(("id", "rsummary"), "from 2 rating"))
    assert "Group average: 2.5" in alice.el("rsummary").text

    # Rating twice is refused with a clear message (FR-15).
    alice.d.find_element(By.XPATH, "//div[@id='stars']/button[normalize-space()='5 ★']").click()
    assert "already rated" in alice.message("already rated")
