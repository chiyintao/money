"""The dashboard is actually served, not just routed.

The root route and the static mount were both registered and both pointed at a
directory that does not exist, so `/` returned 404 and every `/assets/*` request 404'd.
Nothing failed loudly: the process started, `/health` answered 200, and the console was
a blank page. These tests exist because no test touched the root path at all, which is
precisely why a refactor could break it silently.
"""
from pathlib import Path

from app.web import api_routes, web as web_module


def test_web_root_points_at_a_directory_that_exists():
    assert web_module.WEB_ROOT.is_dir(), (
        "WEB_ROOT does not exist, so / would 404: %s" % web_module.WEB_ROOT)


def test_the_root_document_is_present():
    assert (web_module.WEB_ROOT / "index.html").is_file()


def test_the_assets_the_root_document_references_are_present():
    """index.html loads these by absolute path; a missing one is a blank page.

    Checked against the document itself rather than a hardcoded list, so adding a
    script to the page without shipping it fails here instead of in the browser.
    """
    document = (web_module.WEB_ROOT / "index.html").read_text(encoding="utf-8")
    referenced = []
    for token in document.split("/assets/")[1:]:
        referenced.append(token.split('"')[0].split("'")[0].split(">")[0])
    assert referenced, "index.html references no assets, which cannot be right"
    missing = [item for item in referenced if not (web_module.WEB_ROOT / item).exists()]
    assert missing == [], "index.html references missing assets: %s" % missing


def test_web_root_is_not_inside_the_package():
    """Guards the exact regression: parent.parent became wrong after the move.

    While this module lived at `app/web.py`, `parent.parent` was the project root. After
    it moved to `app/web/web.py` the same expression resolved to `app/web/`, where
    nothing is published. If WEB_ROOT is ever inside `app/` again, this fails.
    """
    package = Path(web_module.__file__).resolve().parent
    assert package not in web_module.WEB_ROOT.parents


def test_the_static_mount_serves_the_document_directory():
    """`/assets/index.html` must resolve; it did not while the path was wrong."""
    assert (web_module.WEB_ROOT / "index.html").is_file()
    assert (web_module.WEB_ROOT / "js").is_dir()
    assert (web_module.WEB_ROOT / "css").is_dir()