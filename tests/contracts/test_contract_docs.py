"""Documentation contracts: the README, reports and guides link only to things that exist.

Every relative Markdown link or image must name an existing file or directory
inside the repository, and every ``#anchor`` (same-file or cross-file) must
match a heading slug of its target under GitHub's rules.  The README keeps its
length budget and the stub headings that older links point at.  Pure text
parsing plus at most one ``git check-ignore`` call, so the module runs in well
under a second.
"""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import unicodedata
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[2]
README_MAX_LINES = 550
# Anchors quoted outside the checked documents (older links, docstrings such as
# scripts/plot_diagnostics.py, notebook cells) that must keep resolving in README.md.
STABLE_README_ANCHORS = (
    "quick-start", "method", "normalized-residual-loss-ve", "noise-gated-gaussian-residuals-ve",
    "log-axis-gate-design", "reproduce-the-experiments", "understand-the-mechanism",
    "gmm-loss-comparison", "gmm-gated-comparison", "gmm-plateau-comparison",
    "gmm-spectral-gate-comparison", "gmm-linear-and-tanh-gates", "gmm-log-axis-gates",
    "figures", "repository-layout", "development",
)


def documents():
    docs = ["README.md", "notebooks/README.md"]
    docs += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "reports").rglob("*.md"))
    if (ROOT / "experiments/README.md").is_file():
        docs.append("experiments/README.md")
    docs += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "notebooks").glob("*.ipynb"))
    return docs


# ------------------------------------------------------------------ parsing

FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")
SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
CODE_SPAN = re.compile(r"(`+)(.+?)\1")
INLINE_LINK = re.compile(r"\]\(\s*(<[^>]*>|[^)\s]+)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)")
REFERENCE_DEFINITION = re.compile(r"^ {0,3}\[[^\]]+\]:\s*(<[^>]*>|\S+)")
HTML_TARGET = re.compile(r"""\b(?:href|src)\s*=\s*["']([^"']+)["']""")
HTML_ID = re.compile(r"""<[^>]*\b(?:id|name)\s*=\s*["']([^"']+)["']""")
EXTERNAL = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*:|//)")
CODE_LINE_ANCHOR = re.compile(r"^L\d+(?:-L\d+)?$")


def blocks(path):
    """``[(label, lines)]``: one block for Markdown, one per markdown cell for notebooks."""
    if path.suffix == ".ipynb":
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        return [
            (f"cell {i}", "".join(cell["source"]).splitlines())
            for i, cell in enumerate(cells) if cell["cell_type"] == "markdown"
        ]
    return [("line", path.read_text(encoding="utf-8").splitlines())]


def prose_lines(lines):
    """``(index, line)`` outside fenced code blocks."""
    fence = None
    for i, line in enumerate(lines):
        m = FENCE.match(line)
        if fence is None and m:
            fence = m.group(1)
            continue
        if fence is not None:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and not line.strip(" `~"):
                fence = None
            continue
        yield i, line


def heading_text(raw):
    """Rendered text of an inline heading: markup removed, code-span contents kept."""
    parts = []
    for n, piece in enumerate(CODE_SPAN.split(raw)):
        # split() yields text, fence, code, text, fence, code, ...
        if n % 3 == 1:
            continue
        if n % 3 == 2:
            parts.append(piece.strip())
            continue
        piece = re.sub(r"<[^>]+>", "", piece)
        piece = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", piece)
        piece = re.sub(r"(\*\*|__)(.+?)\1", r"\2", piece)
        piece = re.sub(r"\*(\S(?:.*?\S)?)\*", r"\1", piece)
        piece = re.sub(r"(?<!\w)_(\S(?:.*?\S)?)_(?!\w)", r"\1", piece)
        parts.append(piece)
    return "".join(parts)


def github_slug(raw):
    """github-slugger: lowercase, drop punctuation except '-' and '_', spaces become '-'."""
    text = heading_text(raw).strip().lower()
    kept = "".join(ch for ch in text if ch in " -_" or unicodedata.category(ch)[0] in "LNM")
    return kept.replace(" ", "-")


def unique(slug, occurrences):
    """Duplicate headings get -1, -2, ... exactly as github-slugger numbers them."""
    result = slug
    while result in occurrences:
        occurrences[slug] += 1
        result = f"{slug}-{occurrences[slug]}"
    occurrences[result] = 0
    return result


def headings(lines):
    found, previous = [], None
    for i, line in prose_lines(lines):
        atx = ATX.match(line)
        if atx:
            text = re.sub(r"(?:^|[ \t]+)#+[ \t]*$", "", atx.group(2) or "")
            found.append(text)
            previous = None
            continue
        if (
            SETEXT.match(line) and previous is not None and previous.strip()
            and not re.match(r"^ {0,3}(?:[|>*+-]|\d+[.)])", previous)
        ):
            found.append(previous.strip())
            previous = None
            continue
        previous = line
    return found


_anchor_cache = {}


def anchors(path):
    if path not in _anchor_cache:
        occurrences, result = {}, set()
        for _, lines in blocks(path):
            result |= {unique(github_slug(text), occurrences) for text in headings(lines)}
            for _, line in prose_lines(lines):
                result |= set(HTML_ID.findall(line))
        _anchor_cache[path] = result
    return _anchor_cache[path]


def links(path):
    """``(location, target)`` for inline links, images, reference definitions and HTML."""
    for label, lines in blocks(path):
        for i, line in prose_lines(lines):
            where = f"{path.relative_to(ROOT).as_posix()}:{label} {i + 1}" if label != "line" \
                else f"{path.relative_to(ROOT).as_posix()}:{i + 1}"
            text = CODE_SPAN.sub(lambda m: " " * len(m.group(0)), line)
            targets = INLINE_LINK.findall(text) + HTML_TARGET.findall(text)
            definition = REFERENCE_DEFINITION.match(text)
            if definition:
                targets.append(definition.group(1))
            for target in targets:
                yield where, target.strip("<>")


def resolve(doc, target):
    """``(path or None, anchor)`` for a relative target; ``None`` path means the same file."""
    path, _, anchor = target.partition("#")
    path = unquote(path)
    if not path:
        return doc, unquote(anchor)
    return Path(os.path.normpath(doc.parent / path)), unquote(anchor)


def link_problems(doc):
    problems = []
    for where, target in links(doc):
        if EXTERNAL.match(target):
            continue
        if target.startswith("/"):
            problems.append(f"{where}: absolute path {target!r}; use a path relative to the document")
            continue
        path, anchor = resolve(doc, target)
        if not path.is_relative_to(ROOT):
            problems.append(f"{where}: {target!r} leaves the repository")
            continue
        if not path.exists():
            problems.append(f"{where}: {target!r} does not exist")
            continue
        if not anchor:
            continue
        if path.is_file() and path.suffix in (".md", ".ipynb"):
            if anchor not in anchors(path):
                problems.append(
                    f"{where}: {target!r} has no heading slug {anchor!r} in {path.relative_to(ROOT)}"
                )
        elif not (path.is_file() and CODE_LINE_ANCHOR.match(anchor)):
            problems.append(f"{where}: {target!r} anchors into a file without headings")
    return problems


# -------------------------------------------------------------------- tests


def test_slug_rules_follow_github():
    assert github_slug("GMM linear and tanh gates") == "gmm-linear-and-tanh-gates"
    assert github_slug("Noise-gated Gaussian residuals (VE)") == "noise-gated-gaussian-residuals-ve"
    assert github_slug("MNIST: remaining normalized and gated comparisons") == (
        "mnist-remaining-normalized-and-gated-comparisons"
    )
    assert github_slug("CIFAR-10: scalar versus Fourier covariance") == "cifar-10-scalar-versus-fourier-covariance"
    assert github_slug("The `fourier_score/gmm.py` *shared* **code**") == "the-fourier_scoregmmpy-shared-code"
    assert github_slug("Error at $\\lambda=1$ & σ") == "error-at-lambda1--σ"
    assert github_slug("[Linked](other.md) snake_case __bold__") == "linked-snake_case-bold"
    occurrences = {}
    assert [unique(s, occurrences) for s in ("a", "a", "a-1", "a")] == ["a", "a-1", "a-1-1", "a-2"]
    lines = ["# Title #", "```bash", "# not a heading", "```", "Setext", "---", "## Title"]
    assert headings(lines) == ["Title", "Setext", "Title"]


def test_readme_line_budget():
    count = len((ROOT / "README.md").read_text(encoding="utf-8").splitlines())
    assert count <= README_MAX_LINES, (
        f"README.md has {count} lines (budget {README_MAX_LINES}); move detail into reports/"
    )


def test_readme_keeps_stable_anchors():
    missing = [a for a in STABLE_README_ANCHORS if a not in anchors(ROOT / "README.md")]
    assert not missing, f"README.md lost headings for anchors {missing}; keep a stub heading"


@pytest.mark.parametrize("doc", documents())
def test_relative_links_and_anchors_resolve(doc):
    problems = link_problems(ROOT / doc)
    assert not problems, "\n".join(problems)


def test_link_targets_are_not_gitignored():
    """A link into saved/, data/ or pretrained/ works locally but not on GitHub."""
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    targets = sorted({
        path.relative_to(ROOT).as_posix()
        for doc in documents()
        for _, target in links(ROOT / doc)
        if not EXTERNAL.match(target) and not target.startswith(("/", "#"))
        for path, _ in [resolve(ROOT / doc, target)]
        if path.is_relative_to(ROOT) and path != ROOT
    })
    result = subprocess.run(
        ["git", "check-ignore", "--stdin", "-z"], cwd=ROOT, capture_output=True,
        input="\0".join(targets).encode(),
    )
    assert result.returncode in (0, 1), result.stderr.decode()
    ignored = [name for name in result.stdout.decode().split("\0") if name]
    assert not ignored, f"documents link to git-ignored paths: {ignored}"


def test_reports_are_indexed_and_link_back():
    reports = ROOT / "reports"
    if not reports.is_dir():
        pytest.skip("no reports/ directory")
    index = reports / "README.md"
    assert index.is_file(), "reports/README.md indexes the reports"
    indexed = {
        path for _, target in links(index) if not EXTERNAL.match(target)
        for path, _ in [resolve(index, target)]
    }
    problems = []
    for report in sorted(reports.rglob("*.md")):
        name = report.relative_to(ROOT).as_posix()
        if report != index and report not in indexed:
            problems.append(f"{name} is not linked from reports/README.md")
        lines = report.read_text(encoding="utf-8").splitlines()
        if not lines or not re.match(r"^# \S", lines[0]):
            problems.append(f"{name} must start with a one-line H1 title")
        head = [target for where, target in links(report) if int(where.rsplit(":", 1)[1]) <= 5]
        if not any(resolve(report, t)[0] == ROOT / "README.md" for t in head):
            problems.append(f"{name} needs a 'Back to README' link right below its title")
    assert not problems, "\n".join(problems)
