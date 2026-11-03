#!/usr/bin/env python3
"""Assemble the deterministic Patterns submission folder.

This script builds ``docs/manuscript/patterns/submission/`` from canonical
sources, verifies source integrity against the package checksum file before
copying, and writes a sorted ``MANIFEST.sha256`` describing every file that
was produced. It is idempotent: rerunning with unchanged sources yields an
identical manifest.

Hand-authored submission files (cover letter, highlights, declarations,
checklist) are never touched; only the machine-owned artifacts are cleaned
and rewritten.

Notes
-----
``figures/paper1/figure_manifest.json`` is used as the authoritative figure
registry. Its schema is a JSON object keyed by figure stem, where each entry
exposes repo-relative ``output_pdf`` / ``output_png`` paths plus ``pdf_sha256``
and ``png_sha256`` digests. Because the schema does expose clean output paths,
no filename-pattern fallback is required; a guarded fallback is still kept in
case a future manifest drops those keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

PATTERNS_REL = Path("docs/manuscript/patterns")
CHECKSUMS_REL = PATTERNS_REL / "CHECKSUMS.sha256"
SUBMISSION_REL = PATTERNS_REL / "submission"
MAIN_PDF_REL = PATTERNS_REL / "latex" / "build" / "paper1.pdf"
SUPPLEMENTARY_REL = PATTERNS_REL / "supplementary"
SI_PDF_REL = SUPPLEMENTARY_REL / "build" / "supplementary_information.pdf"
FIGURE_MANIFEST_REL = Path("figures/paper1") / "figure_manifest.json"

MAIN_NAME = "03_manuscript.pdf"
SI_NAME = "04_supplementary.pdf"
MAIN_DIR = "05_figures_main"
SUPP_DIR = "06_figures_supp"
MANIFEST_NAME = "MANIFEST.sha256"

EXPECTED_MAIN_FIGURES = 6
EXPECTED_SUPP_FIGURES = 13

# Files owned by this script; these are the only paths it may clean or write.
OWNED_OUTPUTS = (MAIN_NAME, SI_NAME, MAIN_DIR, SUPP_DIR, MANIFEST_NAME)

# Hand-authored by the parent session; must never be created or removed here.
PROTECTED_OUTPUTS = (
    "00_CHECKLIST.md",
    "01_cover_letter.md",
    "02_highlights_etoc_bigpicture.md",
    "07_declarations.md",
    "08_data_code_availability.md",
)

# Canonical files that must match CHECKSUMS.sha256 before anything is copied.
REQUIRED_CHECKSUMS = (
    MAIN_PDF_REL,
    SI_PDF_REL,
    PATTERNS_REL / "latex" / "paper1.tex",
    PATTERNS_REL / "latex" / "paper1.bib",
    FIGURE_MANIFEST_REL,
)


class AssemblyError(RuntimeError):
    """Raised when integrity checks fail and the build must abort."""


def sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of a file.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Hex-encoded digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksums(path: Path) -> dict[Path, str]:
    """Parse a ``sha256sum``-style checksum file.

    Parameters
    ----------
    path : pathlib.Path
        Checksum file containing ``<sha256>  <relative path>`` lines.

    Returns
    -------
    dict
        Mapping of resolved absolute paths to expected digests.
    """
    entries: dict[Path, str] = {}
    base = path.parent
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        if not rel:
            digest, _, rel = line.partition(" ")
        entries[(base / rel.strip()).resolve()] = digest.strip()
    return entries


def load_figure_registry(
    root: Path,
) -> list[tuple[str, Path, Path, str | None, str | None]]:
    """Load the 19 authoritative figures from the manifest.

    Parameters
    ----------
    root : pathlib.Path
        Repository root.

    Returns
    -------
    list of tuple
        ``(stem, pdf_source, png_source, pdf_hash, png_hash)`` sorted with the
        six main figures first, then supplementary figures by number.
    """
    manifest_path = root / FIGURE_MANIFEST_REL
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssemblyError("figure_manifest.json is not a JSON object")

    registry: list[tuple[str, Path, Path, str | None, str | None]] = []
    for stem, entry in data.items():
        pdf_rel = entry.get("output_pdf") if isinstance(entry, dict) else None
        png_rel = entry.get("output_png") if isinstance(entry, dict) else None
        if not pdf_rel or not png_rel:
            # Fallback: explicit filename patterns under figures/paper1/.
            pdf_rel = (Path("figures/paper1") / f"{stem}.pdf").as_posix()
            png_rel = (Path("figures/paper1") / f"{stem}.png").as_posix()
        registry.append(
            (
                stem,
                root / pdf_rel,
                root / png_rel,
                entry.get("pdf_sha256") if isinstance(entry, dict) else None,
                entry.get("png_sha256") if isinstance(entry, dict) else None,
            )
        )

    registry.sort(key=lambda item: figure_sort_key(item[0]))
    return registry


def is_supplementary(stem: str) -> bool:
    """Return whether a figure stem denotes a supplementary figure.

    Parameters
    ----------
    stem : str
        Figure stem such as ``figure3_x`` or ``figureS11_y``.

    Returns
    -------
    bool
        ``True`` for ``figureS*`` stems.
    """
    return stem[len("figure") :].startswith("S")


def figure_sort_key(stem: str) -> tuple[int, int, str]:
    """Return a deterministic sort key for a figure stem.

    Parameters
    ----------
    stem : str
        Figure stem such as ``figure3_x`` or ``figureS11_y``.

    Returns
    -------
    tuple
        Sort key placing main figures before supplementary ones.
    """
    tail = stem[len("figure") :]
    if tail.startswith("S"):
        digits = tail[1:]
        number = "".join(ch for ch in digits if ch.isdigit())
        return (1, int(number or 0), stem)
    number = "".join(ch for ch in tail if ch.isdigit())
    return (0, int(number or 0), stem)


def verify_checksums(root: Path) -> None:
    """Verify canonical sources against ``CHECKSUMS.sha256``.

    Parameters
    ----------
    root : pathlib.Path
        Repository root.

    Raises
    ------
    AssemblyError
        If any required entry is missing or its digest does not match.
    """
    checksums_path = root / CHECKSUMS_REL
    if not checksums_path.is_file():
        raise AssemblyError(f"missing checksum file: {checksums_path}")
    expected = parse_checksums(checksums_path)

    print("Source integrity (CHECKSUMS.sha256):")
    failures: list[str] = []
    for rel in REQUIRED_CHECKSUMS:
        target = (root / rel).resolve()
        want = expected.get(target)
        if want is None:
            print(f"  MISSING   {rel} (no checksum entry)")
            failures.append(str(rel))
            continue
        if not target.is_file():
            print(f"  MISSING   {rel}")
            failures.append(str(rel))
            continue
        actual = sha256_file(target)
        status = "OK" if actual == want else "MISMATCH"
        print(f"  {status:<9} {rel}")
        if actual != want:
            failures.append(str(rel))
    if failures:
        message = "checksum verification failed for: " + ", ".join(failures)
        raise AssemblyError(message)


def verify_figures(
    root: Path, registry: list[tuple[str, Path, Path, str | None, str | None]]
) -> None:
    """Verify all 19 figures exist and match their manifest digests.

    Parameters
    ----------
    root : pathlib.Path
        Repository root.
    registry : list of tuple
        Output of :func:`load_figure_registry`.

    Raises
    ------
    AssemblyError
        If counts are wrong, files are missing, or digests disagree.
    """
    main_count = sum(1 for item in registry if not is_supplementary(item[0]))
    supp_count = len(registry) - main_count
    print("Figure inventory:")
    print(f"  main figures: {main_count} (expected {EXPECTED_MAIN_FIGURES})")
    print(f"  supp figures: {supp_count} (expected {EXPECTED_SUPP_FIGURES})")

    problems: list[str] = []
    if len(registry) != EXPECTED_MAIN_FIGURES + EXPECTED_SUPP_FIGURES:
        problems.append(f"registry has {len(registry)} figures, expected 19")
    if main_count != EXPECTED_MAIN_FIGURES:
        problems.append(f"main figure count {main_count} != 6")
    if supp_count != EXPECTED_SUPP_FIGURES:
        problems.append(f"supplementary figure count {supp_count} != 13")

    # Guard against a stale registry: the surviving figures must be numbered
    # contiguously and the stems must point at the post-renumbering files.
    expected_supp = {f"figureS{n}" for n in range(1, EXPECTED_SUPP_FIGURES + 1)}
    expected_main = {f"figure{n}" for n in range(1, EXPECTED_MAIN_FIGURES + 1)}
    supp_stems = {
        item[0].split("_", 1)[0] for item in registry if is_supplementary(item[0])
    }
    main_stems = {
        item[0].split("_", 1)[0] for item in registry if not is_supplementary(item[0])
    }
    if supp_stems != expected_supp:
        problems.append(
            "supplementary stems mismatch: "
            f"missing {sorted(expected_supp - supp_stems)}, "
            f"unexpected {sorted(supp_stems - expected_supp)}"
        )
    if main_stems != expected_main:
        problems.append(
            "main stems mismatch: "
            f"missing {sorted(expected_main - main_stems)}, "
            f"unexpected {sorted(main_stems - expected_main)}"
        )

    missing: list[str] = []
    for stem, pdf_src, png_src, pdf_hash, png_hash in registry:
        for src, want in ((pdf_src, pdf_hash), (png_src, png_hash)):
            if not src.is_file():
                missing.append(src.relative_to(root).as_posix())
                continue
            if want and sha256_file(src) != want:
                problems.append(f"figure hash mismatch: {src.name}")
    if missing:
        problems.append("missing figures: " + ", ".join(sorted(missing)))
    if problems:
        raise AssemblyError("; ".join(problems))
    print("  OK        all 19 figures present and hash-matched")


def clean_owned(submission: Path) -> None:
    """Remove only the machine-owned artifacts of a previous build.

    Parameters
    ----------
    submission : pathlib.Path
        Submission directory.
    """
    # PROTECTED_OUTPUTS is intentionally absent from OWNED_OUTPUTS, so the
    # hand-authored parents can never be reached by this cleanup loop.
    for name in OWNED_OUTPUTS:
        target = submission / name
        if target.is_dir():
            resolved = target.resolve()
            if resolved.parent != submission.resolve():
                raise AssemblyError(f"refusing to remove {resolved}")
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def expected_outputs(
    root: Path, registry: list[tuple[str, Path, Path, str | None, str | None]]
) -> dict[str, tuple[Path, str]]:
    """Describe the submission layout that a build would produce.

    Parameters
    ----------
    root : pathlib.Path
        Repository root.
    registry : list of tuple
        Output of :func:`load_figure_registry`.

    Returns
    -------
    dict
        Mapping of submission-relative path to ``(source, source_hash)``.
    """
    expected: dict[str, tuple[Path, str]] = {}
    main_pdf = root / MAIN_PDF_REL
    si_pdf = root / SI_PDF_REL
    expected[MAIN_NAME] = (main_pdf, sha256_file(main_pdf))
    expected[SI_NAME] = (si_pdf, sha256_file(si_pdf))
    for stem, pdf_src, png_src, _pdf_hash, _png_hash in registry:
        directory = SUPP_DIR if is_supplementary(stem) else MAIN_DIR
        expected[f"{directory}/{pdf_src.name}"] = (pdf_src, sha256_file(pdf_src))
        expected[f"{directory}/{png_src.name}"] = (png_src, sha256_file(png_src))
    return expected


def build_manifest(expected: dict[str, tuple[Path, str]]) -> str:
    """Render the sorted ``MANIFEST.sha256`` body.

    Parameters
    ----------
    expected : dict
        Mapping of submission-relative path to ``(source, source_hash)``.

    Returns
    -------
    str
        Two-space separated ``<sha256>  <relative path>`` lines, sorted.
    """
    lines = [f"{expected[rel][1]}  {rel}" for rel in sorted(expected)]
    return "\n".join(lines) + "\n"


def build(root: Path) -> int:
    """Build the submission folder from canonical sources.

    Parameters
    ----------
    root : pathlib.Path
        Repository root.

    Returns
    -------
    int
        Process exit code.
    """
    verify_checksums(root)
    registry = load_figure_registry(root)
    verify_figures(root, registry)
    expected = expected_outputs(root, registry)

    submission = root / SUBMISSION_REL
    submission.mkdir(parents=True, exist_ok=True)
    clean_owned(submission)

    for rel, (source, source_hash) in expected.items():
        destination = submission / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256_file(destination) != source_hash:
            raise AssemblyError(f"copy not byte-identical: {rel}")

    manifest_path = submission / MANIFEST_NAME
    manifest_path.write_text(build_manifest(expected), encoding="utf-8")

    print("\nWritten files:")
    print(f"{'file':<52}{'bytes':>12}  sha256[:12]")
    for rel in sorted(expected):
        destination = submission / rel
        digest = sha256_file(destination)
        size = destination.stat().st_size
        print(f"{rel:<52}{size:>12}  {digest[:12]}")
    m_size = manifest_path.stat().st_size
    m_digest = sha256_file(manifest_path)
    print(f"{MANIFEST_NAME:<52}{m_size:>12}  {m_digest[:12]}")
    print(f"\nsubmission: {submission}")
    return 0


def check(root: Path) -> int:
    """Verify an existing submission folder without writing anything.

    Parameters
    ----------
    root : pathlib.Path
        Repository root.

    Returns
    -------
    int
        ``0`` when everything matches, ``1`` otherwise.
    """
    registry = load_figure_registry(root)
    verify_checksums(root)
    verify_figures(root, registry)
    expected = expected_outputs(root, registry)

    submission = root / SUBMISSION_REL
    print("\nSubmission check:")
    problems: list[str] = []
    if not submission.is_dir():
        print(f"  MISSING   {submission}")
        return 1
    for rel in sorted(expected):
        destination = submission / rel
        if not destination.is_file():
            print(f"  MISSING   {rel}")
            problems.append(rel)
            continue
        actual = sha256_file(destination)
        if actual != expected[rel][1]:
            print(f"  MISMATCH  {rel}")
            problems.append(rel)
        else:
            print(f"  OK        {rel}")

    manifest_path = submission / MANIFEST_NAME
    want_text = build_manifest(expected)
    if not manifest_path.is_file():
        print(f"  MISSING   {MANIFEST_NAME}")
        problems.append(MANIFEST_NAME)
    elif manifest_path.read_text(encoding="utf-8") != want_text:
        print(f"  MISMATCH  {MANIFEST_NAME}")
        problems.append(MANIFEST_NAME)
    else:
        print(f"  OK        {MANIFEST_NAME}")

    for name in PROTECTED_OUTPUTS:
        if (submission / name).exists():
            print(f"  note      hand-authored file present: {name}")

    if problems:
        print(f"\nFAILED: {len(problems)} problem(s)")
        return 1
    print("\nOK: submission matches canonical sources")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to build or check.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Assemble the deterministic Patterns submission folder."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify only, write nothing, exit 1 on any mismatch",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: repo containing this script)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()

    try:
        return check(root) if args.check else build(root)
    except AssemblyError as error:
        print(f"\nABORT: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
