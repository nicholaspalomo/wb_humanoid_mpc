#!/usr/bin/env python3
"""Flags, and can fix, British spellings: this repository is written in American English.

color, behavior, center, initialize, normalize, analyze, modeling, labeled, canceled, defense, gray, program. The words
are matched as whole words and as the parts of a camelCase or snake_case identifier (`normalisedTime`, `kCentreOfMass`),
case preserved, and never inside a URL. A line that has to keep a British spelling - the title of a cited paper, an
external API - carries `NOLINT(american-spelling)`.
"""

import argparse
import os
import re
import sys
from typing import Callable, Iterable, List, NamedTuple, Optional, Tuple

NOLINT = "NOLINT(american-spelling)"

# -our -> -or. Only these stems: "four", "hour", "your", "tour", "pour", "source" and "course" are not British.
_OUR_STEMS = (
    "arm",
    "behavi",
    "clam",
    "col",
    "endeav",
    "fav",
    "flav",
    "harb",
    "hon",
    "hum",
    "lab",
    "neighb",
    "od",
    "parl",
    "rig",
    "rum",
    "sav",
    "splend",
    "tum",
    "val",
    "vap",
    "vig",
)
_OUR_SUFFIXES = (
    "",
    "s",
    "ed",
    "ing",
    "ful",
    "fully",
    "able",
    "ably",
    "ite",
    "ites",
    "al",
    "ally",
    "er",
    "ers",
    "less",
)

# -ise -> -ize, -isation -> -ization. An explicit list: advise, exercise, precise, promise, noise, otherwise and the
# like are spelled -ise in American English too.
_ISE_STEMS = (
    "amort",
    "apolog",
    "author",
    "capital",
    "categor",
    "central",
    "character",
    "critic",
    "custom",
    "deserial",
    "digit",
    "discret",
    "emphas",
    "equal",
    "factor",
    "final",
    "general",
    "harmon",
    "ideal",
    "initial",
    "internal",
    "internation",
    "legal",
    "lineari",
    "linear",
    "local",
    "marginal",
    "material",
    "maxim",
    "memor",
    "minim",
    "mobil",
    "modern",
    "neutral",
    "normal",
    "optim",
    "organ",
    "parameter",
    "parametr",
    "penal",
    "personal",
    "polar",
    "priorit",
    "quant",
    "random",
    "rational",
    "real",
    "recogn",
    "regular",
    "sanit",
    "serial",
    "special",
    "stabil",
    "standard",
    "styl",
    "summar",
    "symmetr",
    "synchron",
    "token",
    "util",
    "vector",
    "visual",
)
_ISE_SUFFIXES = ("e", "ed", "es", "ing", "ation", "ations", "er", "ers", "able")
_YSE_STEMS = ("anal", "catal", "paral")  # analyse -> analyze

# Word for word.
_WORDS = {
    "centre": "center",
    "centres": "centers",
    "centreline": "centerline",
    "centrelines": "centerlines",
    "centred": "centered",
    "centring": "centering",
    "recentre": "recenter",
    "recentres": "recenters",
    "recentred": "recentered",
    "recentring": "recentering",
    "metre": "meter",
    "metres": "meters",
    "millimetre": "millimeter",
    "millimetres": "millimeters",
    "centimetre": "centimeter",
    "centimetres": "centimeters",
    "kilometre": "kilometer",
    "kilometres": "kilometers",
    "litre": "liter",
    "litres": "liters",
    "fibre": "fiber",
    "fibres": "fibers",
    "calibre": "caliber",
    "theatre": "theater",
    "spectre": "specter",
    "lustre": "luster",
    "sombre": "somber",
    "modelling": "modeling",
    "modelled": "modeled",
    "modeller": "modeler",
    "modellers": "modelers",
    "labelled": "labeled",
    "labelling": "labeling",
    "cancelled": "canceled",
    "cancelling": "canceling",
    "travelled": "traveled",
    "travelling": "traveling",
    "traveller": "traveler",
    "signalled": "signaled",
    "signalling": "signaling",
    "levelled": "leveled",
    "levelling": "leveling",
    "totalled": "totaled",
    "totalling": "totaling",
    "fuelled": "fueled",
    "fuelling": "fueling",
    "marshalled": "marshaled",
    "marshalling": "marshaling",
    "channelled": "channeled",
    "channelling": "channeling",
    "dialled": "dialed",
    "dialling": "dialing",
    "equalled": "equaled",
    "equalling": "equaling",
    "tunnelled": "tunneled",
    "focussed": "focused",
    "focussing": "focusing",
    "benefitted": "benefited",
    "benefitting": "benefiting",
    "defence": "defense",
    "defences": "defenses",
    "offence": "offense",
    "offences": "offenses",
    "pretence": "pretense",
    "licence": "license",
    "licences": "licenses",
    "grey": "gray",
    "greys": "grays",
    "programme": "program",
    "programmes": "programs",
    "whilst": "while",
    "amongst": "among",
    "judgement": "judgment",
    "judgements": "judgments",
    "acknowledgement": "acknowledgment",
    "acknowledgements": "acknowledgments",
    "manoeuvre": "maneuver",
    "manoeuvres": "maneuvers",
    "manoeuvring": "maneuvering",
    "catalogue": "catalog",
    "catalogues": "catalogs",
    "analogue": "analog",
    "analogues": "analogs",
    "tyre": "tire",
    "tyres": "tires",
    "aluminium": "aluminum",
    "mould": "mold",
    "moulds": "molds",
    "speciality": "specialty",
    "artefact": "artifact",
    "artefacts": "artifacts",
    "fulfil": "fulfill",
    "fulfils": "fulfills",
    "skilful": "skillful",
    "practise": "practice",
    "practised": "practiced",
    "practising": "practicing",
    "draught": "draft",
    "learnt": "learned",
    "sceptical": "skeptical",
    "plough": "plow",
    "cheque": "check",
    "enrol": "enroll",
    "enrolment": "enrollment",
}


def _build_table() -> dict:
    table = dict(_WORDS)
    for stem in _OUR_STEMS:
        for suffix in _OUR_SUFFIXES:
            table[f"{stem}our{suffix}"] = f"{stem}or{suffix}"
    for stem in _ISE_STEMS:
        for suffix in _ISE_SUFFIXES:
            table[f"{stem}is{suffix}"] = f"{stem}iz{suffix}"
    for stem in _YSE_STEMS:
        # Not "-yses": "analyses" is also the (American) plural of "analysis".
        for suffix in ("e", "ed", "ing", "er", "ers"):
            table[f"{stem}ys{suffix}"] = f"{stem}yz{suffix}"
    return table


def _with_prefixes(table: dict) -> dict:
    """Adds the prefixed forms (uninitialised, decentralised, reorganised, non-normalised, ...)."""
    prefixed = dict(table)
    for prefix in (
        "un",
        "re",
        "de",
        "pre",
        "non",
        "over",
        "under",
        "mis",
        "sub",
        "non-",
        "re-",
        "un-",
        "pre-",
    ):
        for british, american in table.items():
            prefixed.setdefault(prefix + british, prefix + american)
    return prefixed


TABLE = _with_prefixes(_build_table())
# The parts a line is split into: runs of letters, cut at camelCase humps (normalisedTime -> normalised, Time;
# HTMLCentre -> HTML, Centre). Digits, underscores, hyphens and punctuation separate parts too.
_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+")
_URL = re.compile(r"(https?://|www\.)\S+")


def _match_case(original: str, replacement: str) -> str:
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


class Finding(NamedTuple):
    path: str
    line: int
    british: str
    american: str

    def __str__(self) -> str:
        return (
            f"{self.path}:{self.line}: British spelling `{self.british}` - write `{self.american}` (American English "
            f"throughout, see AGENTS.md), or mark the line `{NOLINT}` if it must stay."
        )


def _scan_line(line: str, on_match: Callable[[re.Match, str], None]) -> None:
    urls = [(m.start(), m.end()) for m in _URL.finditer(line)]
    for part in _PART.finditer(line):
        american = TABLE.get(part.group(0).lower())
        if american is None or any(start <= part.start() < end for start, end in urls):
            continue
        on_match(part, _match_case(part.group(0), american))


def check_source(source: str, path: str = "<source>") -> List[Finding]:
    findings: List[Finding] = []
    for number, line in enumerate(source.splitlines(), start=1):
        if NOLINT in line:
            continue
        _scan_line(
            line,
            lambda m, american: findings.append(
                Finding(path, number, m.group(0), american)
            ),
        )
    return findings


def fix_source(source: str) -> Tuple[str, int]:
    """The source with every British spelling replaced, and how many were."""
    count = 0
    out_lines = []
    for line in source.splitlines(keepends=True):
        if NOLINT in line:
            out_lines.append(line)
            continue
        edits: List[Tuple[int, int, str]] = []
        _scan_line(
            line, lambda m, american: edits.append((m.start(), m.end(), american))
        )
        for start, end, american in reversed(edits):
            line = line[:start] + american + line[end:]
        count += len(edits)
        out_lines.append(line)
    return "".join(out_lines), count


def check_files(paths: Iterable[str], root: str) -> List[Finding]:
    findings: List[Finding] = []
    for path in paths:
        with open(path, encoding="utf-8", errors="ignore") as f:
            findings += check_source(f.read(), os.path.relpath(path, root))
    return findings


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--fix", action="store_true", help="rewrite the files in place")
    args = parser.parse_args(argv)
    if args.fix:
        total = 0
        for path in args.paths:
            with open(path, encoding="utf-8") as f:
                source = f.read()
            fixed, count = fix_source(source)
            if count:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(fixed)
                total += count
        print(f"{total} British spelling(s) replaced")
        return 0
    findings = check_files(args.paths, os.getcwd())
    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
