# Copyright (c) 2026, Nicholas Palomo. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Checks for Claudisms in documentation, comments and code (AGENTS.md, "Avoiding Claudisms").

Flags the machine-checkable AI-writing tells listed in AGENTS.md:

- em dashes (`—`), which this repository writes as spaced hyphens (` - `);
- emojis in any file or output;
- bare `---` horizontal section dividers in Markdown (YAML frontmatter at line 1 is exempt);
- `- **Term**: explanation`, `- **Term:** explanation`, `- **Term.** explanation` and `- **Term** - explanation`
  bold-lead list items in Markdown;
- the banned words, phrases and metaphorical constructions in AGENTS.md ("Confirmed Claudisms", "Structural / framing
  tics", "Register tics to avoid" and "Imported from external AI-writing references"), including across line wraps.

A line that must keep a flagged phrase, such as the title of a cited paper, carries `NOLINT(claudisms): <reason>`.
"""

import bisect
import os
import re

from tools.hooks import check_types
from tools.hooks import lint_files

NAME = "claudisms"

# AGENTS.md defines the banlist itself; CODE_OF_CONDUCT.md is upstream Contributor Covenant text; the checker and its
# test list the patterns on purpose.
_OWN_FILES = frozenset(
    {
        "AGENTS.md",
        "CODE_OF_CONDUCT.md",
        "tools/hooks/claudisms.py",
        "tools/hooks/test_claudisms.py",
    }
)

_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2139"
    "\u2600-\u26FF"
    "\u2700-\u2712"
    "\u2714-\u27BF"
    "]"
)

_MD_BOLD_COLON_ITEM = re.compile(
    r"^\s*(?:[-*]|\d+\.)\s+(?:\*\*[^*\n]+\*\*\s*(?:[:.]|\s+-)\s+\S|\*\*[^*:\n]+[:.]\*\*\s+\S)"
)

# (compiled pattern, diagnostic message)
# LINT.IfChange(rules)
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile("\u2014"),
        "em dash `\u2014`: write ` - ` (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        _EMOJI,
        "emoji `{match}`: emojis are banned in all output (AGENTS.md, Avoiding"
        " Claudisms).",
    ),
    (
        re.compile(r"\bsit(?:ting)?\s+with\b", re.IGNORECASE),
        "`{match}`: reflective-pose filler; state the point directly"
        " (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(r"\bworth\s+(?!of\b|per\b)[a-z]+\b", re.IGNORECASE),
        "`{match}`: value-claim filler; state what it does without `worth ...`"
        " (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(r"\bkeep(?:s)?\s+arriving\s+at\b", re.IGNORECASE),
        "`{match}`: pseudo-reflective verb; state the point directly"
        " (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\b(?:keep(?:s)?\s+)?coming\s+back\s+to\b"
            r"|\bbrings\s+(?:me|us)\s+back\s+to\b"
            r"|\bthe\s+question\s+(?:that\s+kept|that\s+keeps|I\s+keep)\b",
            re.IGNORECASE,
        ),
        "`{match}`: pseudo-reflective return framing; state the point directly"
        " (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\bcan't\s+unsee\b|\bdidn't\s+set\s+out\s+to\b"
            r"|\bdidn't\s+plan\s+(?:it|but)\b"
            r"|\bwhat\s+I\s+didn't\s+expect\s+to\s+find\b",
            re.IGNORECASE,
        ),
        "`{match}`: discovery-arc framing; state the observation directly"
        " (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\bsurprised\s+me\s+most\b|\bstruck\s+me\s+hardest\b"
            r"|\bstuck\s+with\s+me\b|\bstayed\s+with\s+me\b"
            r"|\bif\s+I\s+had\s+to\s+pick\s+one\b"
            r"|\bthe\s+best\s+thing\s+you\s+can\s+do\b|\bpays\s+off\s+most\b"
            r"|\bhelps\s+most\b"
            r"|\bthe\s+(?:thing|part)\s+that\s+(?:got|hooked)\s+me\b"
            r"|\bwhat\s+got\s+me\s+was\b|\bit\s+really\s+landed\s+for\s+me\b",
            re.IGNORECASE,
        ),
        "`{match}`: false-singularity / crowned-superlative framing; state the"
        " reason directly (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\bthe\s+physics\s+of\b|\brules\s+of\s+physics\b"
            r"|\bdifferent\s+physics\b",
            re.IGNORECASE,
        ),
        "`{match}`: pseudo-scientific metaphor; state how it works or the"
        " conditions (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\bnames\s+what(?:'s|\s+is)\s+at\s+stake\b"
            r"|\bnaming\s+(?:that|it|the\s+[a-z_-]+)\s+(?:is|changes)\b",
            re.IGNORECASE,
        ),
        "`{match}`: pseudo-elevated `names`/`naming`; state the consequence"
        " directly (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\b(?:quieter|louder)\s+(?:version|claim|disagreement|point"
            r"|argument|move|part|problem|failure|bug)\b"
            r"|\b(?:a|the)\s+second,\s+quieter\b",
            re.IGNORECASE,
        ),
        "`{match}`: metaphorical `quieter`/`louder`; write `less explicit` or"
        " `more direct` (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(r"\bbumped\s+into\b", re.IGNORECASE),
        "`{match}`: write `ran into` (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(r"\bthe\s+shape\s+of\b", re.IGNORECASE),
        "`{match}`: write the structure, dimensions, layout or format directly"
        " (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\bthe\s+engine\s+of\b|\bis\s+the\s+engine\b", re.IGNORECASE
        ),
        "`{match}`: mechanical metaphor; state what drives or produces the"
        " effect directly (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\b(?:hits|lands|strikes)\s+(?:the\s+)?hardest\b", re.IGNORECASE
        ),
        "`{match}`: superlative intensification; state what is most affected"
        " directly (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(r"\bsurfac(?:ed|ing|es)\s+(?:the|a|an)\b", re.IGNORECASE),
        "`{match}`: corporate verb `surface`; write `showed`, `found` or"
        " `exposed` (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\bthat's\s+the\s+tell\b|\bthe\s+tell\s+is\b", re.IGNORECASE
        ),
        "`{match}`: diagnosis diction; state what it shows directly (AGENTS.md,"
        " Avoiding Claudisms).",
    ),
    (
        re.compile(r"\bload[- ]bearing\b", re.IGNORECASE),
        "`{match}`: overused engineering metaphor; state what depends on it"
        " (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\bthe\s+point\s+is\s+(?:that\b|not\b|to\b|:)", re.IGNORECASE
        ),
        "`{match}`: throat-clearing; state the point directly (AGENTS.md,"
        " Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\bhere's\s+where\s+it\s+gets\s+interesting\b"
            r"|\bhere's\s+the\s+(?:analogy|moment|thing|part)\s+that\b"
            r"|\bhere(?:'s|\s+is)\s+where\s+I\s+(?:landed|come\s+down)\b"
            r"|\bwhere\s+I\s+(?:landed|come\s+down|keep\s+landing)\b"
            r"|\bwhat\s+I\s+keep\s+(?:running\s+into|noticing|wanting)\b"
            r"|\bwhat\s+I\s+want\s+(?:you|people)\s+to\b"
            r"|\bleav(?:e|ing)\s+you\s+with\b",
            re.IGNORECASE,
        ),
        "`{match}`: section-announcer / journey framing; state the point"
        " directly (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\b(?:because\s+it|this|and\s+that)\s+matters\b", re.IGNORECASE
        ),
        "`{match}`: presumptuous value claim; state the consequence directly"
        " (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\bthe\s+right\s+(?:time|question|way|answer|tool)\b",
            re.IGNORECASE,
        ),
        "`{match}`: assertion-of-rightness filler; state why it works"
        " (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\bcan't\s+stop\s+thinking\s+about\b"
            r"|\bat\s+the\s+end\s+of\s+the\s+day\b"
            r"|\bdouble-click\s+on\b|\blean(?:ing|s|ed)?\s+(?:into|out|on)\b"
            r"|\bthroughline\b|\blessons\s+learned\b"
            r"|\bstakes\s+of\s+their\s+seat\b|\bparadigm\s+shift\b"
            r"|\bpressure-test\b|\bright-size\b|\bnorth\s+star\b|\btrue\s+north\b"
            r"|\bstrategic\s+imperative\b|\bkey\s+takeaways\b"
            r"|\bthree\s+things\s+to\s+know\b"
            r"|\bseen\s+this\s+(?:movie|one)\s+before\b"
            r"|\bwe(?:'ve|\s+have)\s+(?:been|stood)\s+here\s+before\b"
            r"|\bwe\s+keep\s+meeting\s+it\b"
            r"|\bhit\s+a\s+nerve\b|\bstruck\s+a\s+(?:nerve|chord)\b"
            r"|\bdispatches\s+from\b|\bfield\s+notes\s+from\b"
            r"|\bbest\s+operators\b|\btop\s+practitioners\b"
            r"|\bfirst\s+wave\s+of\s+a\b|\bmulti-year\s+transition\b"
            r"|\bcome(?:s)?\s+along\b",
            re.IGNORECASE,
        ),
        "`{match}`: banned cliche or corporate phrasing (AGENTS.md, Avoiding"
        " Claudisms).",
    ),
    (
        re.compile(
            r"\bthe\s+(?:whole|entire)\s+(?:game|ballgame|point|story|thing"
            r"|lesson|job|idea|reason|purpose|goal|trick|secret|design|content"
            r"|diagnosis|metric)\b"
            r"|\bthe\s+only\s+[a-z_-]+\s+that\s+(?:matters|changed|mattered"
            r"|happened)\b",
            re.IGNORECASE,
        ),
        "`{match}`: totalizing superlative; state what it does at its actual"
        " scope (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\bcarries\s+(?:the|two|three|a)\s+(?:cost|costs|argument|burden"
            r"|risk|penalty)\b"
            r"|\bcarry\s+this\s+with\s+you\b|\bcarry\s+forward\b"
            r"|\bhold\s+that\s+thought\b"
            r"|\bholding\s+them\s+at\s+the\s+same\s+time\b"
            r"|\bin\s+(?:the\s+same|my|the)\s+(?:chest|gut|ribcage)\b",
            re.IGNORECASE,
        ),
        "`{match}`: metaphorical placement/transport; state the effect directly"
        " (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(r"\bturns\s+on\b", re.IGNORECASE),
        "`{match}`: pivot metaphor; write `depends on`, `comes down to` or"
        " `enables` (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(r"\breach(?:es|ed|ing)?\s+for\b", re.IGNORECASE),
        "`{match}`: grasping metaphor; write `uses`, `selects` or state what"
        " happens (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\bdoing\s+(?:the|all\s+the)\s+(?:work|heavy\s+lifting|lifting)\b"
            r"|\bcarrying\s+the\s+argument\b|\bholding\s+the\s+sentence\s+up\b",
            re.IGNORECASE,
        ),
        "`{match}`: abstract-agency metaphor; state how it works in plain"
        " terms (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\bmost\s+people\b|\ba\s+lot\s+of\s+folks\b|\bnobody\s+I\s+know\b"
            r"|\beveryone\s+I've\s+worked\s+with\b"
            r"|\bthe\s+people\s+I've\s+watched\b|\bin\s+my\s+experience\b",
            re.IGNORECASE,
        ),
        "`{match}`: invented-observation framing; state the point without an"
        " unsourced population (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(r"\bquietly\b", re.IGNORECASE),
        "`{match}`: hidden-drama modifier; state what happens or drop the"
        " adverb (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\bunsettled\b"
            r"|\b(?:isn't|is\s+not|not|nobody\s+has|never)\s+settled\b"
            r"|\bsettled\s+(?:question|by\s+the\s+owner)\b",
            re.IGNORECASE,
        ),
        "`{match}`: banned word; state what remains open or decided directly"
        " (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(r"\brid(?:es|ing)\s+(?:along|on)\b", re.IGNORECASE),
        "`{match}`: placement/motion metaphor; write `comes with` or `is"
        " attached` (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\b(?:hands\s+you|hands\s+it\s+back|hands\s+back|nobody\s+handed"
            r"|no\s+one\s+handed|nobody\s+gave\s+you\s+that)\b",
            re.IGNORECASE,
        ),
        "`{match}`: physical-transfer metaphor; write `returns`, `produces` or"
        " `gives` (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\bhand-wav(?:ing|e|ed|es)\b|\ba\s+wave\s+of\s+the\s+hand\b"
            r"|\bwave(?:s|d)?\s+(?:it|the\s+[a-z]+)\s+away\b",
            re.IGNORECASE,
        ),
        "`{match}`: dismissal-by-gesture metaphor; state what went unanswered"
        " (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\bwhat\s+(?:actually\s+)?moved\s+is\b"
            r"|\bwhat\s+changed\s+is\s+the\b"
            r"|\bthe\s+(?:question|conversation|reckoning)\s+"
            r"(?:is\s+coming|is\s+going\s+to\s+happen|will\s+come)\b"
            r"|\bharder\s+to\s+ignore\b",
            re.IGNORECASE,
        ),
        "`{match}`: stakes-gesturing or motion-cleft framing; state the change"
        " directly (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\b(?:stays?\s+yours|the\s+rest\s+is\s+yours"
            r"|yours\s+to\s+(?:keep|own|decide)|the\s+part\s+that\s+stayed)\b",
            re.IGNORECASE,
        ),
        "`{match}`: ownership-retention filler; state what the caller or user"
        " does directly (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\b(?:the\s+useful\s+(?:thing|part)"
            r"|here(?:'s|\s+is)\s+the\s+useful\s+part"
            r"|what(?:'s|\s+is)\s+useful\s+is)\b",
            re.IGNORECASE,
        ),
        "`{match}`: value-announcing filler; state what it does directly"
        " (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\b(?:wreck(?:s|ed|ing)?|shatter(?:s|ed|ing)?"
            r"|obliterat(?:e|es|ed|ing)|blow(?:s|n|ing)?\s+up|blew\s+up)\b",
            re.IGNORECASE,
        ),
        "`{match}`: catastrophizing verb; state the measured effect directly"
        " (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\b(?:a\s+real\s+(?:distinction|improvement|choice|problem"
            r"|difference|issue|question|test|effect|change|cost|gain|benefit)"
            r"|the\s+real\s+(?:thing|problem|way|reason|issue|question|story"
            r"|work|point|trick|test|tools|simulator|SQP|MPCs)"
            r"|real\s+data)\b",
            re.IGNORECASE,
        ),
        "`{match}`: binary `real` framing; name the specific thing instead"
        " (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\bunderscor(?:ing|ed|es\s+the)\b|\bfoster(?:s|ed|ing)?\b"
            r"|\bshed(?:s|ding)?\s+light\s+on\b|\bpav(?:e|es|ed|ing)\s+the\s+way\b"
            r"|\bpivotal\b|\btransformative\b|\bgame-changing\b"
            r"|\bgroundbreaking\b|\bcutting-edge\b|\brobust(?:ness|ly)?\b"
            r"|\bseamless(?:ly)?\b|\bintricate\b|\bcomprehensive(?:ly)?\b"
            r"|\bholistic(?:ally)?\b|\bnavigat(?:e|es|ed|ing)\s+the\b|\brealm\b"
            r"|\btestament\s+to\b|\bdelv(?:e|es|ed|ing)\b"
            r"|\bdive(?:s)?\s+into\b|\bdiving\s+into\b"
            r"|\b(?:to\s+harness|harness(?:es|ed|ing))\b"
            r"|\bleverag(?:e|es|ed|ing)\b|\bmature\b|\bhonest(?:ly)?\b"
            r"|\bessays?\b"
            r"|\bcompound(?:s|ed|ing)\s+(?:over\s+time|into)\b"
            r"|\b(?:advantage|value|trust|benefit|error|effect)\s+compounds\b",
            re.IGNORECASE,
        ),
        "`{match}`: AI-writing vocabulary; use plain, specific wording"
        " (AGENTS.md, Avoiding Claudisms).",
    ),
    (
        re.compile(
            r"\bit(?:'s|\s+is)\s+important\s+to\s+note\b"
            r"|\bwhen\s+it\s+comes\s+to\b|\bat\s+its\s+core\b"
            r"|\bthis\s+is\s+where\s+[^\n.;,]+\s+comes\s+in\b"
            r"|\blet's\s+break\s+it\s+down\b|\bcannot\s+be\s+overstated\b"
            r"|\breflecting\s+a\s+broader\s+trend\b"
            r"|\bmarking\s+a\s+significant\s+shift\b"
            r"|\bgreat\s+question\b|\bI\s+love\s+that\s+you\s+asked\b"
            r"|\bnot\s+only\s+[^,;\n]+,\s+but\b|\bit(?:'s|\s+is)\s+not\s+just\b"
            r"|\bisn't\s+about\s+[^.;\n]+[.;]\s+it's\s+about\b"
            r"|\blet's\s+explore\b|\bnow\s+let's\s+turn\s+to\b"
            r"|\bmoving\s+on\s+to\b|\bin\s+today's\s+rapidly\s+evolving\b",
            re.IGNORECASE,
        ),
        "`{match}`: AI-writing phrase or negative parallelism; state the point"
        " directly (AGENTS.md, Avoiding Claudisms).",
    ),
)
# LINT.ThenChange(//AGENTS.md:claudisms_rules)


def _frontmatter_end(lines: list[str]) -> int:
    """Returns the 1-based line number of a Markdown file's closing YAML frontmatter `---`, or 0 when there is none."""
    if not lines or lines[0].strip() != "---":
        return 0
    for index in range(1, min(len(lines), 25)):
        if lines[index].strip() == "---":
            return index + 1
    return 0


def _line_starts(source: str) -> list[int]:
    """Returns the 0-based character offset where each line of `source` begins."""
    starts = [0]
    for index, char in enumerate(source):
        if char == "\n":
            starts.append(index + 1)
    return starts


def check_source(
    source: str, path: str = "<source>"
) -> list[check_types.Finding]:
    """Returns the Claudism findings in `source`."""
    normalized = lint_files.normalize(path)
    if (
        normalized in _OWN_FILES
        or os.path.splitext(normalized)[1].lower()
        in check_types.MODEL_EXTENSIONS
    ):
        return []
    lines = source.splitlines()
    starts = _line_starts(source)
    is_markdown = normalized.endswith(".md")
    frontmatter_end = _frontmatter_end(lines) if is_markdown else 0
    in_fence = False
    findings: list[check_types.Finding] = []
    for pattern, template in _PATTERNS:
        for match in pattern.finditer(source):
            line_index = bisect.bisect_right(starts, match.start()) - 1
            number = line_index + 1
            column = match.start() - starts[line_index] + 1
            matched_text = " ".join(match.group(0).split())
            findings.append(
                check_types.Finding(
                    path,
                    number,
                    column,
                    NAME,
                    template.format(match=matched_text),
                )
            )
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if is_markdown and stripped.startswith("```"):
            in_fence = not in_fence
        if is_markdown and not in_fence and number > frontmatter_end:
            if stripped == "---":
                findings.append(
                    check_types.Finding(
                        path,
                        number,
                        line.index("---") + 1,
                        NAME,
                        "`---` horizontal rule as section divider: use section"
                        " headers instead (AGENTS.md, Avoiding Claudisms).",
                    )
                )
            elif _MD_BOLD_COLON_ITEM.match(line):
                findings.append(
                    check_types.Finding(
                        path,
                        number,
                        line.index("**") + 1,
                        NAME,
                        "`**Term**: explanation` list format: integrate the"
                        " lead-in into the sentence or use a table (AGENTS.md,"
                        " Avoiding Claudisms).",
                    )
                )
    findings.sort(key=lambda f: (f.line, f.column, f.message))
    return findings


CHECKS = [
    check_types.Check(
        name=NAME,
        languages=frozenset({check_types.Language.TEXT}),
        scope=lint_files.Scope.FIRST_PARTY,
        check_source=check_source,
        description=(
            "no Claudisms: no em dashes, emojis, `---` dividers, `**Term**:`"
            " bullets or banned AI-writing phrases (AGENTS.md)."
        ),
    )
]
