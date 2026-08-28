"""Ask a locally-hosted LLM (via Ollama) for a second opinion on ambiguous
import candidates, and display it alongside the normal candidate list.

This plugin never changes beets' own scoring or auto-picks a candidate --
it only prints an advisory line before the candidate list is shown, for
matches that are already ambiguous (not a `strong` recommendation). If the
model says none of the fetched candidates fit, it falls back to a direct
MusicBrainz lookup by barcode/catalog number pulled from the local tags,
since those are exact identifiers rather than fuzzy attributes -- useful
when the right release isn't among the handful beets already fetched.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import requests

from beets import metadata_plugins, ui, util
from beets.autotag import AlbumMatch, Recommendation
from beets.plugins import BeetsPlugin

if TYPE_CHECKING:
    from beets.autotag import AlbumInfo
    from beets.importer import ImportSession, ImportTask

PROMPT_TEMPLATE = """\
You are assisting a music library import tool. Decide which MusicBrainz
candidate release best matches the local track tags below. Consider title,
artist, track count, year, and disambiguation details. If none of the
candidates look right, say so.

Local tags:
{local}

Candidates:
{candidates}

Respond in at most 2 short sentences of reasoning naming the candidate
number and its MBID (or explaining why none fit). Then, on its own final
line, write exactly one of:
PICK: <candidate number>
PICK: NONE
"""

PICK_RE = re.compile(r"^PICK:\s*(\S+)\s*$", re.IGNORECASE | re.MULTILINE)

# Local Item field -> MusicBrainz search field, for the barcode/catalog#
# fallback lookup. Both are exact identifiers, unlike title/artist fuzzy
# matching, so a single hit is trustworthy enough to surface directly.
_IDENTIFIER_SEARCH_FIELDS = {"catalognum": "catno", "barcode": "barcode"}


class LLMatchPlugin(BeetsPlugin):
    def __init__(self) -> None:
        super().__init__()

        self.config.add(
            {
                "host": "http://127.0.0.1:11434",
                "model": "qwen2.5:3b-instruct",
                "threshold": "medium",
                "max_candidates": 5,
                "timeout": 30,
            }
        )

        self.threshold = self.config["threshold"].as_choice(
            {
                "none": Recommendation.none,
                "low": Recommendation.low,
                "medium": Recommendation.medium,
                "strong": Recommendation.strong,
            }
        )

        self.register_listener(
            "import_task_before_choice", self.suggest_candidate
        )

    def suggest_candidate(
        self, session: ImportSession, task: ImportTask
    ) -> None:
        """Print an LLM opinion on ambiguous album candidates.

        Only fires for album tasks whose recommendation is at or below the
        configured threshold (i.e. beets itself isn't confident), and only
        when there's more than one candidate to choose between.
        """
        if task.rec is None or task.rec > self.threshold:
            return

        candidates = [
            c for c in task.candidates or [] if isinstance(c, AlbumMatch)
        ]
        if len(candidates) < 2:
            return

        max_candidates = int(self.config["max_candidates"].as_number())
        candidates = candidates[:max_candidates]

        prompt = self._build_prompt(task, candidates)

        try:
            suggestion = self._ask(prompt)
        except requests.RequestException as exc:
            self._log.warning(
                "llmatch: request to {} failed: {}",
                self.config["host"].as_str(),
                exc,
            )
            return

        if not suggestion:
            return

        pick_match = PICK_RE.search(suggestion)
        reasoning = PICK_RE.sub("", suggestion).strip()
        picked_none = (
            pick_match is not None and pick_match.group(1).lower() == "none"
        )

        ui.print_(ui.colorize("text_highlight", "  LLM opinion:"))
        for line in reasoning.splitlines():
            ui.print_(f"    {line}")

        if picked_none:
            self._suggest_by_identifier(task)

    def _suggest_by_identifier(self, task: ImportTask) -> None:
        """Look up a release directly by barcode/catalog# from local tags.

        Runs only when the LLM found none of the fetched candidates
        convincing. Barcode and catalog number are exact identifiers, so a
        hit here doesn't need another round of LLM judgment.
        """
        mb = metadata_plugins.get_metadata_source("musicbrainz")
        if mb is None:
            return

        for item_field, mb_field in _IDENTIFIER_SEARCH_FIELDS.items():
            values = [v for i in task.items if (v := i.get(item_field))]
            if not values:
                continue
            value, _ = util.plurality(values)

            try:
                hits = mb.mb_api.search(  # type: ignore[attr-defined]
                    "release", {mb_field: value}, limit=3
                )
            except Exception as exc:
                self._log.warning(
                    "llmatch: {} lookup for {!r} failed: {}",
                    item_field,
                    value,
                    exc,
                )
                continue

            for hit in hits:
                info = mb.album_for_id(hit["id"])
                if info is not None:
                    self._print_identifier_match(item_field, value, info)
                    return

    def _print_identifier_match(
        self, item_field: str, value: str, info: AlbumInfo
    ) -> None:
        details = [
            f"{info.artist} - {info.album}",
            f"{len(info.tracks)} tracks",
        ]
        if info.year:
            details.append(str(info.year))
        if info.country:
            details.append(info.country)
        if info.label:
            details.append(info.label)
        details.append(f"MBID {info.album_id}")

        ui.print_(
            ui.colorize(
                "text_highlight",
                f"  Found via {item_field} {value!r} search "
                "(not in the list above):",
            )
        )
        ui.print_(f"    {', '.join(details)}")

    def _build_prompt(
        self, task: ImportTask, candidates: list[AlbumMatch]
    ) -> str:
        local_lines = [
            f"- {item.artist} - {item.album} - {item.title} "
            f"(track {item.track}, year {item.year or 'unknown'}, "
            f"album artist {item.albumartist or 'unknown'})"
            for item in task.items
        ]
        local = (
            f"{len(task.items)} local tracks total:\n" + "\n".join(local_lines)
            if local_lines
            else "(no local tags available)"
        )

        # `disambig_string` already surfaces year/country/label/catalognum/
        # media/data_source per the user's `match.album_disambig_fields`
        # config, so we only add what it doesn't cover.
        candidate_lines = []
        for i, match in enumerate(candidates, start=1):
            info = match.info
            details = [
                f"{info.artist} - {info.album}",
                f"{len(info.tracks)} tracks",
            ]
            if disambig := match.disambig_string:
                details.append(disambig)
            details.append(f"MBID {info.album_id}")
            candidate_lines.append(f"{i}. " + ", ".join(details))
        candidates_block = "\n".join(candidate_lines)

        return PROMPT_TEMPLATE.format(local=local, candidates=candidates_block)

    def _ask(self, prompt: str) -> str | None:
        host = self.config["host"].as_str().rstrip("/")
        model = self.config["model"].as_str()
        timeout = self.config["timeout"].as_number()

        response = requests.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                # Greedy decoding: same input should give the same
                # candidate pick run-to-run, not just similar wording.
                "options": {"temperature": 0},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json().get("response")
