"""Assemble accepted sign glosses into phrases without blocking recognition."""

from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence


DEFAULT_IDLE_TIMEOUT_MS = 2_500
DEFAULT_DUPLICATE_WINDOW_MS = 1_400
_WHITESPACE = re.compile(r"\s+")


def normalize_gloss(gloss: str) -> str:
    """Return the stable token form passed to a future sentence translator."""
    if not isinstance(gloss, str):
        raise TypeError("gloss must be a string")
    normalized = _WHITESPACE.sub(" ", gloss.replace("_", " ").strip()).casefold()
    if not normalized:
        raise ValueError("gloss must not be blank")
    return normalized


def format_glosses(glosses: Sequence[str], *, final: bool) -> str:
    """Format a cleaned sequence for display when no LLM provider is configured."""
    text = " ".join(glosses).strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if final and text[-1] not in ".!?":
        text += "."
    return text


@dataclass(frozen=True)
class PhraseAppend:
    """Result of offering one recognized gloss to the phrase state machine."""

    appended: bool
    glosses: tuple[str, ...]
    text: str
    idle_timeout_ms: int


class PhraseAssembler:
    """Pure phrase state machine driven by caller-supplied monotonic timestamps."""

    def __init__(
        self,
        *,
        idle_timeout_ms: int = DEFAULT_IDLE_TIMEOUT_MS,
        duplicate_window_ms: int = DEFAULT_DUPLICATE_WINDOW_MS,
    ):
        if idle_timeout_ms <= 0:
            raise ValueError("idle_timeout_ms must be positive")
        if duplicate_window_ms < 0:
            raise ValueError("duplicate_window_ms cannot be negative")
        self.idle_timeout_ms = int(idle_timeout_ms)
        self.duplicate_window_ms = int(duplicate_window_ms)
        self._glosses: list[str] = []
        self._started_ms: int | None = None
        self._last_append_ms: int | None = None

    @property
    def glosses(self) -> tuple[str, ...]:
        return tuple(self._glosses)

    @property
    def empty(self) -> bool:
        return not self._glosses

    def append(self, gloss: str, timestamp_ms: int) -> PhraseAppend:
        cleaned = normalize_gloss(gloss)
        timestamp_ms = int(timestamp_ms)
        duplicate = (
            bool(self._glosses)
            and cleaned == self._glosses[-1]
            and self._last_append_ms is not None
            and 0 <= timestamp_ms - self._last_append_ms <= self.duplicate_window_ms
        )
        if not duplicate:
            if not self._glosses:
                self._started_ms = timestamp_ms
            self._glosses.append(cleaned)
            self._last_append_ms = timestamp_ms
        return PhraseAppend(
            appended=not duplicate,
            glosses=self.glosses,
            text=format_glosses(self._glosses, final=False),
            idle_timeout_ms=self.idle_timeout_ms,
        )

    def remaining_ms(self, timestamp_ms: int) -> int | None:
        if self._last_append_ms is None:
            return None
        return max(0, self.idle_timeout_ms - (int(timestamp_ms) - self._last_append_ms))

    def is_due(self, timestamp_ms: int) -> bool:
        remaining = self.remaining_ms(timestamp_ms)
        return remaining is not None and remaining <= 0

    def age_ms(self, timestamp_ms: int) -> int | None:
        if self._started_ms is None:
            return None
        return max(0, int(timestamp_ms) - self._started_ms)

    def finalize(self) -> tuple[str, ...] | None:
        if not self._glosses:
            return None
        glosses = self.glosses
        self._glosses.clear()
        self._started_ms = None
        self._last_append_ms = None
        return glosses

    def clear(self) -> bool:
        had_phrase = bool(self._glosses)
        self._glosses.clear()
        self._started_ms = None
        self._last_append_ms = None
        return had_phrase


@dataclass(frozen=True)
class TranslationResult:
    """Text returned by a sentence translator and whether a provider rewrote it."""

    text: str
    translated: bool


class SentenceTranslator(Protocol):
    """Provider boundary for turning an ordered cleaned gloss list into English."""

    def translate(self, glosses: Sequence[str]) -> TranslationResult:
        """Translate glosses without mutating their order or contents."""


class FallbackSentenceTranslator:
    """No-network translator used until an LLM provider is configured."""

    def translate(self, glosses: Sequence[str]) -> TranslationResult:
        return TranslationResult(format_glosses(glosses, final=True), translated=False)


@dataclass(frozen=True)
class _Command:
    action: str
    gloss: str | None = None
    timestamp_ms: int | None = None


class PhraseAssemblyWorker:
    """Own phrase state on a worker thread and emit renderer-friendly events."""

    def __init__(
        self,
        emit: Callable[[dict], None],
        *,
        translator: SentenceTranslator | None = None,
        idle_timeout_ms: int = DEFAULT_IDLE_TIMEOUT_MS,
        duplicate_window_ms: int = DEFAULT_DUPLICATE_WINDOW_MS,
        clock_ms: Callable[[], int] | None = None,
    ):
        self._emit = emit
        self._translator = translator or FallbackSentenceTranslator()
        self._fallback = FallbackSentenceTranslator()
        self._assembler = PhraseAssembler(
            idle_timeout_ms=idle_timeout_ms,
            duplicate_window_ms=duplicate_window_ms,
        )
        self._clock_ms = clock_ms or (lambda: time.monotonic_ns() // 1_000_000)
        self._commands: queue.Queue[_Command] = queue.Queue()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._assembler.clear()
        self._commands = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            name="iris-phrase-assembler",
            daemon=True,
        )
        self._thread.start()

    def submit(self, gloss: str, timestamp_ms: int | None = None) -> bool:
        if not self.running:
            return False
        self._commands.put(
            _Command(
                "append",
                gloss=gloss,
                timestamp_ms=self._clock_ms() if timestamp_ms is None else int(timestamp_ms),
            )
        )
        return True

    def finalize(self) -> bool:
        if not self.running:
            return False
        self._commands.put(_Command("finalize"))
        return True

    def clear(self) -> bool:
        if not self.running:
            return False
        self._commands.put(_Command("clear"))
        return True

    def stop(self) -> None:
        if not self.running:
            self._assembler.clear()
            self._thread = None
            return
        self._commands.put(_Command("stop"))
        self._thread.join(timeout=5)
        self._thread = None

    def _wait_seconds(self) -> float | None:
        remaining = self._assembler.remaining_ms(self._clock_ms())
        if remaining is None:
            return None
        return max(remaining / 1_000, 0.001)

    def _run(self) -> None:
        while True:
            try:
                command = self._commands.get(timeout=self._wait_seconds())
            except queue.Empty:
                self._finalize("idle")
                continue

            if command.action == "stop":
                if self._assembler.clear():
                    self._emit({"event": "phrase", "state": "cleared"})
                return
            if command.action == "clear":
                if self._assembler.clear():
                    self._emit({"event": "phrase", "state": "cleared"})
                continue
            if command.action == "finalize":
                self._finalize("manual")
                continue
            if command.action != "append" or command.gloss is None:
                continue

            try:
                update = self._assembler.append(
                    command.gloss,
                    self._clock_ms()
                    if command.timestamp_ms is None
                    else command.timestamp_ms,
                )
            except (TypeError, ValueError):
                continue
            if update.appended:
                self._emit(
                    {
                        "event": "phrase",
                        "state": "building",
                        "glosses": list(update.glosses),
                        "text": update.text,
                        "idleTimeoutMs": update.idle_timeout_ms,
                    }
                )

    def _finalize(self, reason: str) -> None:
        assembly_latency_ms = self._assembler.age_ms(self._clock_ms())
        glosses = self._assembler.finalize()
        if not glosses:
            return
        translation_error = None
        try:
            result = self._translator.translate(glosses)
            if not isinstance(result, TranslationResult) or not result.text.strip():
                raise ValueError("translator returned no text")
        except Exception as error:  # noqa: BLE001 - fallback keeps captions available
            result = self._fallback.translate(glosses)
            translation_error = f"{type(error).__name__}: {error}"
        event = {
            "event": "phrase",
            "state": "final",
            "glosses": list(glosses),
            "sourceText": " ".join(glosses),
            "text": result.text.strip(),
            "translated": bool(result.translated),
            "reason": reason,
            "assemblyLatencyMs": assembly_latency_ms,
        }
        if translation_error:
            event["translationError"] = translation_error
        self._emit(event)
