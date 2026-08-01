"""Vocabulary <-> integer-id codec for the letters/niqqud/dagesh/sin channels,
built from spec["inventories"][name] (either "v1" or "v2")."""

from __future__ import annotations

from typing import Any

from nakdimon.hebrew import HebrewItem


class Inventory:
    def __init__(self, spec_inventory: dict[str, Any]) -> None:
        self.letters: list[str] = spec_inventory["letters"]
        self.letters_encode: dict[str, int] = {c: i for i, c in enumerate(self.letters)}

        self._niqqud_decode: list[str] = spec_inventory["niqqud"]["decode"]
        self._dagesh_decode: list[str] = spec_inventory["dagesh"]["decode"]
        self._sin_decode: list[str] = spec_inventory["sin"]["decode"]

        # Encoding by iterating decode-in-order and overwriting duplicates makes later
        # entries win — e.g. v1's patah appears at decode[9] and decode[15]; encoding
        # always produces 15 (the historic model-compatibility quirk spec.encode_overrides
        # documents). Assert against the spec's overrides so a decode-array edit that
        # silently changes this ordering fails loudly instead of drifting from the model.
        self.niqqud_encode: dict[str, int] = {c: i for i, c in enumerate(self._niqqud_decode)}
        for char, expected_id in spec_inventory["niqqud"].get("encode_overrides", {}).items():
            assert self.niqqud_encode[char] == expected_id, (
                f"niqqud encode_overrides mismatch: {char!r} encodes to "
                f"{self.niqqud_encode[char]}, expected {expected_id}"
            )

        self.dagesh_encode: dict[str, int] = {c: i for i, c in enumerate(self._dagesh_decode)}
        self.sin_encode: dict[str, int] = {c: i for i, c in enumerate(self._sin_decode)}

    def encode_letters(self, normalized: str) -> list[int]:
        return [self.letters_encode[c] for c in normalized]

    def encode_items(self, items: list[HebrewItem]) -> dict[str, list[int]]:
        return {
            "letters": [self.letters_encode[item.normalized] for item in items],
            "niqqud": [self.niqqud_encode[item.niqqud] for item in items],
            "dagesh": [self.dagesh_encode[item.dagesh] for item in items],
            "sin": [self.sin_encode[item.sin] for item in items],
        }

    def decode_niqqud(self, i: int) -> str:
        return self._niqqud_decode[i]

    def decode_dagesh(self, i: int) -> str:
        return self._dagesh_decode[i]

    def decode_sin(self, i: int) -> str:
        return self._sin_decode[i]
