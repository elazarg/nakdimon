// GENERATED from spec/hebrew.json — do not edit. Run npm run embed-spec.
export const SPEC = {
  "spec_version": "2.0.0",
  "oracle": "nakdimon 0.2.1 (repo-root package)",
  "mark_order": {
    "emit": "NFC",
    "emit_channel_order": [
      "niqqud",
      "dagesh",
      "sin"
    ],
    "comment": "v2 is NFC-native: emitted mark order is Unicode canonical — niqqud (ccc 10-20) before dagesh (ccc 21) before shin/sin dot (ccc 24/25). Hebrew presentation forms are composition exclusions, so NFC never composes letter+mark; it is a pure reorder and emitted text is an NFC fixpoint. Parsers must accept marks after a base letter in ANY order (last-wins per channel). NOTE: v1 emitted dagesh-sin-niqqud and its strict parser crashes on NFC-ordered text; legacy corpora stay in v1 order until v1 tooling is retired (DESIGN.md Phase 4)."
  },
  "special_tokens": {
    "foreign": "O",
    "ligature": "H",
    "digit": "5"
  },
  "normalization": {
    "pass_through": [
      " ",
      "!",
      "\"",
      "'",
      "(",
      ")",
      ",",
      "-",
      ".",
      ":",
      ";",
      "?",
      "א",
      "ב",
      "ג",
      "ד",
      "ה",
      "ו",
      "ז",
      "ח",
      "ט",
      "י",
      "ך",
      "כ",
      "ל",
      "ם",
      "מ",
      "ן",
      "נ",
      "ס",
      "ע",
      "ף",
      "פ",
      "ץ",
      "צ",
      "ק",
      "ר",
      "ש",
      "ת"
    ],
    "map": {
      "\n": " ",
      "\t": " ",
      "־": "-",
      "‒": "-",
      "–": "-",
      "—": "-",
      "―": "-",
      "−": "-",
      "[": "(",
      "]": ")",
      "´": "'",
      "‘": "'",
      "’": "'",
      "“": "\"",
      "”": "\"",
      "״": "\"",
      "…": ",",
      "ײ": "H",
      "װ": "H",
      "ױ": "H"
    },
    "digit_rule": {
      "comment": "any char in Unicode category Nd maps to the digit token",
      "category": "Nd",
      "to": "5"
    },
    "fallback": "O",
    "known_divergences_from_v1": [
      "v1 used Python str.isdigit(), which also matches some No/Lo chars (e.g. '²', '½' is False but '²' True); v2 restricts the digit rule to category Nd, so '²' now normalizes to 'O' instead of '5'."
    ]
  },
  "inventories": {
    "v1": {
      "letters": [
        "",
        "H",
        "O",
        "5",
        " ",
        "!",
        "\"",
        "'",
        "(",
        ")",
        ",",
        "-",
        ".",
        ":",
        ";",
        "?",
        "א",
        "ב",
        "ג",
        "ד",
        "ה",
        "ו",
        "ז",
        "ח",
        "ט",
        "י",
        "ך",
        "כ",
        "ל",
        "ם",
        "מ",
        "ן",
        "נ",
        "ס",
        "ע",
        "ף",
        "פ",
        "ץ",
        "צ",
        "ק",
        "ר",
        "ש",
        "ת"
      ],
      "niqqud": {
        "decode": [
          "",
          "ֿ",
          "ְ",
          "ֱ",
          "ֲ",
          "ֳ",
          "ִ",
          "ֵ",
          "ֶ",
          "ַ",
          "ָ",
          "ֹ",
          "ֺ",
          "ֻ",
          "ּ",
          "ַ"
        ],
        "encode_overrides": {
          "ַ": 15
        }
      },
      "dagesh": {
        "decode": [
          "",
          "ֿ",
          "ּ"
        ]
      },
      "sin": {
        "decode": [
          "",
          "ֿ",
          "ׁ",
          "ׂ"
        ]
      },
      "notes": [
        "index 0 is the mask/padding token (empty string)",
        "RAFE (U+05BF) = 'category applies but deliberately absent'",
        "niqqud contains PATAH twice (historic bug): decode index 9 and 15; encoding always produced 15, so the model predicts 15 for patah and index 9 is dead. Preserved for model compatibility.",
        "niqqud contains HOLAM HASER FOR VAV (U+05BA) although the corpus essentially never produces it."
      ]
    },
    "v2": {
      "letters": [
        "",
        "H",
        "O",
        "5",
        " ",
        "!",
        "\"",
        "'",
        "(",
        ")",
        ",",
        "-",
        ".",
        ":",
        ";",
        "?",
        "א",
        "ב",
        "ג",
        "ד",
        "ה",
        "ו",
        "ז",
        "ח",
        "ט",
        "י",
        "ך",
        "כ",
        "ל",
        "ם",
        "מ",
        "ן",
        "נ",
        "ס",
        "ע",
        "ף",
        "פ",
        "ץ",
        "צ",
        "ק",
        "ר",
        "ש",
        "ת"
      ],
      "niqqud": {
        "decode": [
          "",
          "ֿ",
          "ְ",
          "ֱ",
          "ֲ",
          "ֳ",
          "ִ",
          "ֵ",
          "ֶ",
          "ַ",
          "ָ",
          "ׇ",
          "ֹ",
          "ֻ",
          "ּ"
        ],
        "encode_overrides": {}
      },
      "dagesh": {
        "decode": [
          "",
          "ֿ",
          "ּ"
        ]
      },
      "sin": {
        "decode": [
          "",
          "ֿ",
          "ׁ",
          "ׂ"
        ]
      },
      "data_folds": {
        "ֺ": "ֹ"
      },
      "notes": [
        "qamats qatan (U+05C7) is a first-class niqqud label; corpora must be relabeled (e.g. Dicta keepqq=true) before it carries signal. Until then training may fold it into qamats.",
        "models declare their inventory in the sidecar model card; the runtime must not assume v2."
      ]
    }
  },
  "decomposition": {
    "rafe": "ֿ",
    "can_dagesh": "בגדהוזטיכלמנספצקשתךף",
    "can_sin": "ש",
    "can_niqqud": "אבגדהוזחטיכלמנסעפצקרשתךן",
    "channel_order_in_text": [
      "dagesh",
      "sin",
      "niqqud"
    ],
    "vav_shuruk_rule": "if letter is vav with dagesh and no niqqud, reinterpret: dagesh := RAFE, niqqud := U+05BC (shuruk)"
  },
  "text_cleanup": {
    "comment": "chars deleted from input before decomposition/encoding",
    "strip": [
      "ֽ",
      "͏",
      "‍",
      "‎",
      "‏",
      "﻿"
    ]
  },
  "remove_niqqud": {
    "comment": "regex character class of chars removed by remove_niqqud",
    "chars": "ְ-ׇּֿׁׂ"
  },
  "output_convention": "full-script (ktiv male) Hebrew with diacritics layered onto the unchanged input letter sequence, emitted in NFC canonical mark order; remove_niqqud(output) round-trips to the input (modulo whitespace collapsing). This is NOT normative ktiv-haser nikud; teachers/corpora must be projected into this convention."
} as const;
