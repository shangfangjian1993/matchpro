"""canonical_en 队名归一化(审查 A70A601 §十五 + A 专项)。"""

from __future__ import annotations

from app.data.canonical.team_names import canonical_en


def test_canonical_en_unifies_fc_afc_spellings():
    pairs = [
        ("AFC Bournemouth", "Bournemouth"),
        ("Manchester City FC", "Manchester City"),
        ("Liverpool FC", "Liverpool"),
        ("Brighton & Hove Albion FC", "Brighton & Hove Albion"),
        ("Tottenham Hotspur FC", "Tottenham Hotspur"),
        ("Sunderland AFC", "Sunderland"),
    ]
    for a, b in pairs:
        ca, cb = canonical_en(a), canonical_en(b)
        assert ca == cb, f"{a!r}->{ca!r} != {b!r}->{cb!r}"
        assert ca == ca.strip().lower()  # 小写压缩、无多余空格
