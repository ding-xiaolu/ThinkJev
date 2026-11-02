"""Owned C-block generator: reproducibility, schema, balance, and labels recomputed independently.

The label checks below do not call the generator's label code. They re-derive every answer
from the facts stored in `provenance.derivation` (dates, tables, rule facts, statements,
edges, intervals) or by brute force over all assignments, and check that those facts are
the ones shown in the state.
"""

import collections
import datetime as dt
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

import pytest

from thinkjev import _semif

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import generate_owned  # noqa: E402

COMMITTED = ROOT / "benchmarks/data/owned-v1.jsonl"
CJK = re.compile(r"[一-鿿]")


@pytest.fixture(scope="module")
def rows():
    return generate_owned.owned_rows()


def _answer(row):
    return row["options"][row["label"]]["id"]


def _tokens(text):
    """Rough token estimate: one per CJK character, one per four other characters."""
    cjk = len(CJK.findall(text))
    return cjk + (len(text) - cjk) / 4


def test_committed_fixture_is_reproduced_byte_for_byte(rows):
    assert COMMITTED.read_bytes() == generate_owned.dump(rows).encode()


def test_generation_is_deterministic_and_subsets_match():
    first = generate_owned.owned_rows(per_cell=3)
    assert generate_owned.dump(first) == generate_owned.dump(generate_owned.owned_rows(per_cell=3))
    subset = generate_owned.owned_rows(["graph"], per_cell=3)
    assert subset == [row for row in first if row["family"] == "graph"]


def test_sizes_schema_and_prompt(rows):
    counts = collections.Counter((row["family"], row["lang"], row["difficulty"]) for row in rows)
    assert len(counts) == 8 * 2 * 4
    assert all(counts[key] == generate_owned.PER_CELL[key[1]] for key in counts)
    assert sum(row["lang"] == "en" for row in rows) == 2400 and sum(row["lang"] == "zh" for row in rows) == 1600
    assert len({row["id"] for row in rows}) == len(rows)
    for row in rows:
        _semif.validate_row(row)
        assert row["id"] == row["group_id"] == (f"owned/{row['family']}/{row['lang']}/d{row['difficulty']}/"
                                                f"{int(row['id'].rsplit('/', 1)[1]):03d}")
        assert row["difficulty"] in (1, 2, 3, 4) and row["lang"] in ("en", "zh")
        assert 0 <= row["label"] < len(row["options"]) and row["target_distribution"] is None
        assert row["split"] == "test"
        provenance = row["provenance"]
        assert provenance["source"] == "project-authored-programmatic" and provenance["generator"] == "owned-v1"
        assert provenance["derivation"]["answer"] == _answer(row)
        assert len({option["description"] for option in row["options"]}) == len(row["options"])
        assert _tokens(json.dumps(_semif.direct_messages(row), ensure_ascii=False)) < 1500
        if row["lang"] == "zh":
            assert CJK.search(row["state"]) and CJK.search(row["question"])
        else:
            assert not CJK.search(row["state"] + row["question"])


class CharTokenizer:
    """One token per character: enough to run SemIf's encoder and its answer-slot checks."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


def test_semif_prompt_encodes(rows):
    firsts = {}
    for row in rows:
        firsts.setdefault((row["family"], row["lang"], row["difficulty"]), row)
    for row in firsts.values():
        ids, slots, _ = _semif.encode_prompt(CharTokenizer(), row, max_tokens=100_000)
        assert ids and len(slots) == len(row["options"])


def test_answer_positions_and_classes_are_balanced(rows):
    cells = collections.defaultdict(list)
    for row in rows:
        cells[row["family"], row["lang"], row["difficulty"]].append(row)
    for key, cell in cells.items():
        widths = {len(row["options"]) for row in cell}
        assert len(widths) == 1, key
        positions = collections.Counter(row["label"] for row in cell)
        assert len(positions) == widths.pop() and max(positions.values()) - min(positions.values()) <= 1, key
        if key[0] in ("rules", "negation") or (key[0] in ("graph", "scheduling") and key[2] <= (2 if key[0] == "graph" else 1)):
            classes = collections.Counter(_answer(row) for row in cell)
            assert max(classes.values()) - min(classes.values()) <= 1, (key, classes)
        if key[0] in ("dates", "counting", "arithmetic") or (key[0] == "graph" and key[2] >= 3):
            # numeric options: the answer's rank among the values must not give it away
            def rank(row):
                values = [int(re.sub(r"\D", "", option["id"])) for option in row["options"]]
                return sorted(values).index(values[row["label"]])

            ranks = collections.Counter(rank(row) for row in cell)
            assert len(ranks) == 4 and max(ranks.values()) - min(ranks.values()) <= 1, (key, ranks)


# Independent recomputation, one checker per family.

def _next_business(day, holidays):
    while True:
        day += dt.timedelta(days=1)
        if day.isoweekday() <= 5 and day.isoformat() not in holidays:
            return day


def _shown(row, day):
    text = f"{day.year}年{day.month}月{day.day}日" if row["lang"] == "zh" else f"{day:%B} {day.day}, {day.year}"
    return text in row["state"]


def check_dates(row, facts):
    start, holidays = dt.date.fromisoformat(facts["start"]), set(facts["holidays"])
    if facts["kind"] == "calendar_days":
        return dt.date.fromordinal(start.toordinal() + facts["days"]).isoformat(), _shown(row, start)
    if facts["kind"] == "month_end_business_days":
        issued = dt.date.fromisoformat(facts["issued"])
        following = dt.date(issued.year + issued.month // 12, issued.month % 12 + 1, 1)
        assert start == following - dt.timedelta(days=1)
        start_shown = _shown(row, issued)
    else:
        start_shown = _shown(row, start)
    day = start
    for _ in range(facts["days"]):
        day = _next_business(day, holidays)
    assert all(_shown(row, dt.date.fromisoformat(h)) for h in holidays)
    if facts["kind"] != "timezone_business_days":
        return day.isoformat(), start_shown
    moment = dt.datetime(day.year, day.month, day.day) + dt.timedelta(minutes=facts["deadline_minutes"])
    utc = moment - dt.timedelta(minutes=facts["origin_offset_minutes"])
    return (utc + dt.timedelta(minutes=facts["target_offset_minutes"])).strftime("%Y-%m-%dT%H:%M"), start_shown


def check_counting(row, facts):
    def ok(condition, record):
        op, field, *args = condition
        value = record[field]
        if op == "eq":
            return value == args[0]
        if op == "ne":
            return value != args[0]
        if op == "in":
            return value in args[0]
        if op == "gt":
            return value > args[0]
        return args[0] <= value <= args[1]

    count = len([r for r in facts["table"] if all(ok(c, r) for c in facts["conditions"])])
    shown = all(str(r["order"]) in row["state"] and str(r["amount"]) in row["state"] for r in facts["table"])
    return f"n{count}", shown and len(row["state"].splitlines()) == len(facts["table"]) + 2


def check_rules(row, facts):
    level, f = row["difficulty"], facts["facts"]
    tier = 0
    for limit in f["limits"]:
        if f["amount"] > limit:
            tier += 1
    if f["preapproved"] and f["category"] != f["excluded_category"] and not (level == 4 and f["emergency"]):
        tier = max(0, tier - 1)
    if level >= 2 and f["restricted"]:
        tier = max(tier, 2)
    if level >= 3 and f["category"] in ("software", "软件") and f.get("term_months", 0) > 12:
        tier = min(3, tier + 1)
    if level == 4 and f["framework"] and not f["restricted"]:
        tier = min(tier, 1)
    listed = re.search(r"(?:Pre-approved vendors|预批准供应商)[:：]\s*(.*)", row["state"]).group(1)
    shown = (f["vendor"] in listed) == f["preapproved"] and f"{f['amount']:,}" in row["state"]
    shown &= row["state"].count("Rule ") + row["state"].count("规则") >= len(facts["levels_after_each_rule"])
    return ["none", "manager", "director", "vp"][tier], shown


def check_negation(row, facts):
    def truth(statement, world):
        items = [world[i] for i in statement["items"]]
        kind = statement["kind"]
        if kind == "lit":
            value = items[0] if statement["positive"] else not items[0]
        elif kind == "all":
            value = all(items)
        elif kind == "none":
            value = not any(items)
        elif kind == "some":
            value = any(items)
        else:
            value = items.count(True) == statement["k"]
        return value if statement["outer"] % 2 == 0 else not value

    worlds = [w for w in itertools.product((True, False), repeat=facts["items"])
              if all(truth(s, w) for s in facts["statements"])]
    seen = {w[facts["query"]] for w in worlds}
    answer = "undetermined" if len(seen) == 2 else ("yes" if seen == {True} else "no")
    lines = [line for line in row["state"].splitlines() if line.startswith(("Statement", "说法"))]
    wraps = sum(line.count("it is not the case") + line.count("It is not the case") + line.count("it is false")
                + line.count("It is false") + line.count("说法是假的") + line.count("说法不成立") for line in lines)
    return answer, bool(worlds) and len(lines) == len(facts["statements"]) and wraps == sum(
        s["outer"] for s in facts["statements"])


def check_arithmetic(row, facts):
    x = facts["inputs"]
    level = row["difficulty"]
    if level == 1:
        value = x["per_tray"] * x["trays"] + x["extra"] - x["sold"]
    elif level == 2:
        value = x["price"] * x["quantity"] - x["price"] * x["quantity"] * x["discount_percent"] / 100 + x["delivery"]
    elif level == 3:
        value = (x["capacity"] - x["initial"] - x["rate_a"] * x["alone_minutes"]) / (x["rate_a"] + x["rate_b"])
    else:
        stock = x["start"] - x["orders"] * x["units_per_order"] + x["returned"] - x["damaged"]
        stock -= stock * x["sold_percent"] / 100
        cases = 0
        while stock + cases * x["case_size"] < x["target"]:
            cases += 1
        value = cases
    assert value == int(value) and value > 0
    shown = all(re.search(rf"(?<![\d,])({v}|{v:,})(?![\d])", row["state"]) for v in x.values())
    return f"v{int(value)}", shown


def check_logic_grid(row, facts):
    people, attrs = facts["people"], facts["attributes"]
    n = len(people)
    seat = attrs.index("seat") if "seat" in attrs else None

    def clue_attrs(clue):
        kind = clue[0]
        if kind in ("is", "not", "either"):
            return {clue[2]}
        if kind in ("link", "unlink"):
            return {clue[1], clue[3]}
        return {seat} | {ref[1] for ref in clue[1:] if ref[0] == "holder"}

    def holds(clue, sol):  # sol[a][p] = value index of person p
        def who(a, v):
            return sol[a].index(v)

        kind = clue[0]
        if kind == "is":
            return sol[clue[2]][clue[1]] == clue[3]
        if kind == "not":
            return sol[clue[2]][clue[1]] != clue[3]
        if kind == "either":
            return sol[clue[2]][clue[1]] in (clue[3], clue[4])
        if kind in ("link", "unlink"):
            return (sol[clue[3]][who(clue[1], clue[2])] == clue[4]) == (kind == "link")
        spots = [sol[seat][ref[1] if ref[0] == "person" else who(ref[1], ref[2])] for ref in clue[1:]]
        if kind == "end":
            return spots[0] in (0, n - 1)
        a, b = spots
        return {"left": b - a == 1, "before": a < b, "adjacent": abs(a - b) == 1, "apart": abs(a - b) >= 2}[kind]

    solutions, order = [], sorted(range(len(attrs)), key=lambda a: a != seat)

    def search(depth, sol):
        if depth == len(order):
            solutions.append([list(v) for v in sol])
            return
        for perm in itertools.permutations(range(n)):
            sol[order[depth]] = perm
            done = set(order[:depth + 1])
            if all(holds(c, sol) for c in facts["clues"] if clue_attrs(c) <= done):
                search(depth + 1, sol)
        sol[order[depth]] = None

    search(0, [None] * len(attrs))
    assert len(solutions) == 1 and solutions[0] == facts["solution"]
    q = facts["question"]
    value = solutions[0][q["attribute"]][q["person"]]
    lines = [line for line in row["state"].splitlines() if re.match(r"\d+\. ", line)]
    return (f"p{q['person']}" if q["ask"] == "who" else f"v{value}"), (
        len(lines) == len(facts["clues"]) and all(p in row["state"] for p in people))


def check_graph(row, facts):
    edges, n = facts["edges"], len(facts["nodes"])
    lines = [line for line in row["state"].splitlines() if "→" in line and not line.startswith(("Today", "今天"))]
    shown = len(lines) == len(edges) and all(
        line.startswith(f"{facts['nodes'][e[0]]} → {facts['nodes'][e[1]]}") for line, e in zip(lines, edges))
    if row["difficulty"] <= 2:
        reach = {facts["source"]}
        for _ in range(n):
            reach |= {b for a, b in edges if a in reach}
        return ("yes" if facts["target"] in reach else "no"), shown
    inf = float("inf")
    dist = [[0 if i == j else inf for j in range(n)] for i in range(n)]
    for a, b, w in edges:
        if [a, b] != facts["closed"]:
            dist[a][b] = min(dist[a][b], w)
    for k, i, j in itertools.product(range(n), repeat=3):  # Floyd-Warshall, k outermost
        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    s, t, v = facts["source"], facts["target"], facts["via"]
    total = dist[s][t] if v is None else dist[s][v] + dist[v][t]
    return f"m{total}", shown


def check_scheduling(row, facts):
    def clash(a, b):
        return a[0] < b[1] and b[0] < a[1]

    level = row["difficulty"]
    if level == 1:
        overlap = any(clash(facts["proposal"], b) for b in facts["bookings"])
        return ("overlap" if overlap else "free"), all(
            generate_owned._clock(b[0]) in row["state"] for b in facts["bookings"])
    feasible = []
    for option in row["options"]:
        start = int(option["id"][1:])
        if level == 2:
            slot = [start, start + facts["length"]]
            ok = not any(clash(slot, b) for person in facts["busy"] for b in person)
        elif level == 3:
            ok = all(sum(k for a, b, k in facts["reservations"] if a <= minute < b) + facts["need"] <= facts["capacity"]
                     for minute in range(start, start + 60))
        else:
            ok = all(540 <= start + off and start + off + 60 <= 1020
                     and not any(clash([start + off, start + off + 60], b) for b in person)
                     for off, person in zip(facts["offsets"], facts["busy"]))
        if ok:
            feasible.append(option["id"])
    assert len(feasible) == 1  # every distractor fails, exactly one slot works
    return feasible[0], True


CHECKERS = {"dates": check_dates, "counting": check_counting, "rules": check_rules, "negation": check_negation,
            "arithmetic": check_arithmetic, "logic_grid": check_logic_grid, "graph": check_graph,
            "scheduling": check_scheduling}


@pytest.mark.parametrize("family", sorted(CHECKERS))
def test_labels_recomputed_independently(rows, family):
    for row in rows:
        if row["family"] != family:
            continue
        expected, shown = CHECKERS[family](row, row["provenance"]["derivation"])
        assert _answer(row) == expected, row["id"]
        assert shown, row["id"]


def test_cli_is_create_only(tmp_path, monkeypatch):
    output = tmp_path / "owned.jsonl"
    argv = ["generate_owned.py", "--output", str(output), "--per-cell", "2", "--families", "dates", "graph"]
    monkeypatch.setattr(sys, "argv", argv)
    generate_owned.main()
    lines = output.read_text().splitlines()
    assert len(lines) == 2 * 2 * 4 * 2
    with pytest.raises(FileExistsError):
        generate_owned.main()


# Chess: four hand-made puzzles in Lichess CSV layout (checked with python-chess below). Each
# FEN is before the opponent's move; the second move is the solution, here always a mate.

CHESS_CSV = """PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags
t0001,r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3,g8f6 h5f7,900,80,95,5000,mate mateIn1 opening,https://example.org/t0001,
t0002,6k1/5ppp/8/8/8/8/r4PPP/3R2K1 b - - 0 1,a2b2 d1d8,1400,80,90,3000,backRankMate mate mateIn1,https://example.org/t0002,
t0003,6rk/p5pp/8/6N1/8/8/8/6K1 b - - 0 1,a7a6 g5f7,1800,80,88,2500,smotheredMate mate mateIn1,https://example.org/t0003,
t0004,3r2k1/5ppp/8/8/8/8/R4PPP/6K1 w - - 0 1,a2b2 d8d1,2200,80,85,1500,backRankMate mate mateIn1,https://example.org/t0004,
t0005,3r2k1/5ppp/8/8/8/8/R4PPP/6K1 w - - 0 1,a2b2 d8d1,1300,80,40,1500,backRankMate mate mateIn1,https://example.org/t0005,
"""


def test_chess_rows_from_fixture(tmp_path):
    chess = pytest.importorskip("chess")
    path = tmp_path / "puzzles.csv"
    path.write_text(CHESS_CSV)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    english = generate_owned.chess_rows(path, {"en": 1, "zh": 0})
    chinese = generate_owned.chess_rows(path, {"en": 0, "zh": 1})
    assert [row["difficulty"] for row in english] == [1, 2, 3, 4]  # t0005 fails the popularity filter
    expected = {"t0001": ("h5f7", "White"), "t0002": ("d1d8", "White"), "t0003": ("g5f7", "White"),
                "t0004": ("d8d1", "Black")}
    for row in english + chinese:
        _semif.validate_row(row)
        facts = row["provenance"]["derivation"]
        assert row["provenance"]["upstream_csv_sha256"] == digest and row["family"] == "chess"
        assert row["id"] == f"owned/chess/{row['lang']}/d{row['difficulty']}/000"
        solution, side = expected[facts["puzzle_id"]]
        assert _answer(row) == solution and len(row["options"]) == 4
        board = chess.Board(facts["fen_before"])
        board.push_uci(facts["opponent_move"])
        assert board.fen() in row["state"]
        for option in row["options"]:
            move = chess.Move.from_uci(option["id"])
            assert board.is_legal(move) and option["description"].startswith(board.san(move))
            board.push(move)
            assert board.is_checkmate() == (option["id"] == solution)
            board.pop()
        if row["lang"] == "en":
            assert row["question"] == f"Which move is best for {side}?"
        else:
            assert CJK.search(row["state"]) and ("黑方" if side == "Black" else "白方") in row["question"]
