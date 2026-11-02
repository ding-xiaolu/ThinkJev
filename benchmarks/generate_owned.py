"""Generate the owned C block: programmatic S2-type decisions in English and Chinese.

Eight families (date arithmetic, conditional counting, chained rules, stacked negation,
arithmetic word problems, logic grids, graph paths, scheduling) each have four difficulty
levels that add reasoning steps, not just length. Every label is computed by code and the
facts that justify it are stored in `provenance.derivation`. Answer positions and answer
classes are balanced within every (family, lang, difficulty) cell, and each row draws from
its own seeded generator, so any family subset reproduces the same rows.

The optional `chess` family reads the CC0 Lichess puzzle database (decompressed CSV) and
needs the `data` extra (python-chess). Its rows are derived from third-party data: rebuild
them into `build/`, do not commit them. Rows follow the SemIf schema; `split` is a
placeholder, the real split is assigned by `build_cascade.py assemble`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import heapq
import itertools
import json
from pathlib import Path
import random

import numpy as np

GENERATOR = "owned-v1"
SEED = 261001
LEVELS = (1, 2, 3, 4)
LANGS = ("en", "zh")
PER_CELL = {"en": 75, "zh": 50}
CHESS_PER_CELL = {"en": 200, "zh": 100}
RIGHTS = "Project authored; no copied external text"
CHESS_RIGHTS = "Puzzle positions and moves from the Lichess puzzle database (CC0 1.0); question text project authored"
LETTERS = "ABCDE"
ZH_NUMBERS = "零一两三四五六七八九十"
EN_NUMBERS = ["zero", "one", "two", "three", "four", "five"]


def _t(lang: str, en: str, zh: str) -> str:
    return en if lang == "en" else zh


def _opt(option_id, description: str) -> dict:
    return {"id": str(option_id), "description": description}


def _ranked(answer, mistakes, rank: int, step=1, low=None) -> list:
    """Three distractors with exactly `rank` of them below the answer, so the answer's rank among
    the option values is balanced and cannot be guessed from the values alone. Plausible
    mistakes come first on each side; the rest are filled at answer -/+ k * step."""
    below, above = [], []
    for value in mistakes:
        if value != answer and value not in below + above and (low is None or value >= low):
            (below if value < answer else above).append(value)
    below, above = below[:rank], above[:3 - rank]
    for side, need, sign in ((below, rank, -1), (above, 3 - rank, 1)):
        k = 1
        while len(side) < need:
            value = answer + sign * k * step
            k += 1
            if low is not None and value < low:
                raise ValueError("answer too small for the requested rank")
            if value not in side:
                side.append(value)
    return below + above


def _answer_class(index: int, count: int) -> int:
    """Answer class (or, for numeric options, the answer's rank among the option values). The
    position is `index % count`, so this Latin-square pairing balances both over every run of
    `count` rows and pairs every class with every position."""
    return (index + index // count) % count


def _unit(template: str, n: int, lang: str) -> str:
    """Format a count with a "singular|plural" English template."""
    if "|" in template:
        head, plural = template.split("|")
        template = head if n == 1 else head.rsplit(" ", 1)[0] + " " + plural
    return template.format(f"{n:,}" if lang == "en" else n)


def _join(lang: str, items: list[str]) -> str:
    if lang == "zh":
        return "、".join(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# Dates. Weekdays are given for the start date only; the rest is counting.

WEEKDAYS = {
    "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "zh": ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"],
}
ZONES = [  # fixed offsets in minutes; rows state that daylight saving is ignored
    ("Tokyo", "东京", 540), ("Singapore", "新加坡", 480), ("Mumbai", "孟买", 330), ("Dubai", "迪拜", 240),
    ("Berlin", "柏林", 60), ("London", "伦敦", 0), ("São Paulo", "圣保罗", -180),
    ("New York", "纽约", -300), ("Chicago", "芝加哥", -360), ("Los Angeles", "洛杉矶", -480),
]


def _day(lang: str, value: dt.date, weekday: bool = False) -> str:
    if lang == "en":
        text = f"{value:%B} {value.day}, {value.year}"
        return f"{WEEKDAYS['en'][value.weekday()]}, {text}" if weekday else text
    text = f"{value.year}年{value.month}月{value.day}日"
    return f"{text}（{WEEKDAYS['zh'][value.weekday()]}）" if weekday else text


def _clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _moment(lang: str, value: dt.datetime) -> str:
    return _t(lang, f"{_day(lang, value.date())}, {_clock(value.hour * 60 + value.minute)}",
              f"{_day(lang, value.date())} {_clock(value.hour * 60 + value.minute)}")


def _utc(minutes: int) -> str:
    sign, minutes = ("+" if minutes >= 0 else "-"), abs(minutes)
    return f"UTC{sign}{minutes // 60}" + (f":{minutes % 60:02d}" if minutes % 60 else "")


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _business(start: dt.date, days: int, holidays=()) -> dt.date:
    """The `days`-th business day strictly after `start` (Mon-Fri, not a holiday)."""
    current = start
    while days:
        current += dt.timedelta(days=1)
        if current.weekday() < 5 and current not in holidays:
            days -= 1
    return current


def _random_day(rng: random.Random, weekday_only: bool = False) -> dt.date:
    while True:
        value = dt.date(2026, 1, 1) + dt.timedelta(days=rng.randrange(700))
        if not weekday_only or value.weekday() < 5:
            return value


def _holidays(rng: random.Random, after: dt.date, until: dt.date, count: int) -> list[dt.date]:
    span = [after + dt.timedelta(days=d) for d in range(1, (until - after).days + 1)]
    return sorted(rng.sample([d for d in span if d.weekday() < 5], count))


def dates(rng: random.Random, lang: str, level: int, index: int):
    day = dt.timedelta(days=1)
    if level == 1:  # calendar days across a month end
        while True:
            start, days = _random_day(rng), rng.randint(10, 45)
            answer = start + days * day
            if answer.month != start.month:
                break
        wrong = [answer - day, answer + day, rng.choice((answer - 2 * day, answer + 2 * day, answer - 7 * day))]
        state = _t(lang,
                   f"A tenant gave written notice on {_day(lang, start, True)}. The lease requires {days} calendar days "
                   "of notice, counted starting the day after notice is given; the tenancy ends on the last of those days.",
                   f"租客于{_day(lang, start, True)}递交了书面退租通知。租约规定须提前{days}个自然日通知，"
                   f"从通知次日起算，第{days}天为租期的最后一天。")
        question = _t(lang, "On what date does the tenancy end?", "租期在哪一天结束？")
        derivation = {"kind": "calendar_days", "start": start.isoformat(), "days": days, "holidays": []}
    elif level == 2:  # business days
        start, days = _random_day(rng, True), rng.randint(3, 12)
        answer = _business(start, days)
        wrong = [start + days * day, _business(start, days - 1), _business(start, days + 1)]
        state = _t(lang,
                   f"An order was placed on {_day(lang, start, True)}. Orders ship on the {_ordinal(days)} business day "
                   "after the order date. Business days are Monday to Friday; the order date itself does not count.",
                   f"一笔订单于{_day(lang, start, True)}下单。订单在下单日之后的第{days}个工作日发货。"
                   "工作日为周一至周五，下单当天不计入。")
        question = _t(lang, "On what date does the order ship?", "这笔订单在哪一天发货？")
        derivation = {"kind": "business_days", "start": start.isoformat(), "days": days, "holidays": []}
    elif level == 3:  # month end, then business days around listed holidays
        issued, days = _random_day(rng), rng.randint(3, 10)
        month_end = (issued.replace(day=28) + 4 * day).replace(day=1) - day
        plain = _business(month_end, days)
        holidays = _holidays(rng, month_end, plain, rng.randint(1, 2))
        answer = _business(month_end, days, holidays)
        wrong = [plain, _business(issued, days, holidays), _business(month_end, days + 1, holidays),
                 _business(month_end, days - 1, holidays)]
        listed = _join(lang, [_day(lang, h, True) for h in holidays])
        state = _t(lang,
                   f"An invoice was issued on {_day(lang, issued, True)}. Invoices are due on the {_ordinal(days)} "
                   "business day after the last day of the month in which they were issued. Business days are Monday "
                   f"to Friday, excluding company holidays. Company holidays: {listed}.",
                   f"一张发票于{_day(lang, issued, True)}开具。发票的付款期限为开具当月最后一天之后的第{days}个工作日。"
                   f"工作日为周一至周五，公司假日除外。公司假日：{listed}。")
        question = _t(lang, "On what date is the invoice due?", "这张发票的付款期限是哪一天？")
        derivation = {"kind": "month_end_business_days", "start": month_end.isoformat(), "issued": issued.isoformat(),
                      "days": days, "holidays": [h.isoformat() for h in holidays]}
    else:  # business days in one office's calendar, then a fixed-offset time-zone conversion
        origin, target = rng.sample(ZONES, 2)
        filed, days = _random_day(rng, True), rng.randint(2, 6)
        clock = rng.choice((8 * 60, 9 * 60, 10 * 60, 12 * 60, 17 * 60))
        holidays = _holidays(rng, filed, _business(filed, days), 1)
        shift = dt.timedelta(minutes=target[2] - origin[2])

        def local(n: int, hol) -> dt.datetime:
            return dt.datetime.combine(_business(filed, n, hol), dt.time()) + dt.timedelta(minutes=clock)

        answer = local(days, holidays) + shift
        wrong = [local(days, holidays), local(days, holidays) - shift, local(days, ()) + shift,
                 local(days + 1, holidays) + shift]
        o, g = (origin[0], target[0]) if lang == "en" else (origin[1], target[1])
        state = _t(lang,
                   f"A client filed a request with the {o} office ({_utc(origin[2])}) on {_day(lang, filed, True)}. "
                   f"The response deadline is {_clock(clock)} {o} time on the {_ordinal(days)} business day after the "
                   f"filing date, counted on the {o} office calendar: Monday to Friday, excluding its holiday on "
                   f"{_day(lang, holidays[0], True)}. The engineer handling the request works in {g} "
                   f"({_utc(target[2])}). Offsets are fixed; ignore daylight saving time.",
                   f"客户于{_day(lang, filed, True)}向{o}办事处（{_utc(origin[2])}）提交了一项请求。答复期限为提交日之后"
                   f"第{days}个工作日的{o}时间{_clock(clock)}，工作日按{o}办事处的日历计算：周一至周五，"
                   f"不含该办事处在{_day(lang, holidays[0], True)}的假日。负责处理的工程师在{g}（{_utc(target[2])}）。"
                   "时差固定，不考虑夏令时。")
        question = _t(lang, f"What is the deadline in {g} local time?", f"换算成{g}当地时间，答复期限是什么时候？")
        derivation = {"kind": "timezone_business_days", "start": filed.isoformat(), "days": days,
                      "holidays": [h.isoformat() for h in holidays], "deadline_minutes": clock,
                      "origin_offset_minutes": origin[2], "target_offset_minutes": target[2]}
        wrong = _ranked(answer, wrong, _answer_class(index, 4), day)
        options = [_opt(v.isoformat(timespec="minutes"), _moment(lang, v)) for v in [answer] + wrong]
        return state, question, options[0], options[1:], derivation
    wrong = _ranked(answer, wrong, _answer_class(index, 4), day)
    options = [_opt(v.isoformat(), _day(lang, v)) for v in [answer] + wrong]
    return state, question, options[0], options[1:], derivation


# Counting with conjunctive conditions over a listed table.

COUNT_VALUES = {
    "en": {"city": ["Lyon", "Porto", "Graz", "Turin", "Ghent"], "status": ["paid", "pending", "refunded", "shipped"],
           "channel": ["web", "store", "phone"]},
    "zh": {"city": ["杭州", "成都", "西安", "苏州", "青岛"], "status": ["已付款", "待付款", "已退款", "已发货"],
           "channel": ["网店", "门店", "电话"]},
}
COUNT_ROWS = {1: 8, 2: 12, 3: 16, 4: 20}


def _holds(condition: list, row: dict) -> bool:
    op, field, *args = condition
    value = row[field]
    return {"eq": lambda: value == args[0], "ne": lambda: value != args[0], "in": lambda: value in args[0],
            "gt": lambda: value > args[0], "between": lambda: args[0] <= value <= args[1]}[op]()


def _condition_text(lang: str, condition: list) -> str:
    op, field, *args = condition
    names = {"en": {"city": "city", "status": "status", "channel": "channel"},
             "zh": {"city": "城市", "status": "状态", "channel": "渠道"}}[lang]
    if lang == "en":
        return {"eq": lambda: f"{names.get(field)} is '{args[0]}'", "ne": lambda: f"{names.get(field)} is not '{args[0]}'",
                "in": lambda: f"{field} is '{args[0][0]}' or '{args[0][1]}'",
                "gt": lambda: f"amount is greater than ${args[0]}",
                "between": lambda: f"amount is between ${args[0]} and ${args[1]} inclusive"}[op]()
    return {"eq": lambda: f"{names.get(field)}为“{args[0]}”", "ne": lambda: f"{names.get(field)}不是“{args[0]}”",
            "in": lambda: f"状态为“{args[0][0]}”或“{args[0][1]}”", "gt": lambda: f"金额大于{args[0]}元",
            "between": lambda: f"金额在{args[0]}元到{args[1]}元之间（含两端）"}[op]()


def counting(rng: random.Random, lang: str, level: int, index: int):
    values = COUNT_VALUES[lang]
    size, rank = COUNT_ROWS[level], _answer_class(index, 4)
    while True:
        table = [{"order": number, "city": rng.choice(values["city"]),
                  "status": rng.choice(values["status"][:2] * 2 + values["status"]),
                  "channel": rng.choice(values["channel"]), "amount": rng.randrange(20, 1000)}
                 for number in rng.sample(range(1000, 10000), size)]
        amounts = [row["amount"] for row in table]
        status, channel, city = (rng.choice(values[key]) for key in ("status", "channel", "city"))
        pick = lambda: rng.choice(amounts) if rng.random() < 0.5 else rng.randrange(100, 900)  # noqa: E731
        if level == 1:
            conditions = [["eq", "status", status]]
        elif level == 2:
            conditions = [["eq", "status", status], ["eq", "channel", channel]]
        elif level == 3:
            conditions = [["eq", "status", status], ["ne", "city", city], ["gt", "amount", pick()]]
        else:
            low, high = sorted((pick(), pick()))
            conditions = [["in", "status", sorted(rng.sample(values["status"], 2), key=values["status"].index)],
                          ["ne", "channel", channel], ["between", "amount", low, high], ["ne", "city", city]]
        count = sum(all(_holds(c, row) for c in conditions) for row in table)
        loose = sum(all(_holds(c, row) for c in conditions[:-1]) for row in table)
        if max(1, rank) <= count <= size - 2 and (level == 1 or loose != count):
            break
    first = sum(_holds(conditions[0], row) for row in table)
    wrong = _ranked(count, [loose, first], rank, 1, 0)
    if lang == "en":
        lines = ["Order | City | Status | Channel | Amount"]
        lines += [f"#{r['order']} | {r['city']} | {r['status']} | {r['channel']} | ${r['amount']}" for r in table]
        state = "Order log:\n" + "\n".join(lines)
        criteria = "; ".join(_condition_text(lang, c) for c in conditions)
        question = (f"How many orders meet this condition: {criteria}?" if level == 1 else
                    f"How many orders meet all of these conditions: {criteria}?")
        unit = lambda n: f"{n} order" + ("" if n == 1 else "s")  # noqa: E731
    else:
        lines = ["订单号｜城市｜状态｜渠道｜金额"]
        lines += [f"{r['order']}｜{r['city']}｜{r['status']}｜{r['channel']}｜{r['amount']}元" for r in table]
        state = "订单记录：\n" + "\n".join(lines)
        criteria = "；".join(_condition_text(lang, c) for c in conditions)
        question = f"满足以下条件的订单有几笔：{criteria}？" if level == 1 else f"同时满足以下全部条件的订单有几笔：{criteria}？"
        unit = lambda n: f"{n}笔"  # noqa: E731
    options = [_opt(f"n{n}", unit(n)) for n in [count] + wrong]
    derivation = {"table": table, "conditions": conditions, "count": count}
    return state, question, options[0], options[1:], derivation


# Chained purchasing rules with exceptions and overrides, applied in order.

APPROVAL = {"en": ["No approval needed", "Manager approval", "Director approval", "VP approval"],
            "zh": ["无需审批", "部门经理审批", "总监审批", "副总裁审批"]}
VENDORS = {"en": ["Acme Supply", "Borealis Parts", "Cobalt Office", "Dunmore Labs", "Everline Tools", "Fairview Print"],
           "zh": ["恒达供应", "北辰零件", "青松办公", "东岳实验", "万象工具", "明远印务"]}
CATEGORIES = {"en": ["hardware", "software", "services", "repairs"], "zh": ["硬件", "软件", "咨询服务", "维修服务"]}


def approval_level(facts: dict, level: int) -> tuple[int, list[int]]:
    """Apply rules 1..(level + 1 or 5) in order; return the final level and the level after each rule."""
    base = sum(facts["amount"] > limit for limit in facts["limits"])
    trace = [base]
    exempt = facts["preapproved"] and facts["category"] != facts["excluded_category"] and not facts.get("emergency")
    trace.append(max(trace[-1] - 1, 0) if exempt else trace[-1])
    if level >= 2:
        trace.append(max(trace[-1], 2) if facts["restricted"] else trace[-1])
    if level >= 3:
        trace.append(min(trace[-1] + 1, 3) if facts.get("term_months", 0) > 12 else trace[-1])
    if level >= 4:
        trace.append(min(trace[-1], 1) if facts["framework"] and not facts["restricted"] else trace[-1])
    return trace[-1], trace


def rules(rng: random.Random, lang: str, level: int, index: int):
    want = _answer_class(index, 4)
    vendors, categories = VENDORS[lang], CATEGORIES[lang]
    while True:
        scale = rng.choice((1, 2, 5))
        limits = [1000 * scale, 5000 * scale, 20000 * scale]
        tier = rng.randrange(4)
        low, high = ([0] + limits)[tier], (limits + [limits[-1] * 3])[tier]
        amount = low if tier and rng.random() < 0.25 else rng.randrange(low + 10, high + 1, 10)  # boundary is inclusive
        preapproved = rng.sample(vendors, 3)
        framework = rng.sample(vendors, 2)
        restricted_projects = sorted(rng.sample([f"P-{n}" for n in range(10, 40)], 2))
        vendor, category = rng.choice(vendors), rng.choice(categories)
        project = rng.choice(restricted_projects) if rng.random() < 0.4 else f"P-{rng.randrange(10, 40)}"
        facts = {"amount": amount, "limits": limits, "vendor": vendor, "category": category, "project": project,
                 "preapproved": vendor in preapproved, "excluded_category": rng.choice(categories[:2]),
                 "restricted": level >= 2 and project in restricted_projects}
        if level >= 3 and category == categories[1]:
            facts["term_months"] = rng.choice((6, 12, 24, 36))
        if level >= 4:
            facts["framework"] = vendor in framework
            facts["emergency"] = category == categories[3] and rng.random() < 0.6
        answer, trace = approval_level(facts, level)
        # Two rows in three need a later rule to change the level; below level 3 only rule 1 reaches VP.
        if answer == want and (index % 3 == 0 or len(set(trace)) > 1 or (level <= 2 and want == 3)):
            break
    money = (lambda v: f"${v:,}") if lang == "en" else (lambda v: f"{v:,}元")
    l1, l2, l3 = (money(v) for v in limits)
    if lang == "en":
        lines = ["Purchasing policy. Apply the rules in order; each rule adjusts the approval level reached so far. "
                 "Levels from lowest to highest: no approval, manager, director, VP.",
                 f"Rule 1: Purchases up to {l1} need no approval; above {l1} and up to {l2}, manager approval; "
                 f"above {l2} and up to {l3}, director approval; above {l3}, VP approval.",
                 "Rule 2: If the vendor is pre-approved, lower the level by one step (not below no approval), except "
                 f"for {facts['excluded_category']} purchases."]
        if level >= 2:
            lines.append("Rule 3: If the purchase is charged to a restricted project, the level is at least director.")
        if level >= 3:
            lines.append("Rule 4: If the purchase is software with a term longer than 12 months, raise the level by "
                         "one step (not above VP).")
        if level >= 4:
            lines += ["Rule 5: If the vendor has an active framework contract, the level is at most manager; this rule "
                      "does not apply to restricted projects.", "Rule 6: Rule 2 does not apply to emergency repairs."]
        lines.append(f"Pre-approved vendors: {', '.join(sorted(preapproved))}.")
        if level >= 2:
            lines.append(f"Restricted projects: {', '.join(restricted_projects)}.")
        if level >= 4:
            lines.append(f"Vendors with active framework contracts: {', '.join(sorted(framework))}.")
        request = f"Request: {money(amount)} of {category} from {vendor}, charged to project {project}"
        if "term_months" in facts:
            request += f", term {facts['term_months']} months"
        if facts.get("emergency"):
            request += ", emergency repair"
        lines.append(request + ".")
        question = "What approval does this request need under the policy?"
    else:
        lines = ["采购审批制度。按顺序执行各条规则，每条规则都在前面已得出的审批级别上调整。级别从低到高依次为：无需审批、部门经理、总监、副总裁。",
                 f"规则1：金额不超过{l1}的无需审批；超过{l1}、不超过{l2}的需部门经理审批；"
                 f"超过{l2}、不超过{l3}的需总监审批；超过{l3}的需副总裁审批。",
                 f"规则2：供应商在预批准名单内的，审批级别降低一级（最低为无需审批），但{facts['excluded_category']}类采购除外。"]
        if level >= 2:
            lines.append("规则3：记入受限项目的采购，审批级别至少为总监。")
        if level >= 3:
            lines.append("规则4：期限超过12个月的软件采购，审批级别提高一级（最高为副总裁）。")
        if level >= 4:
            lines += ["规则5：与供应商签有有效框架合同的，审批级别最高为部门经理；但本规则不适用于受限项目。",
                      "规则6：紧急维修不适用规则2。"]
        lines.append(f"预批准供应商：{'、'.join(sorted(preapproved))}。")
        if level >= 2:
            lines.append(f"受限项目：{'、'.join(restricted_projects)}。")
        if level >= 4:
            lines.append(f"签有有效框架合同的供应商：{'、'.join(sorted(framework))}。")
        request = f"申请：向{vendor}采购{category}，金额{money(amount)}，记入项目{project}"
        if "term_months" in facts:
            request += f"，期限{facts['term_months']}个月"
        if facts.get("emergency"):
            request += "，属于紧急维修"
        lines.append(request + "。")
        question = "按照这项制度，这笔申请需要哪一级审批？"
    options = [_opt(key, text) for key, text in zip(("none", "manager", "director", "vp"), APPROVAL[lang])]
    derivation = {"facts": facts, "levels_after_each_rule": trace, "rules_applied": len(trace)}
    return "\n".join(lines), question, options[answer], [o for i, o in enumerate(options) if i != answer], derivation


# Stacked negation and quantifiers over a few items; the label is decided over all worlds.

NEGATION_LEVELS = {  # items, statements, statement kinds, outer negations
    1: (2, 2, ("lit",), (0, 2)),
    2: (3, 3, ("lit", "all", "none"), (0, 2)),
    3: (4, 3, ("lit", "all", "none", "some"), (1, 3)),
    4: (5, 4, ("lit", "all", "none", "some", "exactly"), (1, 3)),
}
NEGATION_DOMAINS = {
    "en": [("crate", "crates", "sealed"), ("report", "reports", "signed"), ("room", "rooms", "cleaned"),
           ("valve", "valves", "tested")],
    "zh": [("号箱", "已封装", "没有封装", "封装"), ("号报告", "已签字", "没有签字", "签字"),
           ("号房间", "已打扫", "没有打扫", "打扫"), ("号阀门", "已检测", "没有检测", "检测")],
}


def statement_holds(statement: dict, world: tuple) -> bool:
    kind, items = statement["kind"], statement["items"]
    value = {"lit": lambda: world[items[0]] == statement.get("positive", True), "all": lambda: all(world[i] for i in items),
             "none": lambda: not any(world[i] for i in items), "some": lambda: any(world[i] for i in items),
             "exactly": lambda: sum(world[i] for i in items) == statement.get("k")}[kind]()
    return value != bool(statement["outer"] % 2)


def _statement_text(lang: str, domain: tuple, statement: dict, rng: random.Random) -> str:
    kind, names = statement["kind"], [LETTERS[i] for i in statement["items"]]
    if lang == "en":
        noun, plural, verb = domain
        group = f"{plural} {_join(lang, names)}"
        text = {"lit": lambda: f"{noun} {names[0]} was {'' if statement.get('positive') else 'not '}{verb}",
                "all": lambda: f"all of {group} were {verb}", "none": lambda: f"none of {group} was {verb}",
                "some": lambda: f"at least one of {group} was {verb}",
                "exactly": lambda: f"exactly {EN_NUMBERS[statement.get('k', 0)]} of {group} were {verb}"}[kind]()
        for _ in range(statement["outer"]):
            text = f"{rng.choice(('it is not the case that', 'it is false that'))} {text}"
        return text[0].upper() + text[1:] + "."
    suffix, done, undone, verb = domain
    group = f"{'、'.join(names)}{suffix}"
    text = {"lit": lambda: f"{names[0]}{suffix}{done if statement.get('positive') else undone}",
            "all": lambda: f"{group}都{done}", "none": lambda: f"{group}都{undone}",
            "some": lambda: f"{group}中至少有一个{done}",
            "exactly": lambda: f"{group}中恰好有{ZH_NUMBERS[statement.get('k', 0)]}个{done}"}[kind]()
    for _ in range(statement["outer"]):  # nested quotes alternate “” and ‘’ so the scope is explicit
        inner = text.translate(str.maketrans("“”‘’", "‘’“”"))
        text = f"“{inner}”{rng.choice(('这一说法是假的', '这种说法不成立'))}"
    return text + "。"


def negation(rng: random.Random, lang: str, level: int, index: int):
    want = _answer_class(index, 3)  # 0 yes, 1 no, 2 undetermined
    size, count, kinds, (low, high) = NEGATION_LEVELS[level]
    worlds = list(itertools.product((False, True), repeat=size))
    while True:
        statements = []
        for _ in range(count):
            kind = rng.choice(kinds)
            items = [rng.randrange(size)] if kind == "lit" else sorted(rng.sample(range(size), rng.randint(2, min(4, size))))
            statement = {"kind": kind, "items": items, "outer": rng.randint(low, high)}
            if kind == "lit":
                statement["positive"] = rng.random() < 0.5
            if kind == "exactly":
                statement["k"] = rng.randint(1, len(items) - 1)
            statements.append(statement)
        query = rng.randrange(size)
        if level > 1 and not any(query in s["items"] for s in statements):  # literals alone settle what they mention
            continue
        consistent = [w for w in worlds if all(statement_holds(s, w) for s in statements)]
        values = {w[query] for w in consistent}
        answer = {frozenset({True}): 0, frozenset({False}): 1, frozenset({True, False}): 2}.get(frozenset(values))
        if answer == want:
            break
    domain = rng.choice(NEGATION_DOMAINS[lang])
    texts = [_statement_text(lang, domain, s, rng) for s in statements]
    item = LETTERS[query]
    if lang == "en":
        noun, _, verb = domain
        state = "Inspection notes. Every statement below is true.\n" + "\n".join(
            f"Statement {n}: {text}" for n, text in enumerate(texts, 1))
        question = f"Given these statements, was {noun} {item} {verb}?"
        options = [_opt("yes", f"Yes, {noun} {item} was {verb}"), _opt("no", f"No, {noun} {item} was not {verb}"),
                   _opt("undetermined", "The statements do not settle it")]
    else:
        suffix, done, undone, verb = domain
        state = "检查记录（以下每条说法都属实）：\n" + "\n".join(f"说法{n}：{text}" for n, text in enumerate(texts, 1))
        question = f"根据这些说法，{item}{suffix}是否{done}？"
        options = [_opt("yes", f"是，{item}{suffix}{done}"), _opt("no", f"否，{item}{suffix}{undone}"),
                   _opt("undetermined", "仅凭这些说法无法确定")]
    derivation = {"items": size, "statements": statements, "query": query, "consistent_worlds": len(consistent),
                  "negations": sum(s["outer"] + (s["kind"] in ("lit", "none") and not s.get("positive", False))
                                   for s in statements)}
    return state, question, options[answer], [o for i, o in enumerate(options) if i != answer], derivation


# Multi-step arithmetic; distractors are the results of typical slips.

def arithmetic(rng: random.Random, lang: str, level: int, index: int):
    if level == 1:
        while True:
            a, b = rng.randint(6, 24), rng.randint(3, 12)
            c = rng.randint(1, a - 1)
            d = rng.randint(a, a * b)
            answer = a * b + c - d
            if answer > 3:
                break
        slips = [a * (b + c) - d, a * b - d, a * b + c + d, answer + 1, answer - 1]
        inputs = {"per_tray": a, "trays": b, "extra": c, "sold": d}
        state = _t(lang, f"A bakery puts {a} rolls on each tray. This morning it baked {b} full trays plus {c} extra rolls, "
                         f"then sold {d} rolls.",
                   f"一家面包店每个托盘摆放{a}个面包。今天早上烤了{b}满盘，另外还有{c}个单独的面包，随后卖出了{d}个。")
        question = _t(lang, "How many rolls are left?", "还剩多少个面包？")
        unit = _t(lang, "{} roll|rolls", "{}个")
    elif level == 2:
        while True:
            p, q = rng.randint(3, 40), rng.randint(10, 60)
            d, s = rng.choice((10, 15, 20, 25, 30, 40)), rng.randint(5, 40)
            if p * q * d % 100 == 0:
                break
        answer = p * q * (100 - d) // 100 + s
        slips = [p * q + s, p * q * (100 - d) // 100, p * q - d + s, answer + s]
        if (p * q + s) * (100 - d) % 100 == 0:
            slips.insert(0, (p * q + s) * (100 - d) // 100)
        inputs = {"price": p, "quantity": q, "discount_percent": d, "delivery": s}
        state = _t(lang, f"Notebooks cost ${p} each. A school orders {q} of them. The store takes {d}% off the notebook "
                         f"total and then adds a flat ${s} delivery fee.",
                   f"笔记本每本{p}元，一所学校订购了{q}本。商店对笔记本总价优惠{d}%，之后另收固定配送费{s}元。")
        question = _t(lang, "How much does the school pay in total?", "学校一共要付多少钱？")
        unit = _t(lang, "${}", "{}元")
    elif level == 3:
        x, y, t = rng.randint(2, 15), rng.randint(2, 15), rng.randint(3, 20)
        answer, c0 = rng.randint(4, 40), rng.randrange(10, 300, 10)
        capacity = c0 + x * t + answer * (x + y)
        rest = capacity - c0 - x * t
        slips = [answer + t, answer + 1, answer - 1]
        for numerator, rate in ((capacity - x * t, x + y), (capacity - c0, x + y), (rest, y)):
            if numerator % rate == 0:
                slips.insert(0, numerator // rate)
        inputs = {"capacity": capacity, "initial": c0, "rate_a": x, "rate_b": y, "alone_minutes": t}
        state = _t(lang, f"A tank holds {capacity} liters and already contains {c0} liters. Pump A adds {x} liters per "
                         f"minute. Pump A runs alone for {t} minutes; then pump B, which adds {y} liters per minute, is "
                         "switched on and both run together until the tank is full.",
                   f"一个水箱容量为{capacity}升，里面已有{c0}升水。A泵每分钟注水{x}升。A泵先单独运行{t}分钟，"
                   f"然后打开每分钟注水{y}升的B泵，两台泵一起运行直到水箱注满。")
        question = _t(lang, "How many minutes after pump B is switched on does the tank become full?",
                      "打开B泵之后再过多少分钟水箱注满？")
        unit = _t(lang, "{} minute|minutes", "{}分钟")
    else:
        while True:
            start, orders, each = rng.randint(200, 600), rng.randint(3, 12), rng.randint(5, 20)
            returned = rng.randint(5, 40)
            damaged = rng.randint(1, returned - 1)
            percent, case = rng.choice((10, 20, 25, 40, 50)), rng.randint(12, 50)
            before = start - orders * each + returned - damaged
            if before <= 0 or before * percent % 100:
                continue
            after = before * (100 - percent) // 100
            target = after + rng.randint(3 * case + 1, 8 * case)
            need = target - after
            if need % case:
                break
        answer = -(-need // case)
        careless = before + damaged
        slips = [need // case, -(-(target - careless * (100 - percent) // 100) // case), -(-target // case), answer + 1]
        if start * percent % 100 == 0:
            slips.insert(1, -(-(target - (before - start * percent // 100)) // case))
        inputs = {"start": start, "orders": orders, "units_per_order": each, "returned": returned, "damaged": damaged,
                  "sold_percent": percent, "target": target, "case_size": case}
        state = _t(lang, f"A warehouse starts the week with {start} units of a part. On Monday it ships {orders} orders of "
                         f"{each} units each. On Tuesday {returned} units are returned, but {damaged} of the returned units "
                         f"are damaged and thrown away. On Wednesday it sells {percent}% of the units then in stock. It "
                         f"must then restock to at least {target} units, ordering only whole cases of {case} units.",
                   f"一个仓库周初有某种零件{start}件。周一发出{orders}笔订单，每笔{each}件。周二退回{returned}件，"
                   f"其中{damaged}件损坏被丢弃。周三卖出当时库存的{percent}%。之后必须把库存补到至少{target}件，"
                   f"而且只能按整箱订货，每箱{case}件。")
        question = _t(lang, "What is the smallest number of cases it must order?", "最少需要订多少箱？")
        unit = _t(lang, "{} case|cases", "{}箱")
    wrong = _ranked(answer, slips, _answer_class(index, 4), 1, 1)
    options = [_opt(f"v{v}", _unit(unit, v, lang)) for v in [answer] + wrong]
    return state, question, options[0], options[1:], {"inputs": inputs, "value": answer}


# Logic grids: a unique solution is guaranteed by brute force over all assignments.

GRID_LEVELS = {1: (3, ("pet", "drink")), 2: (4, ("pet", "drink")), 3: (4, ("pet", "drink", "hobby")), 4: (5, ("seat", "pet"))}
GRID_PEOPLE = {"en": ["Ana", "Ben", "Chloe", "Dev", "Elif", "Farid", "Grace", "Hugo", "Iris", "Jonas"],
               "zh": ["王磊", "李娜", "张伟", "刘洋", "陈静", "杨帆", "赵敏", "黄涛", "周婷", "吴昊"]}
GRID_VALUES = {
    "en": {"pet": ["cat", "dog", "parrot", "rabbit", "turtle"], "drink": ["tea", "coffee", "juice", "milk", "cocoa"],
           "hobby": ["chess", "hiking", "painting", "cycling", "baking"]},
    "zh": {"pet": ["猫", "狗", "鹦鹉", "兔子", "乌龟"], "drink": ["茶", "咖啡", "果汁", "牛奶", "可可"],
           "hobby": ["下棋", "徒步", "画画", "骑行", "烘焙"]},
}
GRID_PHRASES = {  # is, is not, either, "the one who", ask who, ask what, option label
    "en": {"pet": ("{s} owns the {v}", "{s} does not own the {v}", "{s} owns either the {v} or the {w}", "the {v} owner",
                   "Who owns the {v}?", "Which pet does {s} own?", "the {v}"),
           "drink": ("{s} drinks {v}", "{s} does not drink {v}", "{s} drinks either {v} or {w}", "the {v} drinker",
                     "Who drinks {v}?", "What does {s} drink?", "{v}"),
           "hobby": ("{s}'s hobby is {v}", "{s}'s hobby is not {v}", "{s}'s hobby is either {v} or {w}",
                     "the person whose hobby is {v}", "Whose hobby is {v}?", "What is {s}'s hobby?", "{v}"),
           "seat": ("{s} sits in seat {v}", "{s} does not sit in seat {v}", "{s} sits in seat {v} or seat {w}",
                    "the person in seat {v}", "Who sits in seat {v}?", "In which seat does {s} sit?", "seat {v}")},
    "zh": {"pet": ("{s}养{v}", "{s}不养{v}", "{s}养的是{v}或{w}", "养{v}的人", "谁养{v}？", "{s}养的是哪种宠物？", "{v}"),
           "drink": ("{s}喝{v}", "{s}不喝{v}", "{s}喝的是{v}或{w}", "喝{v}的人", "谁喝{v}？", "{s}喝的是什么？", "{v}"),
           "hobby": ("{s}的爱好是{v}", "{s}的爱好不是{v}", "{s}的爱好是{v}或{w}", "爱好{v}的人", "谁的爱好是{v}？",
                     "{s}的爱好是什么？", "{v}"),
           "seat": ("{s}坐在{v}号座位", "{s}不坐在{v}号座位", "{s}坐在{v}号或{w}号座位", "坐在{v}号座位的人",
                    "谁坐在{v}号座位？", "{s}坐在几号座位？", "{v}号座位")},
}
SEAT_PHRASES = {
    "en": {"left": "{a} sits immediately to the left of {b}", "before": "{a} sits somewhere to the left of {b}",
           "adjacent": "{a} and {b} sit next to each other", "apart": "{a} and {b} do not sit next to each other",
           "end": "{a} sits at one end of the row"},
    "zh": {"left": "{a}坐在{b}左边紧挨着的座位上", "before": "{a}坐在{b}左边的某个座位上（不一定相邻）",
           "adjacent": "{a}和{b}的座位相邻", "apart": "{a}和{b}的座位不相邻", "end": "{a}坐在这一排的某一端"},
}
_GRID_CACHE: dict = {}


def _grid_setup(n: int, attrs: tuple):
    """All assignments, every candidate clue, and each clue's truth in every assignment (cached)."""
    if (n, attrs) not in _GRID_CACHE:
        perms = np.array(list(itertools.permutations(range(n))), dtype=np.int8)
        index = np.array(list(itertools.product(range(len(perms)), repeat=len(attrs))))
        worlds = perms[index]  # (worlds, attributes, people): value index held by each person
        holders = np.argsort(worlds, axis=2)  # (worlds, attributes, values): person holding each value
        seat = attrs.index("seat") if "seat" in attrs else None
        pool = _grid_pool(n, attrs)
        masks = np.stack([_grid_mask(clue, worlds, holders, seat) for clue in pool])
        _GRID_CACHE[n, attrs] = (worlds, pool, masks)
    return _GRID_CACHE[n, attrs]


def _grid_mask(clue: list, worlds, holders, seat: int | None) -> np.ndarray:
    rows = np.arange(len(worlds))
    kind = clue[0]
    if kind in ("is", "not", "either"):
        held = worlds[:, clue[2], clue[1]]
        mask = held == clue[3] if kind != "either" else (held == clue[3]) | (held == clue[4])
        return ~mask if kind == "not" else mask
    if kind in ("link", "unlink"):
        mask = worlds[rows, clue[3], holders[:, clue[1], clue[2]]] == clue[4]
        return ~mask if kind == "unlink" else mask

    def place(ref):  # seat index of a named person or of the holder of a value
        person = ref[1] if ref[0] == "person" else holders[:, ref[1], ref[2]]
        return worlds[rows, seat, person].astype(int)

    a = place(clue[1])
    if kind == "end":
        return (a == 0) | (a == worlds.shape[2] - 1)
    b = place(clue[2])
    return {"left": a + 1 == b, "before": a < b, "adjacent": abs(a - b) == 1, "apart": abs(a - b) > 1}[kind]


def _grid_pool(n: int, attrs: tuple) -> list[list]:
    pool, m = [], len(attrs)
    for a, p, v in itertools.product(range(m), range(n), range(n)):
        pool += [["is", p, a, v], ["not", p, a, v]]
        pool += [["either", p, a, v, w] for w in range(v + 1, n)]
    for a1, a2 in itertools.permutations(range(m), 2):
        if a1 < a2:
            pool += [[kind, a1, v1, a2, v2] for kind in ("link", "unlink") for v1 in range(n) for v2 in range(n)]
    if "seat" in attrs:
        pet = attrs.index("pet")
        refs = [["person", p] for p in range(n)] + [["holder", pet, v] for v in range(n)]
        pool += [["end", r] for r in refs]
        for r1, r2 in itertools.permutations(refs, 2):
            if r1[0] == "person" or r2[0] == "person":
                pool += [[kind, r1, r2] for kind in ("left", "before", "adjacent")]
            if r1[0] == r2[0] == "person" and r1[1] < r2[1]:
                pool.append(["apart", r1, r2])
    return pool


def _grid_text(lang: str, clue: list, attrs: tuple, people: list, values: dict) -> str:
    def value(a, v):
        return str(v + 1) if attrs[a] == "seat" else values[attrs[a]][v]

    def ref(r):
        return people[r[1]] if r[0] == "person" else GRID_PHRASES[lang][attrs[r[1]]][3].format(v=value(r[1], r[2]))

    kind = clue[0]
    if kind in ("is", "not", "either"):
        phrase = GRID_PHRASES[lang][attrs[clue[2]]][("is", "not", "either").index(kind)]
        text = phrase.format(s=people[clue[1]], v=value(clue[2], clue[3]),
                             w=value(clue[2], clue[4]) if kind == "either" else "")
    elif kind in ("link", "unlink"):
        subject = GRID_PHRASES[lang][attrs[clue[1]]][3].format(v=value(clue[1], clue[2]))
        text = GRID_PHRASES[lang][attrs[clue[3]]][0 if kind == "link" else 1].format(s=subject, v=value(clue[3], clue[4]))
    else:
        text = SEAT_PHRASES[lang][kind].format(a=ref(clue[1]), b=ref(clue[2]) if len(clue) > 2 else "")
    return (text[0].upper() + text[1:] + ".") if lang == "en" else text + "。"


def logic_grid(rng: random.Random, lang: str, level: int, index: int):
    n, attrs = GRID_LEVELS[level]
    worlds, pool, masks = _grid_setup(n, attrs)
    seat = attrs.index("seat") if "seat" in attrs else None
    people = rng.sample(GRID_PEOPLE[lang], n)
    values = {a: sorted(rng.sample(range(5), n)) for a in attrs if a != "seat"}
    values = {a: [GRID_VALUES[lang][a][i] for i in chosen] for a, chosen in values.items()}
    solution = rng.randrange(len(worlds))
    ask_attr = rng.randrange(len(attrs))
    ask_person = rng.randrange(n)
    ask_value = int(worlds[solution, ask_attr, ask_person])
    ask_who = rng.random() < 0.5
    true = [i for i in np.flatnonzero(masks[:, solution]).tolist()
            if not (pool[i][0] == "is" and pool[i][2] == ask_attr and pool[i][1] == ask_person)]
    direct = [i for i in true if pool[i][0] == "is"]
    rest = [i for i in true if pool[i][0] != "is"]
    rng.shuffle(direct)
    rng.shuffle(rest)
    alive, chosen = np.ones(len(worlds), dtype=bool), []
    for i in rest + direct:  # direct placements only when the indirect clues run out
        narrowed = alive & masks[i]
        if narrowed.sum() < alive.sum():
            chosen.append(i)
            alive = narrowed
        if alive.sum() == 1:
            break
    for i in sorted(chosen, key=lambda _: rng.random()):  # drop clues the others make redundant
        others = [j for j in chosen if j != i]
        if np.logical_and.reduce(masks[others]).sum() == 1:
            chosen = others
    chosen = [pool[i] for i in chosen]
    rng.shuffle(chosen)
    phrases = GRID_PHRASES[lang][attrs[ask_attr]]
    label = (lambda a, v: str(v + 1)) if attrs[ask_attr] == "seat" else (lambda a, v: values[attrs[a]][v])
    if ask_who:
        question = phrases[4].format(v=label(ask_attr, ask_value))
        options = [_opt(f"p{p}", people[p]) for p in range(n)]
        answer = ask_person
    else:
        question = phrases[5].format(s=people[ask_person])
        options = [_opt(f"v{v}", phrases[6].format(v=label(ask_attr, v))) for v in range(n)]
        answer = ask_value
    if lang == "en":
        listing = {a: f"({', '.join(values[a])})" for a in values}
        nouns = {"pet": "a different pet", "drink": "prefers a different drink", "hobby": "has a different hobby"}
        if seat is not None:
            intro = (f"{_join(lang, people)} sit in a row of {EN_NUMBERS[n]} seats, numbered 1 to {n} from left to "
                     f"right, one person per seat. Each owns a different pet {listing['pet']}.")
        else:
            parts = [f"owns {nouns['pet']} {listing['pet']}"] + [f"{nouns[a]} {listing[a]}" for a in attrs[1:]]
            intro = f"{_join(lang, people)} are friends. Each {_join(lang, parts)}."
        state = intro + "\nClues:\n" + "\n".join(f"{k}. {_grid_text(lang, c, attrs, people, values)}"
                                                  for k, c in enumerate(chosen, 1))
    else:
        listing = {a: f"（{'、'.join(values[a])}）" for a in values}
        if seat is not None:
            intro = (f"{'、'.join(people)}{ZH_NUMBERS[n]}人坐在一排{ZH_NUMBERS[n]}个座位上，座位从左到右编号为1到{n}，"
                     f"每个座位坐一人。每人养一种不同的宠物{listing['pet']}。")
        else:
            nouns = {"pet": "养一种不同的宠物", "drink": "最爱喝一种不同的饮品", "hobby": "有一种不同的爱好"}
            intro = f"{'、'.join(people)}是朋友，每人{'，每人'.join(nouns[a] + listing[a] for a in attrs)}。"
        state = intro + "\n线索：\n" + "\n".join(f"{k}. {_grid_text(lang, c, attrs, people, values)}"
                                               for k, c in enumerate(chosen, 1))
    derivation = {"people": people, "attributes": list(attrs),
                  "solution": worlds[solution].astype(int).tolist(), "clues": chosen,
                  "question": {"attribute": ask_attr, "person": ask_person, "value": ask_value,
                               "ask": "who" if ask_who else "what"}}
    return state, question, options[answer], [o for i, o in enumerate(options) if i != answer], derivation


# Graphs: reachability over one-way roads, then shortest travel times.

TOWNS = {"en": ["Alder", "Birch", "Cedar", "Dunmore", "Elm", "Fenwick", "Garnet", "Hollow", "Ivy", "Juniper", "Kestrel",
                "Linden"],
         "zh": ["安平", "白河", "长岭", "东坝", "凤鸣", "观山", "河口", "景阳", "柳湾", "梅岭", "南屏", "清溪"]}


def _reach(edges: list, source: int) -> dict[int, int]:
    distance, frontier = {source: 0}, [source]
    while frontier:
        nxt = []
        for node in frontier:
            for a, b, *_ in edges:
                if a == node and b not in distance:
                    distance[b] = distance[node] + 1
                    nxt.append(b)
        frontier = nxt
    return distance


def _dijkstra(edges: list, source: int, target: int, closed=None) -> float:
    best, heap = {source: 0}, [(0, source)]
    while heap:
        cost, node = heapq.heappop(heap)
        if node == target:
            return cost
        if cost > best.get(node, float("inf")):
            continue
        for a, b, w in edges:
            if a == node and [a, b] != closed and cost + w < best.get(b, float("inf")):
                best[b] = cost + w
                heapq.heappush(heap, (cost + w, b))
    return float("inf")


def _paths(edges: list, source: int, target: int, closed=None) -> list[tuple[int, int]]:
    """(total, hops) of every simple path."""
    found, stack = [], [(source, 0, (source,))]
    while stack:
        node, cost, seen = stack.pop()
        if node == target:
            found.append((cost, len(seen) - 1))
            continue
        for a, b, w in edges:
            if a == node and b not in seen and [a, b] != closed:
                stack.append((b, cost + w, seen + (b,)))
    return found


def _random_edges(rng: random.Random, n: int, m: int, weighted: bool) -> list[list[int]]:
    pairs = rng.sample([(a, b) for a in range(n) for b in range(n) if a != b], m)
    return [[a, b, rng.randint(3, 30)] if weighted else [a, b] for a, b in pairs]


def graph(rng: random.Random, lang: str, level: int, index: int):
    towns = TOWNS[lang]
    minutes = _t(lang, "{} minutes", "{}分钟")
    if level <= 2:
        n, m, hops = (5, 6, 2) if level == 1 else (8, 12, 3)
        want = _answer_class(index, 2)  # 0 reachable, 1 not reachable
        while True:
            edges = _random_edges(rng, n, m, False)
            source, target = rng.sample(range(n), 2)
            distance = _reach(edges, source)
            if want == 0 and distance.get(target, 0) >= hops:
                break
            if want == 1 and target not in distance and len(distance) >= level + 1 and any(b == target for _, b in edges):
                break
        s, t = towns[source], towns[target]
        roads = "\n".join(f"{towns[a]} → {towns[b]}" for a, b in edges)
        state = _t(lang, f"One-way roads (each can be driven only in the direction shown):\n{roads}",
                   f"单行道（只能按箭头方向行驶）：\n{roads}")
        question = _t(lang, f"Using only these roads, can you drive from {s} to {t}?",
                      f"只走这些道路，能否从{s}开车到达{t}？")
        options = [_opt("yes", _t(lang, f"Yes, {t} can be reached from {s}", f"能，从{s}可以到达{t}")),
                   _opt("no", _t(lang, f"No, {t} cannot be reached from {s}", f"不能，从{s}无法到达{t}"))]
        derivation = {"nodes": towns[:n], "edges": edges, "source": source, "target": target,
                      "hops": distance.get(target)}
        return state, question, options[want], [options[1 - want]], derivation
    if level == 3:
        while True:
            edges = _random_edges(rng, 7, 12, True)
            source, target = rng.sample(range(7), 2)
            paths = _paths(edges, source, target)
            if len(paths) < 2:
                continue
            answer = min(total for total, _ in paths)
            fewest = min(hops for _, hops in paths)
            naive = min(total for total, hops in paths if hops == fewest)
            if naive != answer:
                break
        others = sorted({total for total, _ in paths} - {answer})
        wrong = _ranked(answer, [naive] + others, _answer_class(index, 4), 1, 1)
        roads = "\n".join(f"{towns[a]} → {towns[b]}: " + minutes.format(w) for a, b, w in edges)
        s, t = towns[source], towns[target]
        state = _t(lang, f"One-way roads with driving times (each can be driven only in the direction shown):\n{roads}",
                   f"单行道及行驶时间（只能按箭头方向行驶）：\n{roads}")
        question = _t(lang, f"What is the shortest possible driving time from {s} to {t}?",
                      f"从{s}开车到{t}，最短需要多少分钟？")
        derivation = {"nodes": towns[:7], "edges": edges, "source": source, "target": target, "via": None,
                      "closed": None, "fewest_roads_time": naive}
    else:
        while True:
            edges = _random_edges(rng, 10, 18, True)
            source, via, target = rng.sample(range(10), 3)
            open_route = _dijkstra(edges, source, via) + _dijkstra(edges, via, target)
            if open_route == float("inf"):
                continue
            closed = rng.choice(edges)[:2]
            answer = _dijkstra(edges, source, via, closed) + _dijkstra(edges, via, target, closed)
            direct = _dijkstra(edges, source, target, closed)
            if answer == float("inf") or direct == float("inf") or len({answer, direct, open_route}) < 3:
                continue
            combos = {a + b for a, _ in _paths(edges, source, via, closed) for b, _ in _paths(edges, via, target, closed)}
            break
        wrong = _ranked(answer, [open_route, direct] + sorted(combos - {answer}), _answer_class(index, 4), 1, 1)
        roads = "\n".join(f"{towns[a]} → {towns[b]}: " + minutes.format(w) for a, b, w in edges)
        s, v, t = towns[source], towns[via], towns[target]
        c = f"{towns[closed[0]]} → {towns[closed[1]]}"
        state = _t(lang, f"One-way roads with driving times (each can be driven only in the direction shown):\n{roads}\n"
                         f"Today the road {c} is closed.",
                   f"单行道及行驶时间（只能按箭头方向行驶）：\n{roads}\n今天{c}这条路封闭。")
        question = _t(lang, f"A courier must drive from {s} to {t}, stopping at {v} on the way. What is the shortest "
                            "possible total driving time?",
                      f"快递员要从{s}出发，途经{v}取件，最后到达{t}。总行驶时间最短是多少分钟？")
        derivation = {"nodes": towns[:10], "edges": edges, "source": source, "target": target, "via": via,
                      "closed": closed, "ignoring_closure": open_route, "ignoring_stop": direct}
    options = [_opt(f"m{v}", minutes.format(v)) for v in [answer] + wrong]
    return state, question, options[0], options[1:], {**derivation, "minutes": answer}


# Scheduling: interval overlap, shared calendars, capacity, and time zones.

MEETINGS = {"en": ["Budget review", "Design sync", "Hiring panel", "Vendor call", "Team retro", "Client demo"],
            "zh": ["预算评审", "设计同步会", "招聘面试", "供应商电话会", "团队复盘", "客户演示"]}
ROOMS = {"en": ["Juniper", "Harbor", "Summit", "Maple"], "zh": ["青松厅", "海港厅", "峰顶厅", "枫叶厅"]}


def _overlap(a: list, b: list) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _blocks(rng: random.Random, count: int, start: int = 540, end: int = 1020, step: int = 15) -> list[list[int]]:
    """Non-overlapping busy blocks inside [start, end) on a `step`-minute grid."""
    while True:
        blocks = []
        for _ in range(count):
            begin = rng.randrange(start, end - 30, step)
            blocks.append([begin, min(end, begin + rng.randrange(30, 121, step))])
        blocks.sort()
        if all(a[1] <= b[0] for a, b in zip(blocks, blocks[1:])):
            return blocks


def _slot(lang: str, interval: list, suffix: str = "") -> str:
    return f"{_clock(interval[0])}–{_clock(interval[1])}{suffix}"


def scheduling(rng: random.Random, lang: str, level: int, index: int):
    if level == 1:
        want = _answer_class(index, 2)  # 0 overlaps, 1 free
        while True:
            bookings = _blocks(rng, rng.randint(3, 4), 480, 1080)
            begin = rng.randrange(480, 1020, 15)
            proposal = [begin, begin + rng.randrange(30, 91, 15)]
            worst = max(_overlap(proposal, b) for b in bookings)
            touches = any(proposal[0] == b[1] or proposal[1] == b[0] for b in bookings)
            if proposal[1] <= 1080 and ((want == 0 and 0 < worst <= 30) or (want == 1 and worst == 0 and touches)):
                break
        titles = rng.sample(MEETINGS[lang], len(bookings))
        listing = "\n".join(f"{_slot(lang, b)} {title}" for b, title in zip(bookings, titles))
        room = rng.choice(ROOMS[lang])
        state = _t(lang, f"Bookings for room {room} on Tuesday:\n{listing}\nA meeting may start at the exact minute "
                         f"another one ends.\nProposed meeting: {_slot(lang, proposal)}.",
                   f"会议室“{room}”周二的预订：\n{listing}\n一场会议可以在另一场结束的同一时刻开始。\n"
                   f"拟定会议：{_slot(lang, proposal)}。")
        question = _t(lang, "Does the proposed meeting overlap an existing booking?", "拟定会议是否与已有预订时间重叠？")
        options = [_opt("overlap", _t(lang, "Yes, it overlaps an existing booking", "是，与已有预订重叠")),
                   _opt("free", _t(lang, "No, the room is free for the whole meeting", "否，整个会议期间会议室空闲"))]
        derivation = {"bookings": bookings, "proposal": proposal, "max_overlap": worst}
        return state, question, options[want], [options[1 - want]], derivation
    if level == 2:
        people = rng.sample(GRID_PEOPLE[lang], 2)
        while True:
            busy = [_blocks(rng, rng.randint(3, 4)) for _ in people]
            length = rng.choice((30, 45, 60))
            slots = [[s, s + length] for s in range(540, 1021 - length, 15)]
            conflicts = [[any(_overlap(slot, b) for b in person) for person in busy] for slot in slots]
            good = [s for s, c in zip(slots, conflicts) if not any(c)]
            near = [s for s, c in zip(slots, conflicts) if sum(c) == 1]
            if good and len(near) >= 3:
                break
        answer, wrong = rng.choice(good), rng.sample(near, 3)
        listing = "\n".join(p + _t(lang, ": busy " + ", ".join(_slot(lang, b) for b in person),
                                    "：" + "、".join(_slot(lang, b) for b in person) + " 忙碌")
                            for p, person in zip(people, busy))
        state = _t(lang, f"Calendars for Thursday (all times local; everyone is available 09:00–17:00 except when busy):\n"
                         f"{listing}", f"周四的日程（均为当地时间；除忙碌时段外，两人09:00–17:00都有空）：\n{listing}")
        question = _t(lang, f"Which slot fits a {length}-minute meeting with both {people[0]} and {people[1]}?",
                      f"哪个时段可以安排{people[0]}和{people[1]}两人都能参加的{length}分钟会议？")
        derivation = {"busy": busy, "length": length}
    elif level == 3:
        while True:
            capacity = rng.randint(4, 8)
            reservations = []
            for _ in range(rng.randint(6, 8)):
                begin = rng.randrange(480, 1020, 30)
                reservations.append([begin, min(1080, begin + rng.randrange(60, 181, 30)), rng.randint(1, capacity - 1)])
            usage = lambda t: sum(k for a, b, k in reservations if a <= t < b)  # noqa: E731
            if max(usage(t) for t in range(480, 1080, 30)) > capacity:
                continue
            need = rng.randint(2, capacity - 1)
            windows = [[s, s + 60] for s in range(480, 1021, 30)]
            peak = [max(usage(t) for t in range(s, e, 30)) for s, e in windows]
            good = [w for w, p in zip(windows, peak) if p + need <= capacity]
            near = [w for w, p in zip(windows, peak) if p + need > capacity and usage(w[0]) + need <= capacity]
            bad = [w for w, p in zip(windows, peak) if p + need > capacity]
            if good and len(near) >= 2 and len(bad) >= 3:
                break
        answer = rng.choice(good)
        wrong = rng.sample(near, 2)
        wrong.append(rng.choice([w for w in bad if w not in wrong]))
        listing = "\n".join(_slot(lang, [a, b]) + _t(lang, f": {k} station" + "s" * (k > 1), f"：{k}台")
                            for a, b, k in reservations)
        state = _t(lang, f"A lab has {capacity} workstations. Existing reservations (a reservation holds its stations from "
                         f"its start time until its end time):\n{listing}",
                   f"一间实验室共有{capacity}台工作站。已有预约（每项预约从开始到结束一直占用所列台数）：\n{listing}")
        question = _t(lang, f"A group needs {need} workstations for one hour. Which slot can be booked without ever "
                            "exceeding the lab's capacity?",
                      f"一个小组需要连续一小时使用{need}台工作站。哪个时段可以预约且全程不超过实验室容量？")
        derivation = {"capacity": capacity, "reservations": reservations, "need": need}
    else:
        while True:
            zones = rng.sample(ZONES[4:] + ZONES[2:4], 3)
            offsets = [z[2] for z in zones]
            if max(offsets) - min(offsets) > 360:
                continue
            people = rng.sample(GRID_PEOPLE[lang], 3)
            busy = [_blocks(rng, rng.randint(1, 2)) for _ in people]
            slots = [[s, s + 60] for s in range(0, 1440 - 60, 30)
                     if all(0 <= s + o and s + o + 60 <= 1440 for o in offsets)]
            failing = [[not (540 <= s + o and e + o <= 1020) or any(_overlap([s + o, e + o], b) for b in person)
                        for o, person in zip(offsets, busy)] for s, e in slots]
            good = [s for s, f in zip(slots, failing) if not any(f)]
            near = [s for s, f in zip(slots, failing) if sum(f) == 1]
            if good and len(near) >= 3:
                break
        answer, wrong = rng.choice(good), rng.sample(near, 3)
        names = [z[0] if lang == "en" else z[1] for z in zones]
        lines = [f"{p} ({city}, {_utc(z[2])}): busy " + ", ".join(_slot(lang, b) for b in person) + " local time"
                 if lang == "en" else
                 f"{p}（{city}，{_utc(z[2])}）：当地时间 " + "、".join(_slot(lang, b) for b in person) + " 忙碌"
                 for p, city, z, person in zip(people, names, zones, busy)]
        state = _t(lang, "Everyone works 09:00–17:00 local time, and the meeting must fall entirely within each "
                         "attendee's working hours and outside their busy times. All times are on the same day. Offsets "
                         "are fixed; ignore daylight saving time.\n" + "\n".join(lines),
                   "每个人的工作时间都是当地时间09:00–17:00，会议必须完全落在每位参会者的工作时间内，并避开其忙碌时段。"
                   "所有时间都在同一天。时差固定，不考虑夏令时。\n" + "\n".join(lines))
        question = _t(lang, "Which one-hour slot (in UTC) works for all three attendees?",
                      "以下哪个一小时时段（UTC时间）三位参会者都能参加？")
        derivation = {"offsets": offsets, "busy": busy, "length": 60}
    suffix = " UTC" if level == 4 else ""
    options = [_opt(f"s{s[0]}", _slot(lang, s, suffix)) for s in [answer] + wrong]
    return state, question, options[0], options[1:], {**derivation, "slot": answer}


FAMILIES = {"dates": dates, "counting": counting, "rules": rules, "negation": negation, "arithmetic": arithmetic,
            "logic_grid": logic_grid, "graph": graph, "scheduling": scheduling}


def _row(family: str, lang: str, level: int, index: int, item, rng: random.Random, provenance: dict | None = None) -> dict:
    state, question, correct, distractors, derivation = item
    options = list(distractors)
    rng.shuffle(options)
    label = index % (len(options) + 1)  # the answer position cycles through every slot
    options.insert(label, correct)
    row_id = f"owned/{family}/{lang}/d{level}/{index:03d}"
    return {
        "id": row_id, "group_id": row_id, "family": family, "lang": lang, "difficulty": level,
        "state": state, "question": question, "options": options, "label": label,
        "target_distribution": None, "split": "test",
        "provenance": provenance or {
            "source": "project-authored-programmatic", "generator": GENERATOR, "seed": SEED,
            "derivation": {**derivation, "answer": correct["id"]}, "rights": RIGHTS,
        },
    }


def owned_rows(families=tuple(FAMILIES), per_cell: int | None = None) -> list[dict]:
    rows = []
    for family in families:
        for lang in LANGS:
            for level in LEVELS:
                for index in range(PER_CELL[lang] if per_cell is None else per_cell):
                    rng = random.Random(f"{GENERATOR}/{SEED}/{family}/{lang}/d{level}/{index}")
                    rows.append(_row(family, lang, level, index, FAMILIES[family](rng, lang, level, index), rng))
    return rows


# Chess: Lichess puzzles. The FEN is the position before the opponent's move; Moves[0] is that
# move and Moves[1] is the first move of the solution.

CHESS_BUCKETS = (1200, 1600, 2000)  # d1 < 1200 <= d2 < 1600 <= d3 < 2000 <= d4
PIECES = {"en": ["King", "Queen", "Rook", "Bishop", "Knight", "Pawn"], "zh": ["王", "后", "车", "象", "马", "兵"]}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _piece_list(board, color, lang: str) -> str:
    import chess

    parts = []
    for piece, name in zip((chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN), PIECES[lang]):
        squares = sorted(board.pieces(piece, color), key=lambda s: (chess.square_file(s), chess.square_rank(s)))
        if squares:
            name += "s" if lang == "en" and len(squares) > 1 else ""
            parts.append(f"{name} " + _t(lang, ", ", "、").join(chess.square_name(s) for s in squares))
    return _t(lang, "; ", "；").join(parts)


def chess_item(puzzle: dict, lang: str):
    """Question for one puzzle row, or None if it cannot give three safe distractors."""
    import chess

    board = chess.Board(puzzle["FEN"])
    moves = puzzle["Moves"].split()
    first, solution = chess.Move.from_uci(moves[0]), chess.Move.from_uci(moves[1])
    if not board.is_legal(first):
        return None
    opponent_san = board.san(first)
    board.push(first)
    if not board.is_legal(solution):
        return None
    candidates = []
    for move in board.legal_moves:
        if move == solution:
            continue
        board.push(move)
        mate = board.is_checkmate()  # another mate could also be correct; never use it as a distractor
        board.pop()
        if not mate:
            plausible = board.is_capture(move) or board.gives_check(move)
            rank = hashlib.sha256(f"{SEED}/{puzzle['PuzzleId']}/{move.uci()}".encode()).hexdigest()
            candidates.append((not plausible, rank, move))
    if len(candidates) < 3:
        return None
    wrong = [move for *_, move in sorted(candidates)[:3]]
    white = board.turn == chess.WHITE
    side, other = (("White", "Black") if white else ("Black", "White")) if lang == "en" else (
        ("白方", "黑方") if white else ("黑方", "白方"))
    mover, waiter = (chess.WHITE, chess.BLACK) if white else (chess.BLACK, chess.WHITE)

    def describe(move):
        a, b = chess.square_name(move.from_square), chess.square_name(move.to_square)
        return _t(lang, f"{board.san(move)} ({a} to {b})", f"{board.san(move)}（{a}→{b}）")

    fen = board.fen()
    if lang == "en":
        state = (f"Chess position (FEN): {fen}\n{side} pieces: {_piece_list(board, mover, lang)}\n"
                 f"{other} pieces: {_piece_list(board, waiter, lang)}\n{other} has just played {opponent_san}. "
                 f"{side} to move.")
        question = f"Which move is best for {side}?"
    else:
        state = (f"棋局（FEN）：{fen}\n{side}棋子：{_piece_list(board, mover, lang)}\n"
                 f"{other}棋子：{_piece_list(board, waiter, lang)}\n{other}刚走了{opponent_san}，现在轮到{side}走棋。")
        question = f"{side}的最佳着法是哪一步？"
    options = [_opt(m.uci(), describe(m)) for m in [solution] + wrong]
    derivation = {"puzzle_id": puzzle["PuzzleId"], "rating": int(puzzle["Rating"]), "fen_before": puzzle["FEN"],
                  "opponent_move": moves[0], "solution_move": moves[1], "solution_san": board.san(solution)}
    return state, question, options[0], options[1:], derivation


def chess_rows(csv_path: Path, per_cell: dict[str, int]) -> list[dict]:
    """Deterministic sample per rating bucket; English takes the first puzzles, Chinese the next."""
    upstream = sha256_file(csv_path)
    need = sum(per_cell.values()) * 2 + 50  # spare puzzles for rows that are skipped
    heaps: dict[int, list] = {level: [] for level in LEVELS}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for puzzle in csv.DictReader(stream):
            if int(puzzle["Popularity"]) < 80 or int(puzzle["NbPlays"]) < 1000:
                continue
            level = 1 + sum(int(puzzle["Rating"]) >= edge for edge in CHESS_BUCKETS)
            key = hashlib.sha256(f"{SEED}/{puzzle['PuzzleId']}".encode()).hexdigest()
            entry = (tuple(-b for b in bytes.fromhex(key)), key, puzzle)  # max-heap on the hash key
            if len(heaps[level]) < need:
                heapq.heappush(heaps[level], entry)
            elif entry > heaps[level][0]:
                heapq.heapreplace(heaps[level], entry)
    rows = []
    for level in LEVELS:
        queue = iter(sorted((key, puzzle) for _, key, puzzle in heaps[level]))
        for lang in LANGS:
            index = 0
            while index < per_cell[lang]:
                entry = next(queue, None)
                if entry is None:
                    break
                item = chess_item(entry[1], lang)
                if item is None:
                    continue
                rng = random.Random(f"{GENERATOR}/{SEED}/chess/{lang}/d{level}/{index}")
                derivation = {**item[4], "answer": item[2]["id"]}
                provenance = {"source": "lichess-puzzle-database", "generator": GENERATOR, "seed": SEED,
                              "upstream": "https://database.lichess.org/lichess_db_puzzle.csv.zst",
                              "upstream_csv_sha256": upstream, "derivation": derivation, "rights": CHESS_RIGHTS}
                rows.append(_row("chess", lang, level, index, item, rng, provenance))
                index += 1
    return rows


def dump(rows: list[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-cell", type=int, help="rows per (family, lang, difficulty); default 75 en / 50 zh, "
                                                     "chess 200 en / 100 zh")
    parser.add_argument("--families", nargs="+", choices=[*FAMILIES, "chess"], default=[*FAMILIES, "chess"])
    parser.add_argument("--chess-csv", type=Path, help="decompressed lichess_db_puzzle.csv; chess is skipped without it")
    args = parser.parse_args()
    rows = owned_rows([f for f in args.families if f != "chess"], args.per_cell)
    if "chess" in args.families and args.chess_csv:
        per_cell = CHESS_PER_CELL if args.per_cell is None else {lang: args.per_cell for lang in LANGS}
        rows += chess_rows(args.chess_csv, per_cell)
    elif "chess" in args.families:
        print("chess skipped: no --chess-csv given")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(dump(rows))
    counts: dict[str, int] = {}
    for row in rows:
        counts[f"{row['family']}/{row['lang']}"] = counts.get(f"{row['family']}/{row['lang']}", 0) + 1
    print(f"wrote {len(rows)} rows " + json.dumps(counts))


if __name__ == "__main__":
    main()
