import argparse
import datetime
import getpass
import json
import pathlib
import re
import time
import urllib.error
import urllib.request

import yaml

BASE_URL = "http://localhost:8000"
PAUSE = 6   # seconds between questions: the Cohere trial key allows 10 re-rank calls per minute
HERE = pathlib.Path(__file__).parent


def post(path, payload, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(BASE_URL + path, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def login(user):
    password = getpass.getpass(f"Password for {user}: ")   # typed, not stored anywhere
    return post("/api/login", {"username": user, "password": password})["token"]


def was_refused(resp):
    a = resp["answer"].lower()
    return (not resp["results"]) or "not available" in a or "couldn't find" in a


def main():
    parser = argparse.ArgumentParser(description="Run the golden-question report card.")
    parser.add_argument("--runs", type=int, default=1, help="how many times to ask each question (default 1)")
    args = parser.parse_args()

    questions = yaml.safe_load((HERE / "questions.yaml").read_text(encoding="utf-8"))
    tokens = {u: login(u) for u in sorted({q["user"] for q in questions})}

    t = {"answer_total": 0, "correct": 0, "has_cites": 0, "verified": 0, "unverified": 0,
         "refuse_total": 0, "refused_ok": 0, "security_total": 0, "security_ok": 0, "errors": 0}
    by_category = {}   # category -> [passed, total]
    log = []

    total_asks = len(questions) * args.runs
    print(f"\nAsking {len(questions)} questions x {args.runs} run(s) = {total_asks} answers "
          f"(about {total_asks * PAUSE // 60 + 1}+ minutes)\n")

    for q in questions:
        cat = q.get("category", "other")
        by_category.setdefault(cat, [0, 0])

        for run in range(1, args.runs + 1):
            time.sleep(PAUSE)
            try:
                resp = post("/api/query", {"query": q["question"]}, tokens[q["user"]])
            except urllib.error.HTTPError as e:
                t["errors"] += 1
                by_category[cat][1] += 1
                print(f"{q['id']:34} run {run}  ✗ ERROR {e.code} (server said: {e.reason})")
                log.append({"id": q["id"], "run": run, "error": e.code})
                continue

            answer = resp["answer"]
            sources = sorted({r["source_file"] for r in resp["results"]})

            if q.get("forbidden_sources"):
                leaked = [s for s in sources if s in q["forbidden_sources"]]
                passed = not leaked
                t["security_total"] += 1
                t["security_ok"] += passed
                line = "✓ no forbidden sources" if passed else f"✗ LEAK: {leaked}"
            elif q["expect"] == "refuse":
                passed = was_refused(resp)
                t["refuse_total"] += 1
                t["refused_ok"] += passed
                line = "✓ refused" if passed else "✗ ANSWERED (should refuse)"
            else:
                passed = any(k.lower() in answer.lower() for k in q["must_contain_any"])
                cites = re.findall(r"\[(\d+)(\?)?\]", answer)
                verified = sum(1 for _, flag in cites if not flag)
                unverified = sum(1 for _, flag in cites if flag)
                t["answer_total"] += 1
                t["correct"] += passed
                t["has_cites"] += bool(cites)
                t["verified"] += verified
                t["unverified"] += unverified
                line = (f"{'✓ correct' if passed else '✗ WRONG/MISSING'}   "
                        f"citations: {len(cites)} (✓ {verified}, ⚠ {unverified})")

            by_category[cat][0] += passed
            by_category[cat][1] += 1
            print(f"{q['id']:34} run {run}  {line}")
            log.append({"id": q["id"], "run": run, "passed": passed, "sources": sources, "answer": answer})

    cited_total = t["verified"] + t["unverified"]
    print("\n================ REPORT CARD ================")
    print(f"Answer correct:        {t['correct']} of {t['answer_total']}")
    print(f"Has citations:         {t['has_cites']} of {t['answer_total']}")
    if cited_total:
        print(f"Citations verified:    {t['verified']} of {cited_total} ({t['verified'] / cited_total:.0%})")
    print(f"Refused correctly:     {t['refused_ok']} of {t['refuse_total']}")
    print(f"Permission checks:     {t['security_ok']} of {t['security_total']}")
    if t["errors"]:
        print(f"Errors (not scored):   {t['errors']}")
    print("\nBy category:")
    for cat, (passed, total) in by_category.items():
        print(f"  {cat:12} {passed:3} of {total}")
    print("=============================================")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    (out / f"{stamp}.json").write_text(
        json.dumps({"runs": args.runs, "totals": t, "by_category": by_category, "answers": log}, indent=2),
        encoding="utf-8")
    print(f"Saved full answers to eval/results/{stamp}.json")


if __name__ == "__main__":
    main()
