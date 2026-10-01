import datetime
import getpass
import json
import pathlib
import re
import urllib.request

import yaml

BASE_URL = "http://localhost:8000"
RUNS = 3   # ask every question this many times (the AI's answers vary)
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
    questions = yaml.safe_load((HERE / "questions.yaml").read_text(encoding="utf-8"))
    tokens = {u: login(u) for u in sorted({q["user"] for q in questions})}

    t = {"answer_total": 0, "correct": 0, "has_cites": 0, "verified": 0, "unverified": 0,
         "refuse_total": 0, "refused_ok": 0}
    log = []

    for q in questions:
        for run in range(1, RUNS + 1):
            resp = post("/api/query", {"query": q["question"]}, tokens[q["user"]])
            answer = resp["answer"]

            if q["expect"] == "refuse":
                ok = was_refused(resp)
                t["refuse_total"] += 1
                t["refused_ok"] += ok
                line = f"{'✓' if ok else '✗'} refused" if ok else "✗ ANSWERED (should refuse)"
            else:
                correct = any(k.lower() in answer.lower() for k in q["must_contain_any"])
                cites = re.findall(r"\[(\d+)(\?)?\]", answer)
                verified = sum(1 for _, flag in cites if not flag)
                unverified = sum(1 for _, flag in cites if flag)
                t["answer_total"] += 1
                t["correct"] += correct
                t["has_cites"] += bool(cites)
                t["verified"] += verified
                t["unverified"] += unverified
                line = (f"{'✓' if correct else '✗'} {'correct' if correct else 'WRONG/MISSING'}   "
                        f"citations: {len(cites)} (✓ {verified}, ⚠ {unverified})")

            print(f"{q['id']:26} run {run}  {line}")
            log.append({"id": q["id"], "run": run, "answer": answer})

    cited_total = t["verified"] + t["unverified"]
    print("\n================ REPORT CARD ================")
    print(f"Answer correct:        {t['correct']} of {t['answer_total']}")
    print(f"Has citations:         {t['has_cites']} of {t['answer_total']}")
    if cited_total:
        print(f"Citations verified:    {t['verified']} of {cited_total} "
              f"({t['verified'] / cited_total:.0%})")
    print(f"Refused correctly:     {t['refused_ok']} of {t['refuse_total']}")
    print("=============================================")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    (out / f"{stamp}.json").write_text(json.dumps({"totals": t, "answers": log}, indent=2), encoding="utf-8")
    print(f"Saved full answers to eval/results/{stamp}.json")


if __name__ == "__main__":
    main()