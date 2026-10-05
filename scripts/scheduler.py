"""Tiny GPU job scheduler for the 40 h run.

Queue: jobs/queue.tsv, one job per line: <id>\t<slots>\t<timeout_h>\t<command>
  slots: 1 = one TP=2 slot (2 GCDs), excl = one slot but only when no other job runs,
         node = all 8 GCDs.
The command runs on the host with GPUS=<gcd list> and NAME=ld_<id> in its environment.
The queue is re-read every poll, so jobs can be appended while the scheduler runs.
Finished/failed jobs are recorded in jobs/state/<id>.{done,failed}; re-running the
scheduler skips them. Status snapshot for monitoring: jobs/status.json.
"""
import json
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUEUE, STATE, LOGS = f"{ROOT}/jobs/queue.tsv", f"{ROOT}/jobs/state", "/var/tmp/arkamukh/logs/jobs"
SLOTS = ["0,1", "2,3", "4,5", "6,7"]
POLL_S = 20


def read_queue():
    jobs = []
    for line in open(QUEUE):
        if line.strip() and not line.startswith("#"):
            jid, slots, timeout_h, cmd = line.rstrip("\n").split("\t", 3)
            jobs.append({"id": jid, "slots": slots, "timeout_h": float(timeout_h), "cmd": cmd})
    return jobs


def finished(jid):
    return any(os.path.exists(f"{STATE}/{jid}.{s}") for s in ("done", "failed"))


def main():
    os.makedirs(STATE, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    running = {}  # jid -> dict(proc, slots, start, timeout_h)
    while True:
        # reap
        for jid, r in list(running.items()):
            rc = r["proc"].poll()
            elapsed = time.time() - r["start"]
            if rc is None and elapsed > r["timeout_h"] * 3600:
                subprocess.run(["docker", "rm", "-f", f"ld_{jid}"], capture_output=True)
                r["proc"].kill()
                rc = "timeout"
            if rc is not None:
                subprocess.run(["docker", "rm", "-f", f"ld_{jid}"], capture_output=True)
                ok = rc == 0
                with open(f"{STATE}/{jid}.{'done' if ok else 'failed'}", "w") as f:
                    json.dump({"rc": rc, "gpus": r["gpus"], "start": r["start"], "seconds": elapsed}, f)
                del running[jid]

        # schedule in queue order; a blocked node/excl job holds back everything behind it
        busy = {s for r in running.values() for s in r["slot_ids"]}
        free = [s for s in SLOTS if s not in busy]
        pending = [j for j in read_queue() if j["id"] not in running and not finished(j["id"])]
        if any(r["exclusive"] for r in running.values()):
            pending = []  # an excl/node job owns the machine until it finishes
        for j in pending:
            if j["slots"] in ("node", "excl"):
                if running:
                    break
                slot_ids = SLOTS if j["slots"] == "node" else [SLOTS[0]]
            elif free:
                slot_ids = [free.pop(0)]
            else:
                break
            gpus = ",".join(slot_ids)
            log = open(f"{LOGS}/{j['id']}.log", "w")
            env = dict(os.environ, GPUS=gpus, NAME=f"ld_{j['id']}")
            proc = subprocess.Popen(j["cmd"], shell=True, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            running[j["id"]] = {"proc": proc, "gpus": gpus, "slot_ids": slot_ids,
                                "start": time.time(), "timeout_h": j["timeout_h"],
                                "exclusive": j["slots"] in ("node", "excl")}
            if j["slots"] in ("node", "excl"):
                break

        all_ids = [j["id"] for j in read_queue()]
        status = {
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "running": {k: {"gpus": v["gpus"], "minutes": round((time.time() - v["start"]) / 60, 1)}
                        for k, v in running.items()},
            "done": [i for i in all_ids if os.path.exists(f"{STATE}/{i}.done")],
            "failed": [i for i in all_ids if os.path.exists(f"{STATE}/{i}.failed")],
            "pending": [i for i in all_ids if i not in running and not finished(i)],
        }
        json.dump(status, open(f"{ROOT}/jobs/status.json.tmp", "w"), indent=1)
        os.replace(f"{ROOT}/jobs/status.json.tmp", f"{ROOT}/jobs/status.json")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
