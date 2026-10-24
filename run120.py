"""Transactional round completion and reusable fold journals across allocations."""

from search import *


class Pause(Exception):
    pass


class Experiment120(Experiment):
    def __init__(self, run):
        super().__init__(run)
        self.finishing = False
        path = self.run / "resume_accounting.json"
        if path.exists():
            accounting = read(path)
            self.initial_cpu = max(self.initial_cpu, accounting["design_cpu_seconds"])
            self.initial_wall = max(self.initial_wall, accounting["elapsed_seconds"])
            self.initial_af3_wall = max(
                self.initial_af3_wall, accounting["af3_wall_seconds"]
            )
        self.journal = (
            read(self.run / "pending_folds.json.gz")
            if (self.run / "pending_folds.json.gz").exists()
            else {}
        )
        (self.run / "paused.json").unlink(missing_ok=True)

    def expired(self):
        if self.finishing:
            return False
        if (
            self.stop
            or time.monotonic() - self.start_wall
            >= self.c.get("allocation_search_hours", 42) * 3600
        ):
            raise Pause(
                "Pause at allocation boundary; unfinished round is not committed"
            )
        return False

    def fold(self, items, round_number):
        # A resumed round regenerates its shortlist, then reuses only journaled
        # folds whose sequence, identity, and parent structure match exactly.
        name = str(round_number)
        signature = [sha(r["sequence"] + "|" + r["id"] + "|" + r["pdb"]) for r in items]
        entry = self.journal.setdefault(name, dict(signature=signature, records=[]))
        if entry["signature"] != signature:
            raise ValueError(
                "Resumed fold shortlist changed; retained checkpoint requires inspection"
            )
        records = entry["records"]
        for first in range(len(records), len(items), 4):
            self.expired()
            block = items[first : first + 4]
            result = self.af3.fold(block, self.c["seed"] + round_number * 10000 + first)
            if len(result) != len(block):
                raise ValueError("Incomplete AF3 batch")
            completed = time.time()
            for r in result:
                r["fold_completed_unix"] = completed
            records.extend(result)
            save(self.run / "pending_folds.json.gz", self.journal)
            save(
                self.run / "live.json",
                dict(
                    phase="fold",
                    round=round_number,
                    completed=len(records),
                    planned=len(items),
                    design_cpu_seconds=self.design_cpu(),
                    elapsed_seconds=self.elapsed(),
                ),
            )
        return list(records)

    def accounting(self):
        return dict(
            design_cpu_seconds=self.design_cpu(),
            elapsed_seconds=self.elapsed(),
            af3_wall_seconds=self.initial_af3_wall + self.af3.wall_seconds,
        )

    def checkpoint(self, state):
        if state.get("archive"):
            best = min(state["archive"], key=lambda r: key(r, self.c["direction"]))
            state["history"][-1]["cutoff_counts"] = best["profile"].get(
                "cutoff_counts", {}
            )
            state["history"][-1]["completed_unix"] = time.time()
        super().checkpoint(state)
        save(self.run / "resume_accounting.json", self.accounting())
        self.journal = {}
        (self.run / "pending_folds.json.gz").unlink(missing_ok=True)

    def pause(self):
        self.cache.snapshot(self.run / "cache.sqlite")
        save(self.run / "resume_accounting.json", self.accounting())
        save(
            self.run / "paused.json",
            dict(
                complete=False,
                completed_rounds=self.state["round"] if self.state else -1,
                next_action="resume_own_checkpoint_and_unfinished_round",
                **self.accounting()
            ),
        )

    def complete(self, reason):
        self.finishing = True
        try:
            self.finish()
            done = read(self.run / "complete.json")
            done["stop_reason"] = reason
            save(self.run / "complete.json", done)
            return done
        finally:
            self.finishing = False

    def main(self):
        try:
            self.expired()
            if not self.state:
                self.initial()
            while True:
                self.expired()
                n = self.state["round"]
                if n >= self.c["rounds"]:
                    self.complete("all_requested_rounds")
                    return
                best = min(
                    self.state["archive"], key=lambda r: key(r, self.c["direction"])
                )
                target = (
                    0
                    if self.c["direction"] == "deimmunize"
                    else len(m.windows(self.c["root_sequence"]))
                    * len(self.c["alleles"])
                    * (8 if scoring.MODE == "weighted8" else 1)
                )
                checked = self.run / "endpoint_attempt.json"
                attempted = (
                    read(checked).get("sequence_sha256") if checked.exists() else None
                )
                if objective(best["profile"]) == target and attempted != sha(
                    best["sequence"]
                ):
                    done = self.complete("verified_objective_endpoint")
                    if done["confirmation_passes"] == 2:
                        return
                    save(
                        checked,
                        dict(
                            sequence_sha256=sha(best["sequence"]),
                            completion=done,
                            confirmation=read(self.run / "confirmation.json.gz"),
                        ),
                    )
                    (self.run / "complete.json").unlink()
                self.one_round(n + 1)
        except Pause:
            self.pause()
        except Exception as exc:
            save(self.run / "failure.json", dict(error=repr(exc), **self.accounting()))
            raise
        finally:
            self.cache.db.close()
            self.backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    Experiment120(parser.parse_args().run).main()
