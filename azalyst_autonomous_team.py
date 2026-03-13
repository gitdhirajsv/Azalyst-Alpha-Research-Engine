"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST AUTONOMOUS RESEARCH TEAM
║        Two AI personas — Researcher + Developer — working autonomously      ║
║        They read results, discuss, fix code, test, repeat until alpha       ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW IT WORKS
────────────
  RESEARCHER (Olivia):
    - Reads performance_metrics.csv
    - Diagnoses what went wrong
    - Proposes hypotheses
    - Keeps prompts small and fast

  DEVELOPER (Marcus):
    - Reads specific code sections (not whole files)
    - Implements targeted fixes
    - Reports back what changed

  Loop:
    1. Researcher reads results → diagnoses
    2. Researcher proposes fix
    3. Developer reads relevant code section
    4. Developer applies surgical fix
    5. Pipeline runs (walkforward_simulator.py)
    6. Researcher reads new results
    7. Keep or revert based on improvement
    8. Repeat until alpha confirmed

USAGE
─────
  Terminal 1: ollama serve
  Terminal 2: python azalyst_autonomous_team.py

NOTE: Signal direction has already been flipped (main alpha fix).
      This team refines from that improved baseline.
"""

import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ← FIX: Set global stdout to UTF-8 (prevents UnicodeDecodeError in parent process)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import requests

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_PATH = r"D:\Personal Files\Azalyst Alpha Research Engine"
MODEL        = "deepseek-r1:14b"
OLLAMA_URL   = "http://localhost:11434/api/generate"
LOG_FILE     = os.path.join(PROJECT_PATH, "team_log.txt")
RESULTS_FILE = os.path.join(PROJECT_PATH, "performance_metrics.csv")

TARGETS = {
    "win_rate":      55.0,
    "sharpe_ratio":   1.0,
    "max_drawdown": -25.0,
    "profit_factor":  1.5,
}

BASELINE = {
    "win_rate":      27.5,
    "sharpe_ratio":  -5.27,
    "max_drawdown":  -50.04,
    "profit_factor":  0.26,
}

MAX_ITERATIONS = 10   # max fix→test cycles before stopping
LLM_TIMEOUT    = 60   # seconds per LLM call before falling back to deterministic tuning

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGER
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str, persona: str = "") -> None:
    ts    = datetime.now().strftime("%H:%M:%S")
    label = f"[{persona}] " if persona else ""
    line  = f"[{ts}] {label}{msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
#  LLM INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def ask_llm(prompt: str, persona: str = "", timeout: int = LLM_TIMEOUT) -> str:
    """
    Send a SHORT prompt to Ollama. Keep under 1500 chars for fast responses.
    The key to making this work: never send whole files, only small sections.
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.15,
            "num_predict": 250,
            "num_gpu": 99,
        }
    }
    log(f"Sending {len(prompt)} char prompt...", persona)
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()
        if "<think>" in answer and "</think>" in answer:
            answer = answer.split("</think>")[-1].strip()
        log(f"Response: {len(answer)} chars", persona)
        return answer
    except requests.exceptions.Timeout:
        log(f"Timeout after {timeout}s — prompt might be too large", persona)
        return ""
    except Exception as e:
        log(f"Error: {e}", persona)
        return ""

# ─────────────────────────────────────────────────────────────────────────────
#  RESEARCHER PERSONA — OLIVIA
# ─────────────────────────────────────────────────────────────────────────────

class Researcher:
    name = "OLIVIA (Researcher)"

    def diagnose(self, metrics: dict) -> dict:
        """
        Read current metrics and propose one specific fix.
        Returns: {"hypothesis": str, "fix_type": "constant"|"logic", "target": ..., "value": ...}
        """
        prompt = f"""You are Olivia, a senior quantitative researcher.
Your colleague Marcus will implement whatever fix you suggest.

CURRENT BACKTESTING METRICS:
  Win Rate:      {metrics.get('win_rate',0):.1f}%  (target >{TARGETS['win_rate']}%)
  Sharpe:        {metrics.get('sharpe_ratio',0):.3f}  (target >{TARGETS['sharpe_ratio']})
  Max Drawdown:  {metrics.get('max_drawdown',0):.1f}%  (target >{TARGETS['max_drawdown']}%)
  Profit Factor: {metrics.get('profit_factor',0):.3f}  (target >{TARGETS['profit_factor']})
  Total Trades:  {metrics.get('total_trades',0)}

SYSTEM CONTEXT:
- Crypto ML trading (LightGBM + walk-forward backtest)
- Signals: BUY when model prob < 0.40, SELL when prob > 0.60
  (counter-predictive flip already applied — do NOT change signal direction)
- Stop-loss: -1.5%, Take-profit: +2.7%
- Regime: Gaussian Mixture on BTC (BULL_TREND/BEAR_TREND/HIGH_VOL_LATERAL/LOW_VOL_GRIND)

Pick ONE fix from these options and respond in this EXACT format:
FIX_TYPE: constant
FILE: walkforward_simulator.py
CONSTANT: <name>
VALUE: <new_value>
REASON: <one sentence>

Available constants to tune:
- BUY_THRESHOLD (current: 0.60) — min confidence for SELL signal (counter-predictive)
- SELL_THRESHOLD (current: 0.40) — min confidence for BUY signal (counter-predictive)  
- stop_loss_pct (-1.5) — stop loss percentage
- take_profit_pct (2.7) — take profit percentage

Only suggest a change that directly addresses the main metric failing above."""

        response = ask_llm(prompt, self.name)

        # Parse the structured response
        result = {
            "hypothesis": response,
            "fix_type": None,
            "file": None,
            "constant": None,
            "value": None,
            "reason": "",
        }

        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("FIX_TYPE:"):
                result["fix_type"] = line.split(":", 1)[1].strip()
            elif line.startswith("FILE:"):
                result["file"] = line.split(":", 1)[1].strip()
            elif line.startswith("CONSTANT:"):
                result["constant"] = line.split(":", 1)[1].strip()
            elif line.startswith("VALUE:"):
                result["value"] = line.split(":", 1)[1].strip()
            elif line.startswith("REASON:"):
                result["reason"] = line.split(":", 1)[1].strip()

        return result

    def review(self, before: dict, after: dict) -> str:
        """Compare before/after metrics and give a brief verdict."""
        wr_delta = after.get("win_rate",0) - before.get("win_rate",0)
        sr_delta = after.get("sharpe_ratio",0) - before.get("sharpe_ratio",0)
        sign_wr  = "+" if wr_delta >= 0 else ""
        sign_sr  = "+" if sr_delta >= 0 else ""

        prompt = f"""You are Olivia, quant researcher.
After a code fix:
  Win Rate:  {before.get('win_rate',0):.1f}% → {after.get('win_rate',0):.1f}% ({sign_wr}{wr_delta:.1f}pp)
  Sharpe:    {before.get('sharpe_ratio',0):.3f} → {after.get('sharpe_ratio',0):.3f} ({sign_sr}{sr_delta:.3f})
  Drawdown:  {before.get('max_drawdown',0):.1f}% → {after.get('max_drawdown',0):.1f}%
  PF:        {before.get('profit_factor',0):.3f} → {after.get('profit_factor',0):.3f}

In one sentence: was this an improvement? Should Marcus keep or revert this change?
Answer: KEEP or REVERT, then one sentence reason."""

        return ask_llm(prompt, self.name, timeout=120)


# ─────────────────────────────────────────────────────────────────────────────
#  DEVELOPER PERSONA — MARCUS
# ─────────────────────────────────────────────────────────────────────────────

class Developer:
    name = "MARCUS (Developer)"

    def apply_constant(self, filepath: str, constant: str, value: str) -> bool:
        """Direct constant replacement — fast and reliable, no LLM needed."""
        log(f"Applying: {constant} = {value} in {os.path.basename(filepath)}", self.name)
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                code = fh.read()

            pattern = rf"^({re.escape(constant)}\s*=\s*)(.+?)(\s*#.*)?$"
            # Preserve inline comment if present
            match = re.search(pattern, code, re.MULTILINE)
            if not match:
                log(f"  '{constant}' not found — trying case-insensitive search", self.name)
                pattern2 = rf"(?i)^({re.escape(constant)}\s*=\s*)(.+?)(\s*#.*)?$"
                match = re.search(pattern2, code, re.MULTILINE)
                if not match:
                    log(f"  '{constant}' not found in file", self.name)
                    return False

            comment = match.group(3) or ""
            new_line = f"{match.group(1)}{value}{comment}"
            new_code = code[:match.start()] + new_line + code[match.end():]

            with open(filepath, "w", encoding="utf-8") as fh:
                fh.write(new_code)
            log(f"  Done: {constant} = {value}", self.name)
            return True
        except Exception as e:
            log(f"  Error: {e}", self.name)
            return False

    def run_simulator(self, project_path: str, coins=None, no_resume=False):
        log("Running walkforward_simulator.py...", self.name)
        cmd = [sys.executable, "walkforward_simulator.py",
               "--data-dir", "./data",
               "--feature-dir", "./feature_cache"]
        if no_resume:
            cmd.append("--no-resume")
        env = os.environ.copy()
        if coins:
            env["AZALYST_TEST_COINS"] = ",".join(coins)
            log(f"Using sample universe: {len(coins)} cached symbols", self.name)
        env.setdefault("AZALYST_LGBM_DEVICE", "cpu")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=project_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env
            )
            start_ts = time.time()
            while True:
                try:
                    stdout, stderr = proc.communicate(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = int(time.time() - start_ts)
                    if elapsed >= 7200:
                        proc.kill()
                        stdout, stderr = proc.communicate()
                        raise subprocess.TimeoutExpired(cmd, 7200, output=stdout, stderr=stderr)
                    log(
                        f"Simulator still running... {elapsed // 60}m {elapsed % 60:02d}s elapsed",
                        self.name,
                    )
            if proc.returncode == 0:
                log("Simulator finished successfully", self.name)
                if stdout:
                    log(f"  STDOUT tail:\n{stdout[-1200:]}", self.name)
            else:
                log(f"Simulator error (code {proc.returncode})", self.name)
                if stdout:
                    log(f"  STDOUT tail:\n{stdout[-1200:]}", self.name)
                if stderr:
                    log(f"  STDERR tail:\n{stderr[-1200:]}", self.name)
            return proc.returncode == 0
        except subprocess.TimeoutExpired as e:
            log("Simulator timed out after 120 min", self.name)
            if e.output:
                log(f"  STDOUT tail:\n{e.output[-1200:]}", self.name)
            if e.stderr:
                log(f"  STDERR tail:\n{e.stderr[-1200:]}", self.name)
            return False
        except Exception as e:
            log(f"Subprocess error: {e}", self.name)
            return False


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def read_metrics(results_file: str) -> dict:
    try:
        with open(results_file, "r") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return {}
        last = rows[-1]
        return {
            "win_rate":      float(last.get("win_rate", 0)),
            "sharpe_ratio":  float(last.get("sharpe_ratio", -99)),
            "max_drawdown":  float(last.get("max_drawdown_pct", -99)),
            "profit_factor": float(last.get("profit_factor", 0)),
            "total_trades":  int(float(last.get("total_trades", 0))),
            "cycle":         last.get("cycle", "?"),
        }
    except Exception:
        return {}

def all_targets_met(m: dict) -> bool:
    return bool(m) and all([
        m.get("win_rate",      0)   >= TARGETS["win_rate"],
        m.get("sharpe_ratio",  -99) >= TARGETS["sharpe_ratio"],
        m.get("max_drawdown",  -99) >= TARGETS["max_drawdown"],
        m.get("profit_factor", 0)   >= TARGETS["profit_factor"],
    ])

def print_table(before: dict, after: dict, label: str = "") -> None:
    log("")
    log(f"  ─── METRICS {label} ───")
    log(f"  {'Metric':<22} {'Before':>8} {'After':>8} {'Δ':>8}  Target")
    log(f"  {'-'*58}")
    for key, name, tgt in [
        ("win_rate",      "Win Rate %",     f">{TARGETS['win_rate']}%"),
        ("sharpe_ratio",  "Sharpe",         f">{TARGETS['sharpe_ratio']}"),
        ("max_drawdown",  "Max Drawdown %", f">{TARGETS['max_drawdown']}%"),
        ("profit_factor", "Profit Factor",  f">{TARGETS['profit_factor']}"),
    ]:
        b  = before.get(key, 0)
        a  = after.get(key, 0)
        ch = a - b
        sign = "+" if ch >= 0 else ""
        log(f"  {name:<22} {b:>8.2f} {a:>8.2f} {sign}{ch:>7.2f}  {tgt}")
    log("")

def get_test_coins(n: int = 25) -> list:
    feature_dir = os.path.join(PROJECT_PATH, "feature_cache")
    try:
        files = sorted([
            f.replace(".parquet", "")
            for f in os.listdir(feature_dir)
            if f.endswith(".parquet") and f.endswith("USDT.parquet")
        ])
        return files[:n]
    except Exception:
        return []

def backup(filepath: str) -> None:
    if os.path.exists(filepath):
        shutil.copy2(filepath, filepath + ".bak")

def restore(filepath: str) -> bool:
    bak = filepath + ".bak"
    if os.path.exists(bak):
        shutil.copy2(bak, filepath)
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  TEAM DISCUSSION — printed as dialogue
# ─────────────────────────────────────────────────────────────────────────────

def team_discussion(researcher: Researcher, metrics: dict) -> dict:
    """Researcher and Developer 'discuss' the results before acting."""
    log("")
    log("══════════════════ TEAM DISCUSSION ══════════════════")
    log(f"[OLIVIA] Current win rate {metrics.get('win_rate',0):.1f}%, "
        f"Sharpe {metrics.get('sharpe_ratio',0):.3f}")
    log("[OLIVIA] Analysing what to try next...")

    plan = researcher.diagnose(metrics)

    if plan.get("constant"):
        log(f"[OLIVIA] Hypothesis: {plan.get('reason', 'needs tuning')}")
        log(f"[OLIVIA] Suggestion: change {plan['constant']} to {plan['value']}")
        log(f"[MARCUS] Got it. I'll update {plan['constant']} = {plan['value']} and run the test.")
    else:
        log("[OLIVIA] LLM gave no clear fix — falling back to standard tuning schedule")
        log("[MARCUS] I'll try tightening the threshold by 0.01")

    log("═════════════════════════════════════════════════════")
    return plan


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TEAM LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.chdir(PROJECT_PATH)

    researcher = Researcher()
    developer  = Developer()

    log("═" * 65)
    log("  AZALYST AUTONOMOUS RESEARCH TEAM")
    log("  Researcher: Olivia  |  Developer: Marcus")
    log(f"  Model: {MODEL}  |  Target: Win Rate >{TARGETS['win_rate']}%")
    log("═" * 65)
    log("  NOTE: Signal direction flip already applied in walkforward_simulator.py")
    log("  Expected baseline win rate: 60-75%+ on first run")
    log("═" * 65)

    # Check Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200:
            raise Exception("Bad status")
        log("[OK] Ollama is running")
    except Exception:
        log("")
        log("  Ollama is NOT running. Please:")
        log("  1. Open a new CMD window")
        log("  2. Type: ollama serve")
        log("  3. Wait until it says 'Listening on...'")
        log("  4. Come back and press Enter")
        log("")
        input("Press Enter when Ollama is running...")

    # Starting state
    current = read_metrics(RESULTS_FILE) or BASELINE.copy()
    test_coins = get_test_coins(25)
    sim_path   = os.path.join(PROJECT_PATH, "walkforward_simulator.py")

    log(f"\n[START] Win Rate: {current.get('win_rate',0):.1f}%  "
        f"Sharpe: {current.get('sharpe_ratio',0):.3f}  "
        f"Trades: {current.get('total_trades',0):,}")

    # ── PHASE 1: First run with signal flip ──────────────────────────────────
    log("\n[PHASE 1] First run — testing signal direction fix...")
    ok = developer.run_simulator(PROJECT_PATH, test_coins or None, no_resume=True)
    if ok:
        after = read_metrics(RESULTS_FILE)
        if after:
            print_table(BASELINE, after, "PHASE 1 — Signal Flip")
            current = after

    # ── PHASE 2: Iterative LLM-guided refinement ─────────────────────────────
    if not all_targets_met(current):
        log(f"\n[PHASE 2] Targets not met — starting {MAX_ITERATIONS} refinement cycles")

        # Standard tuning schedule (fast, no LLM)
        standard_schedule = [
            ("BUY_THRESHOLD",   "0.62"),
            ("SELL_THRESHOLD",  "0.38"),
            ("BUY_THRESHOLD",   "0.65"),
            ("SELL_THRESHOLD",  "0.35"),
        ]

        for iteration in range(MAX_ITERATIONS):
            log(f"\n[ITERATION {iteration+1}/{MAX_ITERATIONS}]")

            if all_targets_met(current):
                log("[DONE] All targets met!")
                break

            # Try LLM diagnosis first (if prompt is small enough it's fast)
            plan = team_discussion(researcher, current)

            fix_applied = False
            if plan.get("constant") and plan.get("value"):
                # Use LLM-suggested fix
                const = plan["constant"]
                value = plan["value"]
                backup(sim_path)
                ok = developer.apply_constant(sim_path, const, value)
                if ok:
                    ran = developer.run_simulator(PROJECT_PATH, test_coins or None)
                    if ran:
                        after = read_metrics(RESULTS_FILE)
                        if after:
                            verdict = researcher.review(current, after)
                            log(f"[OLIVIA] {verdict}")
                            print_table(current, after, f"iter {iteration+1}")
                            if "KEEP" in verdict.upper() or (
                                after.get("win_rate",0) > current.get("win_rate",0) or
                                after.get("sharpe_ratio",-99) > current.get("sharpe_ratio",-99)
                            ):
                                log("[MARCUS] Keeping changes.")
                                current = after
                            else:
                                log("[MARCUS] Reverting.")
                                restore(sim_path)
                            fix_applied = True

            # Fallback: use standard schedule if LLM gave no fix or it failed
            if not fix_applied and standard_schedule:
                const, value = standard_schedule.pop(0)
                log(f"[MARCUS] (Fallback) Trying {const} = {value}")
                backup(sim_path)
                ok = developer.apply_constant(sim_path, const, value)
                if ok:
                    ran = developer.run_simulator(PROJECT_PATH, test_coins or None)
                    if ran:
                        after = read_metrics(RESULTS_FILE)
                        if after:
                            print_table(current, after, f"fallback {const}={value}")
                            if (after.get("win_rate",0) > current.get("win_rate",0) or
                                    after.get("sharpe_ratio",-99) > current.get("sharpe_ratio",-99)):
                                log("[MARCUS] Improvement — keeping.")
                                current = after
                            else:
                                log("[MARCUS] No improvement — reverting.")
                                restore(sim_path)
                elif not standard_schedule:
                    log("[MARCUS] Exhausted standard schedule. Stopping iterations.")
                    break

    # ── PHASE 3: Confirm with 3 consecutive passes ───────────────────────────
    if all_targets_met(current):
        log("\n[PHASE 3] Confirming alpha — 3 consecutive passes...")
        passes = 0
        for i in range(3):
            log(f"[CONFIRM {i+1}/3]")
            ok = developer.run_simulator(PROJECT_PATH, test_coins or None)
            m  = read_metrics(RESULTS_FILE)
            if m:
                print_table(BASELINE, m, f"confirm {i+1}/3")
                if all_targets_met(m):
                    passes += 1
                    log(f"[OLIVIA] Confirm {i+1}/3: PASS ({passes}/3 needed)")
                else:
                    passes = 0
                    log(f"[OLIVIA] Confirm {i+1}/3: FAIL — resetting count")
            time.sleep(2)

        # ── PHASE 4: Full universe ─────────────────────────────────────────────
        if passes >= 3:
            log("\n[PHASE 3] CONFIRMED — 3/3 passes on sample")
            log("[PHASE 4] Testing on ALL coins...")
            ok = developer.run_simulator(PROJECT_PATH, coins=None, no_resume=True)
            final = read_metrics(RESULTS_FILE)
            if final:
                print_table(BASELINE, final, "FINAL — ALL COINS")
                if all_targets_met(final):
                    log("")
                    log("═" * 65)
                    log("  *** ALPHA CONFIRMED ***")
                    log("  Olivia: Excellent work Marcus. Strategy validated.")
                    log("  Marcus: All systems green. Ready for deployment review.")
                    log("  Small sample : PASS")
                    log("  Full universe: PASS")
                    log("═" * 65)
                else:
                    log("[OLIVIA] Works on sample, needs more work for full universe.")
                    log("[MARCUS] Noted. Need more iterations on full coin set.")
        else:
            log(f"\n[OLIVIA] Only {passes}/3 passes. More refinement needed.")
            log("[MARCUS] I'll keep the best version saved. Re-run to continue.")
    else:
        log(f"\n[OLIVIA] Current best: WR {current.get('win_rate',0):.1f}% — targets not met.")
        log("[MARCUS] Best config saved. Re-run to continue optimisation.")

    log(f"\n[DONE] Log saved to: {LOG_FILE}")
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
