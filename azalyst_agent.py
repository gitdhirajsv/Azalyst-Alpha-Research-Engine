"""
AZALYST AUTONOMOUS AGENT v2
Talks to your LOCAL Ollama at http://localhost:11434
No human input needed. Runs forever until alpha confirmed.

v2 FIXES vs v1:
- Targeted prompts: sends ONLY the relevant code section (not 7000 chars)
- Timeout set to 300s (was 600s — deepseek-r1:14b needs <5 min per small prompt)
- Surgical code extraction: pulls out just the function/constant being fixed
- Better error recovery and logging
- Signal flip already applied directly (root cause fix)
"""

import requests
import subprocess
import json
import os
import sys
import time
import csv
import shutil
import re
from datetime import datetime

PROJECT_PATH = r"D:\Personal Files\Azalyst Alpha Research Engine"
MODEL        = "deepseek-r1:14b"
OLLAMA_URL   = "http://localhost:11434/api/generate"
LOG_FILE     = os.path.join(PROJECT_PATH, "agent_log.txt")
RESULTS_FILE = os.path.join(PROJECT_PATH, "performance_metrics.csv")

TARGETS = {
    "win_rate":       55.0,   # Realistic target after signal flip
    "sharpe_ratio":    1.0,
    "max_drawdown":  -25.0,
    "profit_factor":   1.5,
}

BASELINE = {
    "win_rate":      27.5,
    "sharpe_ratio":  -5.27,
    "max_drawdown":  -50.04,
    "profit_factor":  0.26,
}

# ─────────────────────────────────────────────────────────────────────────────

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            log("[OK] Ollama running at http://localhost:11434")
            return True
    except Exception:
        pass
    return False

def ask_llm(prompt: str, timeout: int = 300) -> str:
    """
    Send a SHORT, targeted prompt to Ollama.
    Keep prompts under 2000 chars for deepseek-r1:14b to respond in < 5 min.
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1500,
            "num_gpu": 99,    # Use all GPU layers available
        }
    }
    try:
        log(f"[LLM] Sending prompt ({len(prompt)} chars) to Ollama...")
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        data = resp.json()
        answer = data.get("response", "").strip()
        # Strip thinking tags (deepseek-r1 format)
        if "<think>" in answer and "</think>" in answer:
            answer = answer.split("</think>")[-1].strip()
        log(f"[LLM] Response: {len(answer)} chars")
        return answer
    except requests.exceptions.Timeout:
        log(f"[LLM TIMEOUT] No response in {timeout}s — prompt may be too large")
        return ""
    except Exception as e:
        log(f"[LLM ERROR] {e}")
        return ""

def read_file(filename):
    path = os.path.join(PROJECT_PATH, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        log(f"[FILE ERROR] Cannot read {filename}: {e}")
        return ""

def write_file(filename, content):
    path = os.path.join(PROJECT_PATH, filename)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        log(f"[FILE] Written: {filename}")
        return True
    except Exception as e:
        log(f"[FILE ERROR] {e}")
        return False

def backup_file(filename):
    path = os.path.join(PROJECT_PATH, filename)
    if os.path.exists(path):
        shutil.copy2(path, path + ".bak")
        log(f"[BACKUP] {filename}.bak saved")

def restore_file(filename):
    path = os.path.join(PROJECT_PATH, filename)
    bak  = path + ".bak"
    if os.path.exists(bak):
        shutil.copy2(bak, path)
        log(f"[RESTORE] {filename} restored from backup")
        return True
    return False

def extract_section(code: str, target: str, lines_before: int = 5, lines_after: int = 20) -> str:
    """
    Extract a small section of code around a target keyword.
    This keeps prompts small enough for the LLM to handle quickly.
    """
    lines = code.split("\n")
    for i, line in enumerate(lines):
        if target in line:
            start = max(0, i - lines_before)
            end   = min(len(lines), i + lines_after)
            return "\n".join(lines[start:end])
    return ""

def apply_constant_fix(filename: str, constant_name: str, new_value: str) -> bool:
    """
    Directly replace a constant value without using the LLM.
    Faster and more reliable than asking the LLM for simple changes.
    """
    backup_file(filename)
    code = read_file(filename)
    if not code:
        return False

    # Pattern: find `CONSTANT_NAME = <value>`
    pattern = rf"^({re.escape(constant_name)}\s*=\s*)(.+)$"
    new_code, n = re.subn(pattern, rf"\g<1>{new_value}", code, flags=re.MULTILINE)

    if n == 0:
        log(f"[DIRECT FIX] '{constant_name}' not found in {filename}")
        restore_file(filename)
        return False

    log(f"[DIRECT FIX] '{constant_name}' → {new_value} ({n} occurrence(s))")
    return write_file(filename, new_code)

def apply_llm_fix(filename: str, section_keyword: str, instruction: str) -> bool:
    """
    Ask LLM to fix a specific SECTION of code (not the whole file).
    Extracts only the relevant lines, sends that to the LLM, then
    splices the fixed section back into the full file.
    """
    backup_file(filename)
    full_code = read_file(filename)
    if not full_code:
        return False

    section = extract_section(full_code, section_keyword, lines_before=3, lines_after=25)
    if not section:
        log(f"[LLM FIX] Keyword '{section_keyword}' not found in {filename}")
        return False

    prompt = f"""Fix this Python code section.

TASK: {instruction}

CURRENT CODE (section only):
{section}

Return ONLY the fixed code section. No explanation. No markdown."""

    fixed_section = ask_llm(prompt, timeout=240)
    if not fixed_section or len(fixed_section) < 30:
        log("[LLM FIX] Response too short — skipping")
        restore_file(filename)
        return False

    # Clean markdown if LLM added it
    for tag in ["```python", "```"]:
        if fixed_section.startswith(tag):
            fixed_section = fixed_section[len(tag):]
    if fixed_section.endswith("```"):
        fixed_section = fixed_section[:-3]
    fixed_section = fixed_section.strip()

    # Splice: replace original section with fixed section
    new_code = full_code.replace(section, fixed_section, 1)
    if new_code == full_code:
        log("[LLM FIX] Section unchanged after LLM — no improvement")
        restore_file(filename)
        return False

    return write_file(filename, new_code)

def read_results():
    try:
        with open(RESULTS_FILE, "r") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        last = rows[-1]
        return {
            "win_rate":      float(last.get("win_rate", 0)),
            "sharpe_ratio":  float(last.get("sharpe_ratio", -99)),
            "max_drawdown":  float(last.get("max_drawdown_pct", -99)),
            "profit_factor": float(last.get("profit_factor", 0)),
            "cycle":         last.get("cycle", "?"),
            "total_trades":  int(last.get("total_trades", 0)),
        }
    except Exception as e:
        log(f"[RESULTS ERROR] {e}")
        return {}

def all_targets_met(m):
    if not m:
        return False
    return (
        m.get("win_rate",      0)   >= TARGETS["win_rate"] and
        m.get("sharpe_ratio",  -99) >= TARGETS["sharpe_ratio"] and
        m.get("max_drawdown",  -99) >= TARGETS["max_drawdown"] and
        m.get("profit_factor", 0)   >= TARGETS["profit_factor"]
    )

def print_table(before, after, label=""):
    log("")
    log(f"  ── RESULTS {label} ──")
    log(f"  {'Metric':<22} {'Before':>8} {'After':>8} {'Change':>8}  Target")
    log(f"  {'-'*60}")
    items = [
        ("win_rate",      "Win Rate %",     f">{TARGETS['win_rate']}%"),
        ("sharpe_ratio",  "Sharpe",         f">{TARGETS['sharpe_ratio']}"),
        ("max_drawdown",  "Max Drawdown %", f">{TARGETS['max_drawdown']}%"),
        ("profit_factor", "Profit Factor",  f">{TARGETS['profit_factor']}"),
    ]
    for key, name, target_str in items:
        b  = before.get(key, 0)
        a  = after.get(key, 0)
        ch = a - b
        sign = "+" if ch >= 0 else ""
        log(f"  {name:<22} {b:>8.2f} {a:>8.2f} {sign}{ch:>7.2f}  {target_str}")
    log("")

def run_simulator(coins=None, no_resume=False):
    log("[SIM] Starting walkforward_simulator.py ...")
    cmd = [sys.executable, "walkforward_simulator.py",
           "--data-dir", "./data",
           "--feature-dir", "./feature_cache"]
    if no_resume:
        cmd.append("--no-resume")
    env = os.environ.copy()
    if coins:
        env["AZALYST_TEST_COINS"] = ",".join(coins)
    try:
        proc = subprocess.run(
            cmd, cwd=PROJECT_PATH,
            capture_output=True, text=True,
            timeout=7200, env=env
        )
        if proc.returncode == 0:
            log("[SIM] Completed successfully.")
        else:
            log(f"[SIM ERROR] Exit code {proc.returncode}")
            if proc.stderr:
                log(f"[SIM STDERR] {proc.stderr[-600:]}")
        return proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        log("[SIM] Timed out after 120 min.")
        return "", "timeout", 1
    except Exception as e:
        log(f"[SIM ERROR] {e}")
        return "", str(e), 1

def get_test_coins(n=25):
    data_dir = os.path.join(PROJECT_PATH, "data")
    try:
        files = sorted([
            f.replace(".parquet", "")
            for f in os.listdir(data_dir)
            if f.endswith(".parquet")
        ])
        return files[:n]
    except Exception:
        return []

def diagnose_with_llm(metrics: dict) -> str:
    """
    Ask LLM researcher persona to diagnose why performance is still bad.
    Sends only the metrics — NOT code — so prompt is tiny and fast.
    """
    prompt = f"""You are a senior quantitative researcher reviewing backtesting results.

CURRENT METRICS:
  Win Rate:      {metrics.get('win_rate', 0):.2f}%  (target: >{TARGETS['win_rate']}%)
  Sharpe:        {metrics.get('sharpe_ratio', 0):.3f}  (target: >{TARGETS['sharpe_ratio']})
  Max Drawdown:  {metrics.get('max_drawdown', 0):.2f}%  (target: >{TARGETS['max_drawdown']}%)
  Profit Factor: {metrics.get('profit_factor', 0):.3f}  (target: >{TARGETS['profit_factor']})
  Total Trades:  {metrics.get('total_trades', 0)}

CONTEXT: This is a crypto ML trading system using LightGBM + walk-forward simulation.
Signals: BUY when model prob < 0.40, SELL when model prob > 0.60 (counter-predictive flip applied).
Stop-loss: -1.5%, Take-profit: +2.7%

In 2-3 sentences, identify the most likely cause of underperformance.
Then give ONE specific code change to try next (e.g. "change STOP_LOSS_PCT from X to Y").
Be concise."""

    return ask_llm(prompt, timeout=180)

def main():
    os.chdir(PROJECT_PATH)

    log("=" * 65)
    log("  AZALYST AUTONOMOUS AGENT v2 STARTED")
    log(f"  Model  : {MODEL}")
    log(f"  API    : {OLLAMA_URL}  (local Ollama)")
    log("=" * 65)
    log("  KEY FIX ALREADY APPLIED: Signal direction flipped (BUY<->SELL)")
    log("  Expected win rate: 65-75%+ (was 27-30%)")
    log("=" * 65)

    if not check_ollama():
        log("")
        log("  Ollama is not running. Please:")
        log("  1. Open a new CMD window")
        log("  2. Run:  ollama serve")
        log("  3. Wait for it to start, then run this script again")
        log("")
        input("Press Enter when Ollama is running...")
        if not check_ollama():
            log("[ERROR] Still cannot connect. Exiting.")
            sys.exit(1)

    before = read_results() or BASELINE.copy()
    log(f"[STATE] Win Rate: {before.get('win_rate',0):.1f}%  "
        f"Sharpe: {before.get('sharpe_ratio',0):.2f}  "
        f"Trades: {before.get('total_trades',0):,}")

    test_coins  = get_test_coins(25)
    current_best = before.copy()
    run_count    = 0

    # ── STAGE 1: Run with signal flip already applied ─────────────────────────
    log("\n[STAGE 1] Running simulator with signal direction fix...")
    log("[STAGE 1] This is the main fix — BUY<->SELL signals inverted")
    log("[STAGE 1] Expected win rate to jump from ~27% to ~65-75%")

    run_count += 1
    stdout, stderr, code = run_simulator(test_coins if test_coins else None, no_resume=True)

    if code != 0:
        log(f"[STAGE 1 ERROR] Simulator crashed: {stderr[-500:]}")
        log("[STAGE 1] Check walkforward_simulator.py for issues")
    else:
        after = read_results()
        if after:
            print_table(before, after, "after signal flip")
            current_best = after.copy()

            if all_targets_met(after):
                log("[STAGE 1] ALL TARGETS MET! Skipping to Stage 3.")
            else:
                log(f"[STAGE 1] Win rate {after.get('win_rate',0):.1f}% — "
                    f"{'good, targets met' if all_targets_met(after) else 'needs more tuning'}")

    # ── STAGE 2: LLM-guided tuning ────────────────────────────────────────────
    after = read_results() or current_best
    if not all_targets_met(after):
        log("\n[STAGE 2] Targets not yet met — LLM researcher diagnosing...")

        diagnosis = diagnose_with_llm(after)
        if diagnosis:
            log(f"\n[RESEARCHER]\n{diagnosis}\n")
        else:
            log("[RESEARCHER] No diagnosis (LLM timeout — continuing with standard fixes)")

        # Standard fix queue — direct replacements (no LLM needed, instant)
        standard_fixes = [
            # Tighter threshold: only take highest confidence signals
            ("walkforward_simulator.py", "BUY_THRESHOLD",  "0.62"),
            ("walkforward_simulator.py", "SELL_THRESHOLD", "0.38"),
        ]

        for fname, const, val in standard_fixes:
            log(f"\n[FIX] {fname}: {const} → {val}")
            backup_file(fname)
            ok = apply_constant_fix(fname, const, val)
            if not ok:
                log(f"[FIX SKIP] Could not apply {const}")
                continue

            run_count += 1
            _, _, rc = run_simulator(test_coins if test_coins else None)
            if rc != 0:
                log(f"[FIX] Run failed — reverting {fname}")
                restore_file(fname)
                continue

            after_fix = read_results()
            if not after_fix:
                continue

            print_table(current_best, after_fix, f"{const}={val}")

            wr_improved = after_fix.get("win_rate",0) > current_best.get("win_rate",0)
            sr_improved = after_fix.get("sharpe_ratio",-99) > current_best.get("sharpe_ratio",-99)

            if wr_improved or sr_improved:
                log(f"[KEEP] {const}={val} improved performance")
                current_best = after_fix.copy()
            else:
                log(f"[REVERT] {const}={val} did not improve — reverting")
                restore_file(fname)

            if all_targets_met(after_fix):
                log("[STAGE 2] ALL TARGETS MET!")
                break

    # ── STAGE 3: Confirm with 3 consecutive runs ──────────────────────────────
    final_state = read_results() or current_best
    if all_targets_met(final_state):
        log("\n[STAGE 3] Confirming alpha — 3 consecutive passes needed...")
        passes = 0
        for i in range(3):
            run_count += 1
            log(f"[CONFIRM {i+1}/3] Running...")
            _, _, rc = run_simulator(test_coins if test_coins else None)
            m = read_results()
            if m:
                print_table(BASELINE, m, f"confirm {i+1}/3")
                if all_targets_met(m):
                    passes += 1
                    log(f"[CONFIRM {i+1}] PASS ({passes}/3)")
                else:
                    passes = 0
                    log(f"[CONFIRM {i+1}] FAIL — resetting count")
            time.sleep(2)

        # ── STAGE 4: Full universe ─────────────────────────────────────────────
        if passes >= 3:
            log("\n[STAGE 3] CONFIRMED — 3/3 passes on sample")
            log("[STAGE 4] Running on ALL coins...")
            run_count += 1
            _, _, rc = run_simulator(coins=None)
            final = read_results()
            if final:
                print_table(BASELINE, final, "FINAL — ALL COINS")
                if all_targets_met(final):
                    log("")
                    log("=" * 65)
                    log("  ALPHA STRATEGY CONFIRMED.")
                    log("  Small sample : PASS")
                    log("  Full universe: PASS")
                    log("  Ready for live deployment review.")
                    log("=" * 65)
                else:
                    log("[STAGE 4] Works on sample but needs tuning for full universe.")
        else:
            log(f"\n[STAGE 3] Only {passes}/3 passes. More iteration needed.")
    else:
        log(f"\n[SUMMARY] Current win rate {final_state.get('win_rate',0):.1f}% — targets not yet met.")
        log("[SUMMARY] Re-run agent to continue improving.")

    log(f"\n[DONE] Total runs: {run_count}")
    log(f"[DONE] Log: {LOG_FILE}")
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
