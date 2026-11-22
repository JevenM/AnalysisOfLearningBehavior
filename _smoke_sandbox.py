# -*- coding: utf-8 -*-
"""两种沙箱模式取数验证：上传 → 用已上传数据跑链；全管线应被拒绝且不覆盖。"""
import json, sys, threading, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"d:\Git\AnalysisOfLearningBehavior\选题7_最终交付\03_源代码")
import model_console as mc
from http.server import ThreadingHTTPServer
httpd = ThreadingHTTPServer(("127.0.0.1", 8043), mc.Handler)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
time.sleep(0.3)
BASE = "http://127.0.0.1:8043"
CSV = (r"d:\Git\AnalysisOfLearningBehavior\选题7_最终交付"
       r"\02_数据与数据集\测试上传数据\test_upload_harvardx_holdout.csv")

def unified_rows():
    with open(mc.SANDBOX + r"\up_unified.csv", encoding="utf-8-sig") as f:
        return sum(1 for _ in f) - 1

# 1) 上传留出数据
body = open(CSV, "rb").read()
req = urllib.request.Request(
    BASE + "/api/data?target=sandbox&filename=test_upload_harvardx_holdout.csv",
    data=body, method="POST", headers={"Content-Type": "text/csv"})
r = json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
print("[1] 上传:", r, flush=True)

# 2) 尝试沙箱全管线 → 应被拒绝
req = urllib.request.Request(
    BASE + "/api/run", data=json.dumps({"scope": "sandbox"}).encode(),
    headers={"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
print("[2] 沙箱全管线 → ok:", r.get("ok"), "| error:", (r.get("error") or "")[:80], flush=True)

# 3) 确认上传数据未被覆盖
print("[3] 上传数据仍在: 行数 =", unified_rows(), flush=True)

# 4) 用已上传数据跑链（真实链，约 2 分钟）
req = urllib.request.Request(
    BASE + "/api/run", data=json.dumps({"scope": "sandbox_upload"}).encode(),
    headers={"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
print("[4] sandbox_upload 启动:", r.get("ok"), r.get("stages", [])[:2], flush=True)
t0 = time.time()
while time.time() - t0 < 600:
    time.sleep(10)
    st = json.loads(urllib.request.urlopen(BASE + "/api/state", timeout=30).read().decode())
    if not st["run"]["running"] and st["run"].get("rc") is not None:
        break
print("[5] 链路 rc:", st["run"].get("rc"), f"耗时 {time.time()-t0:.0f}s", flush=True)

# 6) 终检：沙箱产物基于 8,000 行、画像 4 行
import csv
rows = unified_rows()
with open(mc.SANDBOX + r"\up_persona_recommendations.csv", encoding="utf-8-sig") as f:
    recs = list(csv.DictReader(f))
learners = sum(1 for _ in open(mc.SANDBOX + r"\up_learners.csv", encoding="utf-8-sig")) - 1
print(f"[6] unified={rows} 行 / learners={learners} 行 / 推荐表={len(recs)} 行", flush=True)
for rec in recs:
    print("   ", rec["persona"], rec["n_students"], "人", flush=True)
ok = rows == 8000 and learners == 8000 and len(recs) == 4 and st["run"].get("rc") == 0
print("SMOKE", "OK" if ok else "FAIL", flush=True)
