# -*- coding: utf-8 -*-
"""看板沙箱/正式切换端到端冒烟。"""
import json, re, sys, threading, time, urllib.request, urllib.error
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"d:/Git/AnalysisOfLearningBehavior/选题7_最终交付/03_源代码")
import model_console as mc
from http.server import ThreadingHTTPServer

httpd = ThreadingHTTPServer(("127.0.0.1", 8034), mc.Handler)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
time.sleep(0.3)
BASE = "http://127.0.0.1:8034"
g = lambda p: urllib.request.urlopen(BASE + p, timeout=15).read().decode("utf-8")

canon = g("/dash/dashboard.html")
sand = g("/dash/dashboard.html?scope=sandbox")
d_canon = json.loads(canon.split('application/json">')[1].split("</script>")[0])
d_sand = json.loads(sand.split('application/json">')[1].split("</script>")[0])
print("canonical N:", d_canon["overview"]["n"], "| sandbox N:", d_sand["overview"]["n"],
      "| data differs:", d_canon["overview"]["n"] != d_sand["overview"]["n"])
print("sandbox K:", d_sand.get("k"), "| canonical K:", d_canon.get("k"))
print("badge 正式:", "当前显示：正式结果" in canon)
print("badge 沙箱:", "当前显示：沙箱测试数据" in sand)
print("sandbox 无正式横幅:", "当前显示：正式结果" not in sand)
print("fig_prefix sandbox:", d_sand.get("fig_prefix"), "| canonical:", d_canon.get("fig_prefix"))
# 沙箱证据图可访问
b = urllib.request.urlopen(BASE + "/dash/figures/sandbox/up_pca_clusters.png", timeout=15).read()
print("sandbox fig png:", b[:4] == b"\x89PNG", len(b), "B")
# 形态/画像看板沙箱版应 404（未构建）
for u in ("/dash/persona_dashboard.html?scope=sandbox",
          "/dash/profile_dashboard.html?scope=sandbox"):
    try:
        urllib.request.urlopen(BASE + u, timeout=15)
        print(u, "UNEXPECTED 200")
    except urllib.error.HTTPError as e:
        print(u.split('/')[-1], "HTTP", e.code, "(跳过构建)")
print("SMOKE OK")
