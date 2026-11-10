# save_images.py — render every tab of an sbatchman-visualize workspace to PNG,
# as the app would: the tab's SQL (at most 10000 rows, the app's limit), its
# transform and its layout script. One file per tab: <out_dir>/<workspace>_<tab>.png
#
# Run with the Python of sbatchman (it has pandas and matplotlib):
#     PY=$(head -1 $(which sbatchman) | sed 's/^#!//')
#     $PY tools/save_images.py SbatchMan/data.sqlite dgx-spark/barriers/plots-warp0.json slides/images/barriers
import json, os, re, sqlite3, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

db_path, workspace, out_dir = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)
ws = json.load(open(workspace))
dpi = ws.get("style", {}).get("dpi", 160)
prefix = os.path.splitext(os.path.basename(workspace))[0]
db = sqlite3.connect(db_path)
for t in ws["tabs"]:
    p = t["single"]
    g = {"df": pd.read_sql(p["sql"], db).head(10000), "pd": pd}
    exec(p["transformScript"], g)
    exec(p["layoutScript"], g)
    fig = plt.figure()
    g["customize_matplotlib"](fig, fig.gca(), {})
    slug = re.sub(r"[^a-z0-9]+", "-", t["label"].lower()).strip("-")
    path = os.path.join(out_dir, f"{prefix}_{slug}.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(path)
