import io, os
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
import ml

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "household_power_consumption.txt")   # UCI Individual Household
MON = os.path.join(BASE, "data", "monthly_uci.csv")                    # dataset đã xử lý
app = FastAPI(title="SMART ENERGY PRO")
S = {}


def sample():
    r = np.random.default_rng(7)
    ms = pd.date_range("2018-01-01", periods=48, freq="MS")
    rows = []
    for h in range(1, 13):
        b, a, tr = 140 + 22 * h + r.random() * 50, .12 + r.random() * .22, r.random() * .8
        for i, m in enumerate(ms):
            rows.append({"household_id": f"HH{h:02d}", "month": m,
                         "kwh": round(b * (1 + a * np.cos(2 * np.pi * (m.month - 4) / 12)) + tr * i + (r.random() - .5) * b * .12, 1)})
    return pd.DataFrame(rows)


def setup(df, name):
    r = ml.evaluate(df)
    S.update(df=df, name=name, H=r["H"], models=r["models"], cmp=r["compare"], best=r["best"], tree=r["tree"],
             imp=r["importance"], wf=r["wf"], nt=r["test_months"])
    ml.save(r, os.path.join(BASE, "models"))


@app.on_event("startup")
def startup():
    try:
        if not os.path.exists(RAW):
            ml.download(RAW)
    except Exception as e:
        print("Không tải được dataset UCI, dùng dữ liệu mẫu:", e)
    if os.path.exists(RAW):
        df = ml.uci_monthly(RAW)
        df.to_csv(MON, index=False)
        setup(df, "UCI Individual Household")
    else:
        setup(sample(), "Dữ liệu mẫu giả lập (chưa có household_power_consumption.txt)")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        if "Global_active_power" in raw[:400].decode("utf-8", "ignore"):
            df = ml.uci_monthly(io.BytesIO(raw))
        else:
            df = ml.prep_monthly(pd.read_csv(io.BytesIO(raw), sep=None, engine="python"))
        setup(df, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.get("/api/state")
def state():
    df = S["df"]
    tot = df.groupby("month")["kwh"].sum()
    return {"name": S["name"], "n_hh": int(df["household_id"].nunique()), "months": int(len(tot)),
            "from": tot.index[0].strftime("%Y-%m"), "to": tot.index[-1].strftime("%Y-%m"),
            "total_kwh": round(float(df["kwh"].sum()), 1), "avg": round(float(df["kwh"].mean()), 1),
            "series": [{"m": m.strftime("%Y-%m"), "v": round(float(v), 1)} for m, v in tot.items()],
            "households": sorted(S["H"])[:1000], "compare": S["cmp"], "best": S["best"], "tree": S["tree"],
            "importance": S["imp"], "nt": S["nt"],
            "rows": df.sort_values(["household_id", "month"]).head(500).assign(month=lambda x: x["month"].dt.strftime("%Y-%m")).to_dict("records")}


@app.get("/api/household/{hid}")
def household(hid: str):
    s = S["H"].get(hid)
    if s is None:
        raise HTTPException(404, "Không có hộ này")
    preds = {}
    for n in S["cmp"]:
        p = ml.predict_next({hid: s}, S["models"], n)
        if p:
            preds[n] = p[0]
    wf = S["wf"][S["wf"]["household_id"] == hid]
    return {"id": hid, "best": S["best"], "preds": preds,
            "history": [{"m": m.strftime("%Y-%m"), "v": None if np.isnan(v) else round(float(v), 1)} for m, v in s.items()],
            "wf": [{"m": r.month.strftime("%Y-%m"), "actual": round(r.actual, 1), "pred": round(r.pred, 1)} for r in wf.itertuples()]}


@app.get("/api/multi")
def multi(model: str = ""):
    model = model if model in S["cmp"] else S["best"]
    rows = ml.predict_next(S["H"], S["models"], model)
    for r in rows:
        r["change"] = round((r["pred"] - r["last"]) / r["last"] * 100, 1) if r["last"] else 0
    return {"model": model, "rows": sorted(rows, key=lambda r: -r["pred"])}


@app.post("/api/reset")
def reset():
    """Quay về dataset mặc định (UCI hoặc dữ liệu mẫu)."""
    startup()
    return {"ok": True}


app.mount("/", StaticFiles(directory=os.path.join(BASE, "..", "frontend"), html=True), name="ui")