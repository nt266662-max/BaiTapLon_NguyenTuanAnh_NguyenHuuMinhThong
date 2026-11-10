import io, os, zipfile, urllib.request
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
FEATS = ["lag_1", "lag_2", "lag_3", "lag_12", "roll3_mean", "roll3_std", "month_sin", "month_cos"]


# ---------- 1. Dữ liệu: UCI Individual Household -> bảng tháng chuẩn (household_id, month, kwh) ----------
def download(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw = urllib.request.urlopen(URL, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = [n for n in z.namelist() if n.endswith("household_power_consumption.txt")][0]
        with open(path, "wb") as f:
            f.write(z.read(name))


def uci_monthly(src):
    """File TXT phân tách ';', '?' là thiếu. Global_active_power (kW mỗi phút) -> kWh = kW/60.
    Ngày hợp lệ: có >= 1200/1440 phút. Tháng hợp lệ: >= 90% số ngày hợp lệ.
    kWh tháng = trung bình kWh/ngày (đã quy đủ 1440 phút) x số ngày của tháng."""
    d = pd.read_csv(src, sep=";", usecols=["Date", "Global_active_power"], na_values=["?"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], format="%d/%m/%Y", errors="coerce")
    d["Global_active_power"] = pd.to_numeric(d["Global_active_power"], errors="coerce")
    g = d.dropna(subset=["Date"]).groupby("Date")["Global_active_power"].agg(["sum", "count"])
    g = g[g["count"] >= 1200]
    g["kwh"] = g["sum"] / 60 * 1440 / g["count"]
    m = g["kwh"].resample("MS").agg(["mean", "count"])
    m["dim"] = m.index.days_in_month
    m = m[m["count"] >= 0.9 * m["dim"]]
    if m.empty:
        raise ValueError("Không tạo được tháng hợp lệ nào từ file này.")
    return pd.DataFrame({"household_id": "UCI_01", "month": m.index, "kwh": (m["mean"] * m["dim"]).round(1).values})


def prep_monthly(df):
    df = df.rename(columns=lambda c: str(c).strip().lower())
    miss = [c for c in ("household_id", "month", "kwh") if c not in df]
    if miss:
        raise ValueError("CSV cần các cột: household_id, month, kwh (thiếu: " + ", ".join(miss) + ")")
    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
    df = df.dropna(subset=["month", "kwh"])
    df["household_id"] = df["household_id"].astype(str)
    return df.drop_duplicates(["household_id", "month"], keep="last")[["household_id", "month", "kwh"]]


# ---------- 2. Đặc trưng (chỉ dùng thông tin đến tháng t-1, không rò rỉ) ----------
def feat(y, m):
    return {"lag_1": y[-1], "lag_2": y[-2], "lag_3": y[-3], "lag_12": y[-12],
            "roll3_mean": float(np.mean(y[-3:])), "roll3_std": float(np.std(y[-3:])),
            "month_sin": float(np.sin(2 * np.pi * m.month / 12)), "month_cos": float(np.cos(2 * np.pi * m.month / 12))}


def history(df):
    return {h: g.set_index("month")["kwh"].sort_index().asfreq("MS") for h, g in df.groupby("household_id")}


def table(H):
    rows = []
    for h, s in H.items():
        y = s.values
        for i in range(12, len(y)):
            if np.isnan(y[i]):
                continue
            f = feat(y[:i], s.index[i])
            if not any(np.isnan(v) for v in f.values()):
                rows.append({"household_id": h, "month": s.index[i], "y": float(y[i]), **f})
    return pd.DataFrame(rows)


# ---------- 3. Mô hình và đánh giá ----------
def make():
    ms = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=0, n_jobs=-1)}
    if XGBRegressor:
        ms["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=.05, random_state=0)
    return ms


def score(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"mae": round(float(np.abs(y - p).mean()), 2), "rmse": round(float(np.sqrt(((y - p) ** 2).mean())), 2),
            "mape": round(float((np.abs(y - p) / np.abs(y)).mean() * 100), 2),
            "wape": round(float(np.abs(y - p).sum() / np.abs(y).sum() * 100), 2),
            "r2": round(float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 3)}


def evaluate(df):
    H = history(df)
    T = table(H)
    if T.empty or T["month"].nunique() < 15:
        raise ValueError("Cần ít nhất khoảng 27 tháng liên tục cho mỗi hộ (12 tháng đầu để tạo lag, rồi mới có dữ liệu huấn luyện).")
    months = sorted(T["month"].unique())
    test = months[-min(12, len(months) // 3):]
    P = {n: [] for n in make()}
    P["Seasonal Naive"] = []
    A, ref = [], []
    for tm in test:                                      # walk-forward: học đến tháng tm-1, dự đoán tháng tm
        tr, te = T[T["month"] < tm], T[T["month"] == tm]
        if len(tr) < 10:
            continue
        A += te["y"].tolist()
        ref.append(te[["household_id", "month"]])
        for n, m in make().items():
            m.fit(tr[FEATS], tr["y"])
            P[n] += m.predict(te[FEATS]).tolist()
        P["Seasonal Naive"] += te["lag_12"].tolist()      # cùng tháng năm trước
    if not A:
        raise ValueError("Không đủ dữ liệu để đánh giá walk-forward.")
    cmp = {n: score(A, p) for n, p in P.items()}
    best = min(cmp, key=lambda n: cmp[n]["wape"])
    tree = min([n for n in cmp if n in ("Random Forest", "XGBoost")], key=lambda n: cmp[n]["wape"])
    final = {n: m.fit(T[FEATS], T["y"]) for n, m in make().items()}
    imp = final[tree].feature_importances_
    try:
        import shap
        imp = np.abs(shap.TreeExplainer(final[tree]).shap_values(T[FEATS])).mean(0)
    except Exception:
        pass
    imp = imp / imp.sum() * 100
    importance = sorted([{"f": c, "v": round(float(v), 1)} for c, v in zip(FEATS, imp)], key=lambda x: -x["v"])
    wf = pd.concat(ref).assign(actual=A, pred=P[best])
    return {"H": H, "models": final, "compare": cmp, "best": best, "tree": tree, "importance": importance, "wf": wf, "test_months": len(test)}


def predict_next(H, models, name):
    out = []
    for h, s in H.items():
        # UCI có vài tháng bị loại (thiếu dữ liệu) nên chuỗi có lỗ hổng. Nội suy lỗ hổng ngắn (<= 2 tháng)
        # để vẫn dự đoán được; nếu lỗ hổng dài hơn thì bỏ qua hộ đó.
        y = s.interpolate(limit=2, limit_area="inside").values
        if len(y) < 12 or np.isnan(y[-12:]).any():
            continue
        nm = s.index[-1] + pd.offsets.MonthBegin(1)
        f = feat(y, nm)
        p = float(f["lag_12"]) if name == "Seasonal Naive" else float(models[name].predict(pd.DataFrame([f])[FEATS])[0])
        out.append({"id": h, "month": nm.strftime("%Y-%m"), "last": round(float(y[-1]), 1), "pred": round(max(p, 0), 1)})
    return out


def save(res, folder):
    os.makedirs(folder, exist_ok=True)
    for n, m in res["models"].items():
        joblib.dump({"model": m, "features": FEATS}, os.path.join(folder, n.replace(" ", "_") + ".joblib"))