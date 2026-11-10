"""Chạy:  python train.py
Tải UCI Individual Household (nếu chưa có) -> tổng hợp theo tháng -> walk-forward 4 mô hình -> in bảng -> lưu joblib."""
import os
import ml
from main import MON, RAW

if not os.path.exists(RAW):
    print("Đang tải dataset UCI Individual Household (khoảng 20 MB)...")
    ml.download(RAW)

df = ml.uci_monthly(RAW)
df.to_csv(MON, index=False)
print(f"{len(df)} tháng hợp lệ ({df['month'].min():%Y-%m} đến {df['month'].max():%Y-%m}). Đã lưu {MON}")

res = ml.evaluate(df)
print(f"\nWalk-forward trên {res['test_months']} tháng cuối")
print(f"{'Mô hình':<20}{'MAE':>8}{'RMSE':>8}{'MAPE%':>8}{'WAPE%':>8}{'R2':>8}")
for n, m in res["compare"].items():
    print(f"{n + (' *' if n == res['best'] else ''):<20}{m['mae']:>8}{m['rmse']:>8}{m['mape']:>8}{m['wape']:>8}{m['r2']:>8}")
print("\nĐặc trưng ảnh hưởng nhất (SHAP):", ", ".join(f"{x['f']} {x['v']}%" for x in res["importance"][:5]))
ml.save(res, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
print("Đã lưu mô hình vào backend/models/")