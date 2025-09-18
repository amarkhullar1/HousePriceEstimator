# train.py
import pandas as pd, numpy as np, joblib
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from pathlib import Path

DATA = "data/london_ppd_prepared.parquet"   # your cleaned/joined dataset
MODEL_DIR = Path("model"); MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_parquet(DATA)

# --- columns ---
TARGET = "price_paid"
CATS   = ["property_type","tenure","postcode","borough"]
NUMS   = ["floor_area","bedrooms","bathrooms","distance_to_station_km",
          "sale_year","sale_quarter","imd_decile","lat","lon"]
FEATS  = CATS + NUMS

df = df.dropna(subset=[TARGET, *FEATS]).copy()
df["log_price"] = np.log1p(df[TARGET])

# Time-aware split by year group (no leakage across time)
df["year_group"] = (df["sale_year"]//2)*2  # 2-year buckets for grouping
gkf = GroupKFold(n_splits=5)

oof, preds = [], []
for fold, (tr, va) in enumerate(gkf.split(df, groups=df["year_group"])):
    train_pool = Pool(df.iloc[tr][FEATS], label=df.iloc[tr]["log_price"], cat_features=CATS)
    valid_pool = Pool(df.iloc[va][FEATS], label=df.iloc[va]["log_price"], cat_features=CATS)

    model = CatBoostRegressor(
        depth=8, learning_rate=0.05, loss_function="MAE",
        n_estimators=4000, subsample=0.8, colsample_bylevel=0.8,
        random_seed=42, od_type="Iter", od_wait=200, verbose=False
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    p = model.predict(valid_pool)
    oof += list(np.expm1(p))
    preds += list(df.iloc[va][TARGET].values)

mae = mean_absolute_error(preds, oof)
print(f"OOF MAE: £{mae:,.0f}")  # sanity: ~£40–£80k depending on features

# Train final model on all data
final_pool = Pool(df[FEATS], label=df["log_price"], cat_features=CATS)
final_model = CatBoostRegressor(
    depth=8, learning_rate=0.05, loss_function="MAE",
    n_estimators=final_model.get_best_iteration()+100 if 'final_model' in locals() else 3000,
    subsample=0.8, colsample_bylevel=0.8, random_seed=42, verbose=False
)
final_model.fit(final_pool)

artifacts = {
    "model": final_model,
    "features": FEATS,
    "cat_features": CATS,
    "num_features": NUMS,
    "target": TARGET
}
joblib.dump(artifacts, MODEL_DIR / "house_price_catboost.joblib")
print("Saved model → model/house_price_catboost.joblib")
