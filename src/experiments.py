"""Travel Reviews deneyleri: siniflandirma, sizintisiz gorev ve kumeleme.

Calistirma (proje kok dizininden):
    python src/experiments.py

Cikti: results/*.csv, results/summary.json, docs/images/*.png
"""
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, calinski_harabasz_score,
    cohen_kappa_score, confusion_matrix, davies_bouldin_score, f1_score,
    log_loss, matthews_corrcoef, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV, KFold, RepeatedStratifiedKFold, StratifiedKFold,
    cross_validate, train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
IMG = ROOT / "docs" / "images"
RES.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

CAT_NAMES = {
    "Category 1": "Sanat galerileri", "Category 2": "Dans kulüpleri",
    "Category 3": "Meyve suyu barları", "Category 4": "Restoranlar",
    "Category 5": "Müzeler", "Category 6": "Tatil köyleri",
    "Category 7": "Park/piknik", "Category 8": "Plajlar",
    "Category 9": "Tiyatrolar", "Category 10": "Dini mekanlar",
}


def label(c):
    return f"{c.split()[1]} – {CAT_NAMES[c]}"


# ---------------------------------------------------------------- veri
df = pd.read_csv(ROOT / "tripadvisor_review.xls")  # uzantisi .xls, icerigi CSV
feat_cols = [c for c in df.columns if c.startswith("Category")]
X = df[feat_cols]
df["Target"] = X.idxmax(axis=1)
le = LabelEncoder()
y = le.fit_transform(df["Target"])
summary = {"n_users": int(len(df)), "n_features": len(feat_cols),
           "missing_values": int(df.isna().sum().sum()),
           "class_counts": df["Target"].value_counts().to_dict()}
desc = X.describe().T
desc["skew"] = X.skew()
desc.to_csv(RES / "descriptive_stats.csv")
print("Sınıf dağılımı:\n", df["Target"].value_counts())


BEST_K = [4]  # CV ile secilince guncellenir


def models():
    return {
        "Karar Ağacı": DecisionTreeClassifier(random_state=SEED),
        "Rastgele Orman": RandomForestClassifier(random_state=SEED),
        "Lojistik Regresyon": LogisticRegression(max_iter=1000, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=BEST_K[0]),
        "SVM (RBF)": SVC(probability=True, random_state=SEED),
        "Naive Bayes": GaussianNB(),
        "Yapay Sinir Ağı (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000,
                                              random_state=SEED),
    }


# --------------------------------------------- A) orijinal gorev (hold-out)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)  # orijinal bolme

# KNN icin k, test seti yerine egitim seti uzerinde 5-kat CV ile secilir
knn_grid = GridSearchCV(make_pipeline(MinMaxScaler(), KNeighborsClassifier()),
                        {"kneighborsclassifier__n_neighbors": range(1, 21)},
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="accuracy")
knn_grid.fit(Xtr, ytr)
best_k = knn_grid.best_params_["kneighborsclassifier__n_neighbors"]
knn_curve_cv = knn_grid.cv_results_["mean_test_score"]
knn_curve_test = []
for k in range(1, 21):
    m = make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=k)).fit(Xtr, ytr)
    knn_curve_test.append(accuracy_score(yte, m.predict(Xte)))
BEST_K[0] = int(best_k)
summary["knn_best_k_cv"] = int(best_k)
summary["knn_best_k_test_original"] = int(np.argmax(knn_curve_test) + 1)
fig, ax = plt.subplots(figsize=(7, 4.2))
ks = list(range(1, 21))
ax.plot(ks, knn_curve_cv, "o-", label="5-kat CV (eğitim seti) – seçim ölçütü")
ax.plot(ks, knn_curve_test, "s--", alpha=.7, label="Test seti")
ax.axvline(best_k, color="gray", ls=":")
ax.set(xlabel="k (komşu sayısı)", ylabel="Doğruluk", title="KNN için k seçimi")
ax.legend()
fig.tight_layout(); fig.savefig(IMG / "knn-k-degeri-dogruluk.png", dpi=150); plt.close(fig)


def metrics(y_true, y_pred, proba=None, classes=None):
    out = {
        "Doğruluk": accuracy_score(y_true, y_pred),
        "Dengeli Doğruluk": balanced_accuracy_score(y_true, y_pred),
        "Makro Kesinlik": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Makro Duyarlılık": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Makro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Ağırlıklı F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Cohen Kappa": cohen_kappa_score(y_true, y_pred),
    }
    if proba is not None:  # egitimde gorulmeyen siniflara ~0 olasilik atanir
        full = np.full((len(proba), len(le.classes_)), 1e-15)
        full[:, classes] = proba
        out["Log-Loss"] = log_loss(y_true, full / full.sum(axis=1, keepdims=True),
                                   labels=np.arange(len(le.classes_)))
    return out


rows, orig_rows, preds = [], [], {}
allcls = np.arange(len(le.classes_))
dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
rows.append({"Model": "Taban çizgisi (çoğunluk sınıfı)", **metrics(yte, dummy.predict(Xte))})
for name, mdl in models().items():  # orijinal calisma: olceklenmemis ham veri
    raw = mdl.fit(Xtr, ytr)
    orig_rows.append({"Model": name, "Doğruluk (ham veri, orijinal)": accuracy_score(yte, raw.predict(Xte))})
for name, mdl in models().items():
    pipe = make_pipeline(MinMaxScaler(), mdl).fit(Xtr, ytr)
    p = pipe.predict(Xte)
    pr = pipe.predict_proba(Xte)
    preds[name] = p
    rows.append({"Model": name, **metrics(yte, p, pr, pipe.classes_)})
holdout = pd.DataFrame(rows).set_index("Model")
holdout.to_csv(RES / "holdout_metrics.csv")
pd.DataFrame(orig_rows).set_index("Model").to_csv(RES / "original_reproduction.csv")
print(holdout.round(4).to_string())
print(pd.DataFrame(orig_rows).round(4).to_string())

# oracle kural: etiket girdinin argmax'i oldugundan trivial kural %100 verir (sizinti kaniti)
oracle = np.array([np.where(le.classes_ == c)[0][0] for c in Xte.idxmax(axis=1)])
summary["oracle_argmax_accuracy"] = float(accuracy_score(yte, oracle))
summary["majority_share"] = float(df["Target"].value_counts(normalize=True).iloc[0])

# 5x3 tekrarli stratified CV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
scoring = {"acc": "accuracy", "bacc": "balanced_accuracy", "f1m": "f1_macro", "f1w": "f1_weighted"}
cv_rows, fold_scores = [], {}
for name, mdl in models().items():
    r = cross_validate(make_pipeline(MinMaxScaler(), mdl), X, y, cv=cv, scoring=scoring, n_jobs=-1)
    fold_scores[name] = r["test_acc"]
    cv_rows.append({"Model": name,
                    "Doğruluk": f"{r['test_acc'].mean():.4f} ± {r['test_acc'].std():.4f}",
                    "Dengeli Doğruluk": f"{r['test_bacc'].mean():.4f} ± {r['test_bacc'].std():.4f}",
                    "Makro F1": f"{r['test_f1m'].mean():.4f} ± {r['test_f1m'].std():.4f}",
                    "Ağırlıklı F1": f"{r['test_f1w'].mean():.4f} ± {r['test_f1w'].std():.4f}",
                    "_acc": r["test_acc"].mean()})
cv_df = pd.DataFrame(cv_rows).sort_values("_acc", ascending=False).drop(columns="_acc").set_index("Model")
cv_df.to_csv(RES / "cv_metrics.csv")
print(cv_df.to_string())

order = sorted(fold_scores, key=lambda n: -fold_scores[n].mean())
wil = {}
for a, b in [(order[0], "Lojistik Regresyon"), (order[0], "Naive Bayes"), ("Rastgele Orman", "Karar Ağacı")]:
    d = fold_scores[a] - fold_scores[b]
    wil[f"{a} vs {b}"] = None if np.allclose(d, 0) else float(wilcoxon(fold_scores[a], fold_scores[b]).pvalue)
summary["wilcoxon_p"] = wil

fig, ax = plt.subplots(figsize=(9, 4.8))
names = list(models().keys())
cv_means = [fold_scores[n].mean() for n in names]
xx = np.arange(len(names))
w = .27
ax.bar(xx - w, holdout.loc[names, "Doğruluk"], w, label="Doğruluk (hold-out)")
ax.bar(xx, cv_means, w, label="Doğruluk (5x3 CV)")
ax.bar(xx + w, holdout.loc[names, "Makro F1"].values, w, label="Makro F1 (hold-out)")
ax.set_xticks(xx)
ax.set_xticklabels(names, rotation=20, ha="right")
ax.set_ylim(0, 1.05)
ax.set_title("Model karşılaştırması")
ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig(IMG / "model-karsilastirmasi.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, nm in zip(axs, ["Rastgele Orman", "Naive Bayes"]):
    cm = confusion_matrix(yte, preds[nm], labels=allcls)
    present = sorted(set(yte) | set(preds[nm]))
    tick = [le.classes_[i].replace("Category ", "K") for i in present]
    sns.heatmap(cm[np.ix_(present, present)], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=tick, yticklabels=tick)
    ax.set(title=f"Karışıklık matrisi – {nm}", xlabel="Tahmin", ylabel="Gerçek")
fig.tight_layout(); fig.savefig(IMG / "karisiklik-matrisi.png", dpi=150); plt.close(fig)

# ozellik onemi: impurity ve permutation
rf = RandomForestClassifier(random_state=SEED).fit(Xtr, ytr)
perm = permutation_importance(rf, Xte, yte, n_repeats=30, random_state=SEED)
imp = pd.DataFrame({"Özellik": feat_cols, "Impurity (Gini)": rf.feature_importances_,
                    "Permütasyon": perm.importances_mean}).set_index("Özellik")
imp.to_csv(RES / "feature_importance.csv")
fig, ax = plt.subplots(figsize=(8, 5))
imp.sort_values("Impurity (Gini)").rename(index=label).plot.barh(ax=ax)
ax.set_title("Rastgele Orman özellik önemi")
fig.tight_layout(); fig.savefig(IMG / "ozellik-onemi-karsilastirma.png", dpi=150); plt.close(fig)
dt = DecisionTreeClassifier(random_state=SEED).fit(Xtr, ytr)
fig, ax = plt.subplots(figsize=(8, 4.5))
pd.Series(dt.feature_importances_, index=[label(c) for c in feat_cols]).sort_values().plot.barh(ax=ax, color="steelblue")
ax.set_title("Karar ağacı özellik önemi")
fig.tight_layout(); fig.savefig(IMG / "karar-agaci-ozellik-onemi.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
mm = X.mean().rename(index=label)
sns.barplot(x=mm.values, y=mm.index, hue=mm.index, palette="viridis", legend=False, ax=ax)
ax.set(xlabel="Ortalama puan (0–4)", title="Kategorilere göre ortalama puanlar")
fig.tight_layout(); fig.savefig(IMG / "kategori-ortalama-puanlar.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5.5))
tl = [c.split()[1] for c in feat_cols]
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, xticklabels=tl, yticklabels=tl)
ax.set_title("Kategori puanları korelasyon matrisi")
fig.tight_layout(); fig.savefig(IMG / "korelasyon-matrisi.png", dpi=150); plt.close(fig)
corr = X.corr().values
summary["max_abs_corr"] = float(np.abs(corr[np.triu_indices_from(corr, 1)]).max())

# ------------------------ B) sizintisiz gorev: Category 10 yuksek mi?
target_cat = "Category 10"
Xb = X.drop(columns=target_cat)
thr = float(X[target_cat].median())
yb = (X[target_cat] > thr).astype(int)
summary["task_b_threshold_median"] = thr
summary["task_b_pos_rate"] = float(yb.mean())
bmodels = models()
bmodels["Taban çizgisi"] = DummyClassifier(strategy="prior")
brows = []
for name, mdl in bmodels.items():
    r = cross_validate(make_pipeline(MinMaxScaler(), mdl), Xb, yb, cv=cv, n_jobs=-1,
                       scoring={"acc": "accuracy", "bacc": "balanced_accuracy", "f1": "f1",
                                "auc": "roc_auc", "ap": "average_precision", "mcc": "matthews_corrcoef",
                                "brier": "neg_brier_score"})
    brows.append({"Model": name, "Doğruluk": r["test_acc"].mean(), "Dengeli Doğruluk": r["test_bacc"].mean(),
                  "F1": r["test_f1"].mean(), "ROC-AUC": r["test_auc"].mean(),
                  "PR-AUC": r["test_ap"].mean(), "MCC": r["test_mcc"].mean(),
                  "Brier": -r["test_brier"].mean()})
taskb = pd.DataFrame(brows).set_index("Model").sort_values("ROC-AUC", ascending=False)
taskb.to_csv(RES / "task_b_leakage_free.csv")
print(taskb.round(4).to_string())
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=.2, random_state=SEED, stratify=yb)
fig, ax = plt.subplots(figsize=(6, 5))
for name in ["Rastgele Orman", "Lojistik Regresyon", "SVM (RBF)", "Naive Bayes"]:
    p = make_pipeline(MinMaxScaler(), bmodels[name]).fit(Xb_tr, yb_tr).predict_proba(Xb_te)[:, 1]
    fpr, tpr, _ = roc_curve(yb_te, p)
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(yb_te, p):.3f})")
ax.plot([0, 1], [0, 1], "k:")
ax.set(xlabel="Yanlış pozitif oranı", ylabel="Doğru pozitif oranı", title="ROC – Kategori 10 yüksek mi? (sızıntısız)")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(IMG / "roc-egrileri.png", dpi=150); plt.close(fig)

reg_rows = []
for name, regr in {"Ridge": Ridge(), "Rastgele Orman Regresyon": RandomForestRegressor(random_state=SEED, n_jobs=-1)}.items():
    maes, rmses, r2s = [], [], []
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(Xb):
        regr.fit(Xb.iloc[tr], X[target_cat].iloc[tr])
        p = regr.predict(Xb.iloc[te])
        t = X[target_cat].iloc[te]
        maes.append(mean_absolute_error(t, p)); rmses.append(mean_squared_error(t, p) ** .5); r2s.append(r2_score(t, p))
    reg_rows.append({"Model": name, "MAE": np.mean(maes), "RMSE": np.mean(rmses), "R²": np.mean(r2s)})
bt = X[target_cat]
reg_rows.append({"Model": "Taban çizgisi (ortalama)", "MAE": float(np.mean(np.abs(bt - bt.mean()))),
                 "RMSE": float(bt.std(ddof=0)), "R²": 0.0})
reg = pd.DataFrame(reg_rows).set_index("Model")
reg.to_csv(RES / "task_b_regression.csv")
print(reg.round(4).to_string())

# --------------------------------------------------- C) kumeleme
Xs = MinMaxScaler().fit_transform(X)
krows = []
for k in range(2, 9):
    lab = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Xs)
    krows.append({"k": k, "Silhouette": silhouette_score(Xs, lab),
                  "Davies-Bouldin": davies_bouldin_score(Xs, lab),
                  "Calinski-Harabasz": calinski_harabasz_score(Xs, lab)})
kdf = pd.DataFrame(krows).set_index("k")
kdf.to_csv(RES / "kmeans_metrics.csv")
print(kdf.round(4).to_string())
best_c = int(kdf["Silhouette"].idxmax())
summary["kmeans_best_k_silhouette"] = best_c
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, col in zip(axs, kdf.columns):
    ax.plot(kdf.index, kdf[col], "o-")
    ax.set(title=col, xlabel="k")
fig.tight_layout(); fig.savefig(IMG / "kmeans-metrikleri.png", dpi=150); plt.close(fig)
lab = KMeans(n_clusters=best_c, n_init=10, random_state=SEED).fit_predict(Xs)
prof = X.groupby(lab).mean()
prof["Kullanıcı sayısı"] = pd.Series(lab).value_counts().sort_index()
prof.to_csv(RES / "kmeans_profiles.csv")
fig, ax = plt.subplots(figsize=(9, 3 + .4 * best_c))
sns.heatmap(prof.drop(columns="Kullanıcı sayısı"), annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, xticklabels=tl)
ax.set(title=f"K-Means (k={best_c}) küme profilleri – ortalama puan", ylabel="Küme", xlabel="Kategori")
fig.tight_layout(); fig.savefig(IMG / "kmeans-profilleri.png", dpi=150); plt.close(fig)

(RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
