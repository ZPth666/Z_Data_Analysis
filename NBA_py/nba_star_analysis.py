# -*- coding: utf-8 -*-
from pathlib import Path
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------- 0. Settings ----------
RANDOM_STATE = 42
MIN_GP = 20
MIN_MINUTES = 15
STAR_RATIO = 0.10

try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    PROJECT_DIR = Path.cwd()

OUTPUT_DIR = PROJECT_DIR / "NBA_star_analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "Arial Unicode MS",
    "Noto Sans CJK SC", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 1. Load data ----------
def find_input():
    csvs = [
        PROJECT_DIR / "NBA数据分析输出" / "清洗后_球员数据.csv",
        PROJECT_DIR / "清洗后_球员数据.csv",
    ]
    for p in csvs:
        if p.exists():
            return "csv", p

    excels = [
        PROJECT_DIR / "NBA_2025-2026_三表中文版(1).xlsx",
        PROJECT_DIR / "NBA_2025-2026_三表中文版.xlsx",
    ]
    for p in excels:
        if p.exists():
            return "xlsx", p

    for p in PROJECT_DIR.glob("*.xlsx"):
        if "三表中文版" in p.name:
            return "xlsx", p

    raise FileNotFoundError(
        "Cannot find input data. Put this file next to the notebook:\n"
        "NBA_2025-2026_三表中文版.xlsx"
    )

def load_players():
    typ, path = find_input()
    if typ == "csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
    else:
        df = pd.read_excel(path, sheet_name="球员数据")
    print("Loaded:", path)
    return df.copy()

# ---------- 2. Feature engineering ----------
def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return pd.Series(np.where(np.abs(b) > 1e-12, a / b, np.nan), index=a.index)

def normalize_position(x):
    s = str(x).strip()
    if s in ["后卫", "控球后卫", "得分后卫", "G", "PG", "SG"]:
        return "后卫"
    if s in ["前锋", "大前锋", "小前锋", "F", "PF", "SF"]:
        return "前锋"
    if s in ["中锋", "C"]:
        return "中锋"
    u = s.upper()
    if "G" in u:
        return "后卫"
    if "F" in u:
        return "前锋"
    if "C" in u:
        return "中锋"
    return s

def prepare(df):
    df = df.copy()

    num_cols = [
        "出场次数", "场均时间(分钟)", "场均得分",
        "场均投篮命中数", "场均投篮出手数",
        "场均三分命中数", "场均三分出手数",
        "场均罚球命中数", "场均罚球出手数",
        "场均篮板", "场均助攻", "场均抢断",
        "场均盖帽", "场均失误", "两双次数", "三双次数"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["位置大类"] = df["位置"].apply(normalize_position)

    df["每36分钟得分"] = safe_div(df["场均得分"] * 36, df["场均时间(分钟)"])
    df["每36分钟篮板"] = safe_div(df["场均篮板"] * 36, df["场均时间(分钟)"])
    df["每36分钟助攻"] = safe_div(df["场均助攻"] * 36, df["场均时间(分钟)"])

    df["有效投篮命中率eFG(%)"] = (
        safe_div(
            df["场均投篮命中数"] + 0.5 * df["场均三分命中数"],
            df["场均投篮出手数"]
        ) * 100
    )

    ts_den = 2 * (df["场均投篮出手数"] + 0.44 * df["场均罚球出手数"])
    df["真实命中率TS(%)"] = safe_div(df["场均得分"], ts_den) * 100

    df["三分出手占比"] = safe_div(df["场均三分出手数"], df["场均投篮出手数"])
    df["助攻失误比"] = df["场均助攻"] / (df["场均失误"] + 0.5)
    df["防守活动"] = df["场均抢断"] + df["场均盖帽"]

    df = df[
        (df["出场次数"] >= MIN_GP) &
        (df["场均时间(分钟)"] >= MIN_MINUTES)
    ].copy()

    return df.reset_index(drop=True)

# ---------- 3. Star index ----------
STAR_WEIGHTS = {
    "场均得分": 0.30,
    "真实命中率TS(%)": 0.15,
    "场均助攻": 0.15,
    "场均篮板": 0.10,
    "场均抢断": 0.10,
    "场均盖帽": 0.05,
    "场均时间(分钟)": 0.10,
    "助攻失误比": 0.05
}

def winsorize(s):
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    return s.clip(lo, hi)

def add_star_index(df):
    out = df.copy()
    features = list(STAR_WEIGHTS.keys())

    X = pd.DataFrame(index=out.index)
    for c in features:
        X[c] = winsorize(out[c].astype(float))

    scaler = StandardScaler()
    Z = pd.DataFrame(scaler.fit_transform(X), columns=features, index=out.index)

    score = np.zeros(len(out))
    for c, w in STAR_WEIGHTS.items():
        score += w * Z[c].values

    out["巨星指数"] = score
    out["巨星指数百分位"] = out["巨星指数"].rank(pct=True) * 100

    # same-position percentile index
    pos_score = np.zeros(len(out))
    for c, w in STAR_WEIGHTS.items():
        pct = out.groupby("位置大类")[c].transform(lambda s: s.rank(pct=True) * 100)
        pos_score += w * pct.values
    out["同位置表现指数"] = pos_score

    star_n = max(1, math.ceil(len(out) * STAR_RATIO))
    order = out["巨星指数"].rank(method="first", ascending=False)
    out["是否数据型巨星"] = np.where(order <= star_n, "巨星组", "其他球员")

    print("Eligible players:", len(out))
    print("Star players:", star_n)
    return out

# ---------- 4. Ranking ----------
def export_ranking(df):
    cols = [
        "球员姓名", "位置大类", "出场次数", "场均时间(分钟)",
        "场均得分", "真实命中率TS(%)", "场均篮板", "场均助攻",
        "场均抢断", "场均盖帽", "场均失误", "助攻失误比",
        "每36分钟得分", "两双次数", "三双次数",
        "巨星指数", "巨星指数百分位", "同位置表现指数",
        "是否数据型巨星"
    ]
    r = df[cols].sort_values("巨星指数", ascending=False).reset_index(drop=True)
    r.insert(0, "综合排名", np.arange(1, len(r) + 1))
    r.to_csv(OUTPUT_DIR / "01_star_ranking.csv", index=False, encoding="utf-8-sig")
    return r

# ---------- 5. Star vs others ----------
def cohens_d(a, b):
    a = pd.Series(a).dropna().astype(float)
    b = pd.Series(b).dropna().astype(float)
    pooled = (((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1))
              / (len(a)+len(b)-2))
    if pooled <= 0:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(pooled)

def compare_groups(df):
    features = [
        "场均得分", "每36分钟得分", "真实命中率TS(%)",
        "有效投篮命中率eFG(%)", "场均时间(分钟)",
        "场均篮板", "场均助攻", "场均抢断",
        "场均盖帽", "场均失误", "助攻失误比",
        "两双次数", "三双次数"
    ]
    stars = df[df["是否数据型巨星"] == "巨星组"]
    others = df[df["是否数据型巨星"] == "其他球员"]

    rows = []
    for c in features:
        rows.append({
            "指标": c,
            "巨星组均值": stars[c].mean(),
            "其他球员均值": others[c].mean(),
            "均值差": stars[c].mean() - others[c].mean(),
            "Cohen_d": cohens_d(stars[c], others[c])
        })
    out = pd.DataFrame(rows)
    out["绝对效应量"] = out["Cohen_d"].abs()
    out = out.sort_values("绝对效应量", ascending=False)
    out.to_csv(OUTPUT_DIR / "02_star_vs_others.csv", index=False, encoding="utf-8-sig")
    return out

# ---------- 6. Radar ----------
def radar_plot(df):
    dims = {
        "得分产量": "场均得分",
        "进攻效率": "真实命中率TS(%)",
        "组织能力": "场均助攻",
        "篮板贡献": "场均篮板",
        "防守活动": "防守活动",
        "比赛负荷": "场均时间(分钟)"
    }

    temp = df.copy()
    star_vals, other_vals = [], []

    for label, col in dims.items():
        p = temp[col].rank(pct=True) * 100
        temp[f"p_{label}"] = p
        star_vals.append(temp.loc[temp["是否数据型巨星"]=="巨星组", f"p_{label}"].mean())
        other_vals.append(temp.loc[temp["是否数据型巨星"]=="其他球员", f"p_{label}"].mean())

    labels = list(dims.keys())
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    s = star_vals + star_vals[:1]
    o = other_vals + other_vals[:1]

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, s, linewidth=2, label="巨星组")
    ax.fill(angles, s, alpha=0.15)
    ax.plot(angles, o, linewidth=2, label="其他球员")
    ax.fill(angles, o, alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0,100)
    ax.set_title("数据型巨星与其他球员特征雷达图", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25,1.1))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig01_star_radar.png", dpi=180, bbox_inches="tight")
    plt.close()

# ---------- 7. PCA ----------
PCA_FEATURES = [
    "场均得分", "真实命中率TS(%)", "场均篮板", "场均助攻",
    "场均抢断", "场均盖帽", "场均时间(分钟)", "助攻失误比"
]

def pca_analysis(df):
    X = df[PCA_FEATURES].copy()
    for c in PCA_FEATURES:
        X[c] = winsorize(X[c])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(Xs)

    plot_df = df[["球员姓名", "是否数据型巨星"]].copy()
    plot_df["PC1"] = pcs[:,0]
    plot_df["PC2"] = pcs[:,1]

    fig, ax = plt.subplots(figsize=(10,7))
    for g in ["其他球员", "巨星组"]:
        sub = plot_df[plot_df["是否数据型巨星"]==g]
        ax.scatter(sub["PC1"], sub["PC2"], label=g, alpha=0.7,
                   s=55 if g=="巨星组" else 25)
    for _, row in plot_df[plot_df["是否数据型巨星"]=="巨星组"].iterrows():
        ax.annotate(row["球员姓名"], (row["PC1"], row["PC2"]), fontsize=8)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("巨星与其他球员PCA二维分布")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig02_star_pca.png", dpi=180, bbox_inches="tight")
    plt.close()

    loadings = pd.DataFrame(
        pca.components_.T,
        index=PCA_FEATURES,
        columns=["PC1","PC2"]
    )
    loadings.to_csv(OUTPUT_DIR / "03_pca_loadings.csv", encoding="utf-8-sig")
    return pca, plot_df

# ---------- 8. KMeans inside stars ----------
CLUSTER_FEATURES = [
    "场均得分", "真实命中率TS(%)", "场均篮板", "场均助攻",
    "场均抢断", "场均盖帽", "场均三分出手数", "助攻失误比"
]

def cluster_stars(df):
    stars = df[df["是否数据型巨星"]=="巨星组"].copy()
    X = stars[CLUSTER_FEATURES].copy()
    for c in CLUSTER_FEATURES:
        X[c] = winsorize(X[c])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    results = []
    max_k = min(5, len(stars)-1)
    for k in range(2, max_k+1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=30)
        labels = km.fit_predict(Xs)
        results.append({
            "K": k,
            "silhouette": silhouette_score(Xs, labels),
            "inertia": km.inertia_
        })

    eval_df = pd.DataFrame(results)
    best_k = int(eval_df.loc[eval_df["silhouette"].idxmax(), "K"])

    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=50)
    labels = km.fit_predict(Xs)
    stars["巨星类型"] = [f"类型{x+1}" for x in labels]

    stars.sort_values(["巨星类型","巨星指数"], ascending=[True,False]).to_csv(
        OUTPUT_DIR / "04_star_clusters.csv", index=False, encoding="utf-8-sig"
    )
    eval_df.to_csv(
        OUTPUT_DIR / "05_cluster_k_evaluation.csv", index=False, encoding="utf-8-sig"
    )

    # K selection plot
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(eval_df["K"], eval_df["silhouette"], marker="o")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette score")
    ax.set_title("巨星内部聚类K值选择")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig03_cluster_k.png", dpi=180, bbox_inches="tight")
    plt.close()

    # cluster center plot
    centers = pd.DataFrame(km.cluster_centers_, columns=CLUSTER_FEATURES)
    fig, ax = plt.subplots(figsize=(12,6))
    x = np.arange(len(CLUSTER_FEATURES))
    width = 0.8 / best_k
    for i in range(best_k):
        offset = (i-(best_k-1)/2)*width
        ax.bar(x+offset, centers.iloc[i], width=width, label=f"类型{i+1}")
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(CLUSTER_FEATURES, rotation=35, ha="right")
    ax.set_ylabel("标准化聚类中心")
    ax.set_title("不同巨星类型特征画像")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig04_cluster_profiles.png", dpi=180, bbox_inches="tight")
    plt.close()

    return eval_df, stars, best_k

# ---------- 9. Main ----------
def main():
    raw = load_players()
    players = prepare(raw)
    players = add_star_index(players)

    ranking = export_ranking(players)
    comparison = compare_groups(players)
    radar_plot(players)
    pca_model, pca_df = pca_analysis(players)
    cluster_eval, star_clusters, best_k = cluster_stars(players)

    players.to_csv(
        OUTPUT_DIR / "06_players_with_star_index.csv",
        index=False, encoding="utf-8-sig"
    )

    # Top30 plot
    top = ranking.head(30).sort_values("巨星指数")
    fig, ax = plt.subplots(figsize=(10,10))
    ax.barh(top["球员姓名"], top["巨星指数"])
    ax.set_xlabel("数据型巨星指数")
    ax.set_title("数据型巨星综合指数 Top30")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig05_star_top30.png", dpi=180, bbox_inches="tight")
    plt.close()

    print("\nTop 10:")
    print(ranking.head(10)[[
        "综合排名","球员姓名","位置大类","场均得分",
        "真实命中率TS(%)","场均篮板","场均助攻",
        "巨星指数","同位置表现指数"
    ]].to_string(index=False))

    print("\nMost distinctive features:")
    print(comparison.head(10)[[
        "指标","巨星组均值","其他球员均值","Cohen_d"
    ]].to_string(index=False))

    print("\nBest K for star clustering:", best_k)
    print("Output folder:", OUTPUT_DIR)

    return {
        "ranking": ranking,
        "comparison": comparison,
        "cluster_eval": cluster_eval,
        "star_clusters": star_clusters,
        "players": players
    }

if __name__ == "__main__":
    results = main()
