# -*- coding: utf-8 -*-
"""
NBA 2025-2026 数据挖掘项目（承接数据分析部分）
=================================================
项目目标：
1. 球队无监督聚类：识别不同球队技术风格
2. 球队 PCA：提取主要竞争特征并二维可视化
3. 球员无监督聚类：识别不同球员角色/风格
4. 球员 PCA：观察球员角色空间结构
5. 球队胜率回归：用纯技术统计预测胜率
6. Top10 强队分类：用纯技术统计识别强队
7. 自动输出结果表、图像和文字摘要

重要原则：
- 预测模型严禁使用 联盟排名、胜场、负场、胜场差、胜率、场均净胜分、
  主/客场战绩、最近10场 等直接或近直接包含赛季结果的信息作为自变量，
  防止“目标泄露（data leakage）”。
- 球队样本只有 30 支，因此回归和分类使用重复交叉验证，结果只作为探索性结论，
  不把高分简单解释为真实泛化能力。
- 聚类前对变量标准化，球员数据额外进行分位数截尾，降低极端值对 K-Means 的影响。

推荐运行方式：
1. 先运行上一份“NBA_2025-2026_数据分析项目_完整源代码.py”；
2. 将本文件与“NBA数据分析输出”文件夹放在同一目录；
3. 在 Anaconda / Spyder 中直接运行本文件。

如果找不到清洗后 CSV，本程序也会尝试从原始 Excel 中读取并做最小必要预处理。

依赖：
pandas, numpy, matplotlib, scikit-learn, openpyxl
Anaconda 一般已包含。
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    silhouette_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
    KFold,
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
)
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier


# =========================================================
# 0. 全局设置
# =========================================================
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)

RANDOM_STATE = 42
OUTPUT_DIR_NAME = "NBA数据挖掘输出"
ANALYSIS_OUTPUT_DIR_NAME = "NBA数据分析输出"


def setup_chinese_font():
    """自动选择常见中文字体，避免图像中文乱码。"""
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "Arial Unicode MS",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in installed:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def get_base_dir():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def safe_divide(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return np.where(b != 0, a / b, np.nan)


def parse_record(series):
    """把 30-9 形式的战绩拆为胜、负、胜率。"""
    temp = series.astype(str).str.extract(r"(\d+)\s*-\s*(\d+)")
    w = pd.to_numeric(temp[0], errors="coerce")
    l = pd.to_numeric(temp[1], errors="coerce")
    pct = safe_divide(w, w + l)
    return w, l, pct


def load_cleaned_or_raw(base_dir):
    """
    优先读取数据分析阶段生成的清洗后数据；
    若不存在，则读取原始 Excel 并完成数据挖掘所需的最小预处理。
    """
    analysis_dir = base_dir / ANALYSIS_OUTPUT_DIR_NAME
    team_clean = analysis_dir / "清洗后_球队综合数据.csv"
    player_clean = analysis_dir / "清洗后_球员数据.csv"

    if team_clean.exists() and player_clean.exists():
        print("[读取] 使用数据分析阶段生成的清洗后 CSV。")
        team = pd.read_csv(team_clean, encoding="utf-8-sig")
        player = pd.read_csv(player_clean, encoding="utf-8-sig")
        return team, player

    print("[提示] 未找到清洗后 CSV，将从原始 Excel 进行最小必要预处理。")
    required_sheets = {"球员数据", "排名数据", "球队数据"}
    excel_file = None
    preferred = [
        "NBA_2025-2026_三表中文版(1).xlsx",
        "NBA_2025-2026_三表中文版.xlsx",
    ]
    for name in preferred:
        p = base_dir / name
        if p.exists():
            excel_file = p
            break
    if excel_file is None:
        for p in base_dir.glob("*.xlsx"):
            try:
                if required_sheets.issubset(set(pd.ExcelFile(p).sheet_names)):
                    excel_file = p
                    break
            except Exception:
                pass
    if excel_file is None:
        raise FileNotFoundError(
            "未找到清洗后数据，也未找到三表中文版 Excel。\n"
            "请先运行数据分析代码，或把 Excel 与本脚本放在同一目录。"
        )

    player = pd.read_excel(excel_file, sheet_name="球员数据")
    standing = pd.read_excel(excel_file, sheet_name="排名数据")
    team_stats = pd.read_excel(excel_file, sheet_name="球队数据")

    # 统一列名：下面兼容当前中文版三表字段。
    # 排名表 + 球队表合并
    team = standing.merge(team_stats, on="球队", how="inner", suffixes=("", "_球队表"))

    # 如果原始排名表还没有衍生字段，则构造。
    if "主场胜率" not in team.columns and "主场战绩" in team.columns:
        team["主场胜场"], team["主场负场"], team["主场胜率"] = parse_record(team["主场战绩"])
    if "客场胜率" not in team.columns and "客场战绩" in team.columns:
        team["客场胜场"], team["客场负场"], team["客场胜率"] = parse_record(team["客场战绩"])

    # 球队特征工程
    if "三分出手占比" not in team.columns:
        team["三分出手占比"] = safe_divide(team["场均三分出手数"], team["场均投篮出手数"])
    if "助攻失误比" not in team.columns:
        team["助攻失误比"] = safe_divide(team["场均助攻"], team["场均失误"])
    if "前场篮板占比" not in team.columns:
        team["前场篮板占比"] = safe_divide(team["场均前场篮板"], team["场均篮板"])
    if "罚球出手率" not in team.columns:
        team["罚球出手率"] = safe_divide(team["场均罚球出手数"], team["场均投篮出手数"])

    # 球员特征工程
    if "原始位置" not in player.columns:
        player["原始位置"] = player["位置"].astype(str)
    if "位置大类" not in player.columns:
        position_map = {
            "后卫": "后卫", "前锋": "前锋", "中锋": "中锋", "大前锋": "前锋",
            "G": "后卫", "F": "前锋", "C": "中锋", "PF": "前锋",
        }
        player["位置大类"] = player["位置"].astype(str).map(position_map).fillna(player["位置"].astype(str))

    minute = pd.to_numeric(player["场均时间(分钟)"], errors="coerce")
    for src, dst in [
        ("场均得分", "每36分钟得分"),
        ("场均篮板", "每36分钟篮板"),
        ("场均助攻", "每36分钟助攻"),
        ("场均抢断", "每36分钟抢断"),
        ("场均盖帽", "每36分钟盖帽"),
        ("场均失误", "每36分钟失误"),
    ]:
        if dst not in player.columns:
            player[dst] = pd.to_numeric(player[src], errors="coerce") * 36 / minute.replace(0, np.nan)

    if "有效投篮命中率eFG(%)" not in player.columns:
        player["有效投篮命中率eFG(%)"] = safe_divide(
            player["场均投篮命中数"] + 0.5 * player["场均三分命中数"],
            player["场均投篮出手数"],
        ) * 100
    if "真实命中率TS(%)" not in player.columns:
        player["真实命中率TS(%)"] = safe_divide(
            player["场均得分"],
            2 * (player["场均投篮出手数"] + 0.44 * player["场均罚球出手数"]),
        ) * 100
    if "三分出手占比" not in player.columns:
        player["三分出手占比"] = safe_divide(player["场均三分出手数"], player["场均投篮出手数"])
    if "助攻失误比" not in player.columns:
        player["助攻失误比"] = safe_divide(player["场均助攻"], player["场均失误"])

    return team, player


def ensure_numeric(df, columns):
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def winsorize_frame(df, cols, lower=0.01, upper=0.99):
    """按列分位数截尾，降低极端值对 K-Means 的影响。"""
    out = df.copy()
    for c in cols:
        lo = out[c].quantile(lower)
        hi = out[c].quantile(upper)
        out[c] = out[c].clip(lo, hi)
    return out


def kmeans_select_k(X_scaled, k_values):
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        rows.append({"K值": k, "轮廓系数": score, "簇内平方和Inertia": model.inertia_})
    result = pd.DataFrame(rows)
    best_k = int(result.loc[result["轮廓系数"].idxmax(), "K值"])
    return result, best_k


def make_cluster_profile(z_df, labels, feature_names):
    """生成每个聚类相对总体均值的高低画像（基于标准化均值）。"""
    temp = z_df.copy()
    temp["聚类"] = labels
    centers_z = temp.groupby("聚类")[feature_names].mean()
    profiles = []
    for cluster_id, row in centers_z.iterrows():
        s = row.sort_values(ascending=False)
        high = "、".join([f"{idx}({val:+.2f}σ)" for idx, val in s.head(3).items()])
        low_s = row.sort_values(ascending=True)
        low = "、".join([f"{idx}({val:+.2f}σ)" for idx, val in low_s.head(3).items()])
        profiles.append({"聚类": cluster_id, "相对突出指标": high, "相对偏低指标": low})
    return pd.DataFrame(profiles), centers_z


# =========================================================
# 1. 球队聚类 + PCA
# =========================================================
def team_clustering(team, output_dir):
    print("\n========== 1. 球队聚类与 PCA ==========")

    features = [
        "场均得分",
        "投篮命中率(%)",
        "三分命中率(%)",
        "场均篮板",
        "场均助攻",
        "场均抢断",
        "场均盖帽",
        "场均失误",
        "三分出手占比",
        "助攻失误比",
    ]
    team = ensure_numeric(team, features + ["胜率", "联盟排名"])
    use = team[["球队", "分区", "联盟排名", "胜率"] + features].dropna().copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(use[features])

    k_eval, best_k = kmeans_select_k(X_scaled, range(2, 7))
    save_csv(k_eval, output_dir / "01_球队聚类_K值评估.csv")
    print(f"球队聚类自动选择 K = {best_k}，依据为轮廓系数最大。")

    model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=100)
    labels = model.fit_predict(X_scaled)
    use["球队聚类"] = labels + 1

    # PCA 二维投影
    pca = PCA(n_components=min(len(features), len(use)))
    pcs = pca.fit_transform(X_scaled)
    use["PC1"] = pcs[:, 0]
    use["PC2"] = pcs[:, 1]

    save_csv(use.sort_values(["球队聚类", "联盟排名"]), output_dir / "02_球队聚类结果.csv")

    centers_raw = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=features,
    )
    centers_raw.insert(0, "球队聚类", np.arange(1, best_k + 1))
    save_csv(centers_raw, output_dir / "03_球队聚类中心_原始尺度.csv")

    z_df = pd.DataFrame(X_scaled, columns=features, index=use.index)
    profile, centers_z = make_cluster_profile(z_df, labels + 1, features)
    counts = use["球队聚类"].value_counts().sort_index().rename("球队数")
    avg_result = use.groupby("球队聚类")[["胜率", "联盟排名"]].mean().rename(
        columns={"胜率": "平均胜率", "联盟排名": "平均联盟排名"}
    )
    profile = profile.merge(counts, left_on="聚类", right_index=True).merge(avg_result, left_on="聚类", right_index=True)
    save_csv(profile, output_dir / "04_球队聚类画像.csv")

    # PCA 解释率、载荷
    pca_info = pd.DataFrame({
        "主成分": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "方差解释率": pca.explained_variance_ratio_,
        "累计解释率": np.cumsum(pca.explained_variance_ratio_),
    })
    save_csv(pca_info, output_dir / "05_球队PCA_解释方差.csv")

    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    ).reset_index().rename(columns={"index": "指标"})
    save_csv(loadings, output_dir / "06_球队PCA_载荷矩阵.csv")

    # 图1：K 选择
    plt.figure(figsize=(7, 5))
    plt.plot(k_eval["K值"], k_eval["轮廓系数"], marker="o")
    plt.axvline(best_k, linestyle="--", alpha=0.6)
    plt.xlabel("聚类数 K")
    plt.ylabel("轮廓系数")
    plt.title("球队 K-Means 聚类数选择")
    plt.grid(alpha=0.2)
    save_figure(output_dir / "图01_球队聚类_K值选择.png")

    # 图2：PCA 聚类散点图
    plt.figure(figsize=(10, 7))
    for cid in sorted(use["球队聚类"].unique()):
        part = use[use["球队聚类"] == cid]
        plt.scatter(part["PC1"], part["PC2"], s=75, label=f"聚类{cid}", alpha=0.8)
    for _, row in use.iterrows():
        plt.annotate(row["球队"], (row["PC1"], row["PC2"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("球队技术风格聚类：PCA 二维投影")
    plt.legend()
    plt.grid(alpha=0.2)
    save_figure(output_dir / "图02_球队聚类_PCA二维图.png")

    # 图3：标准化聚类中心画像
    center_plot = centers_z.copy()
    center_plot.index = [f"聚类{i}" for i in center_plot.index]
    plt.figure(figsize=(12, 5 + 0.5 * best_k))
    im = plt.imshow(center_plot.values, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    plt.colorbar(im, label="标准化均值（Z-score）")
    plt.xticks(range(len(features)), features, rotation=45, ha="right")
    plt.yticks(range(best_k), center_plot.index)
    plt.title("球队聚类中心画像（相对联盟平均水平）")
    for i in range(best_k):
        for j in range(len(features)):
            plt.text(j, i, f"{center_plot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    save_figure(output_dir / "图03_球队聚类中心画像.png")

    return {
        "best_k": best_k,
        "silhouette": float(k_eval.loc[k_eval["K值"] == best_k, "轮廓系数"].iloc[0]),
        "pca2": float(pca.explained_variance_ratio_[:2].sum()),
        "features": features,
        "result": use,
        "profile": profile,
    }


# =========================================================
# 2. 球员聚类 + PCA
# =========================================================
def player_clustering(player, output_dir):
    print("\n========== 2. 球员聚类与 PCA ==========")

    # 过滤极低样本球员，避免少量出场导致每36分钟数据被放大。
    min_gp = 20
    min_min = 10
    features = [
        "每36分钟得分",
        "每36分钟篮板",
        "每36分钟助攻",
        "每36分钟抢断",
        "每36分钟盖帽",
        "每36分钟失误",
        "真实命中率TS(%)",
        "三分出手占比",
        "助攻失误比",
    ]

    player = ensure_numeric(player, features + ["出场次数", "场均时间(分钟)", "场均得分", "场均篮板", "场均助攻"])
    base_cols = ["球员姓名", "位置大类", "出场次数", "场均时间(分钟)", "场均得分", "场均篮板", "场均助攻"]
    use = player.loc[
        (player["出场次数"] >= min_gp) & (player["场均时间(分钟)"] >= min_min),
        base_cols + features,
    ].dropna().copy()

    # 1%-99% 截尾减少极端比例指标的影响
    work = winsorize_frame(use, features, 0.01, 0.99)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(work[features])

    k_eval, best_k = kmeans_select_k(X_scaled, range(3, 7))
    save_csv(k_eval, output_dir / "07_球员聚类_K值评估.csv")
    print(f"纳入球员 {len(use)} 人；自动选择 K = {best_k}。")

    model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=100)
    labels = model.fit_predict(X_scaled)
    use["球员聚类"] = labels + 1

    pca = PCA(n_components=min(len(features), len(use)))
    pcs = pca.fit_transform(X_scaled)
    use["PC1"] = pcs[:, 0]
    use["PC2"] = pcs[:, 1]
    save_csv(use.sort_values(["球员聚类", "场均得分"], ascending=[True, False]), output_dir / "08_球员聚类结果.csv")

    centers_raw = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=features)
    centers_raw.insert(0, "球员聚类", np.arange(1, best_k + 1))
    save_csv(centers_raw, output_dir / "09_球员聚类中心_原始尺度.csv")

    z_df = pd.DataFrame(X_scaled, columns=features, index=use.index)
    profile, centers_z = make_cluster_profile(z_df, labels + 1, features)
    counts = use["球员聚类"].value_counts().sort_index().rename("球员数")
    profile = profile.merge(counts, left_on="聚类", right_index=True)
    save_csv(profile, output_dir / "10_球员聚类画像.csv")

    # 位置构成：帮助解释聚类角色，但不参与聚类本身。
    pos_dist = pd.crosstab(use["球员聚类"], use["位置大类"], normalize="index") * 100
    pos_dist = pos_dist.round(2).reset_index()
    save_csv(pos_dist, output_dir / "11_各球员聚类_位置构成百分比.csv")

    pca_info = pd.DataFrame({
        "主成分": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "方差解释率": pca.explained_variance_ratio_,
        "累计解释率": np.cumsum(pca.explained_variance_ratio_),
    })
    save_csv(pca_info, output_dir / "12_球员PCA_解释方差.csv")
    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    ).reset_index().rename(columns={"index": "指标"})
    save_csv(loadings, output_dir / "13_球员PCA_载荷矩阵.csv")

    # K 选择图
    plt.figure(figsize=(7, 5))
    plt.plot(k_eval["K值"], k_eval["轮廓系数"], marker="o")
    plt.axvline(best_k, linestyle="--", alpha=0.6)
    plt.xlabel("聚类数 K")
    plt.ylabel("轮廓系数")
    plt.title("球员 K-Means 聚类数选择")
    plt.grid(alpha=0.2)
    save_figure(output_dir / "图04_球员聚类_K值选择.png")

    # PCA 聚类图
    plt.figure(figsize=(10, 7))
    for cid in sorted(use["球员聚类"].unique()):
        part = use[use["球员聚类"] == cid]
        plt.scatter(part["PC1"], part["PC2"], s=35, label=f"聚类{cid}", alpha=0.7)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("球员角色聚类：PCA 二维投影")
    plt.legend()
    plt.grid(alpha=0.2)
    save_figure(output_dir / "图05_球员聚类_PCA二维图.png")

    # 聚类中心画像
    center_plot = centers_z.copy()
    center_plot.index = [f"聚类{i}" for i in center_plot.index]
    plt.figure(figsize=(12, 5 + 0.5 * best_k))
    im = plt.imshow(center_plot.values, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    plt.colorbar(im, label="标准化均值（Z-score）")
    plt.xticks(range(len(features)), features, rotation=45, ha="right")
    plt.yticks(range(best_k), center_plot.index)
    plt.title("球员聚类中心画像")
    for i in range(best_k):
        for j in range(len(features)):
            plt.text(j, i, f"{center_plot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    save_figure(output_dir / "图06_球员聚类中心画像.png")

    return {
        "n_players": len(use),
        "best_k": best_k,
        "silhouette": float(k_eval.loc[k_eval["K值"] == best_k, "轮廓系数"].iloc[0]),
        "pca2": float(pca.explained_variance_ratio_[:2].sum()),
        "result": use,
        "profile": profile,
    }


# =========================================================
# 3. 球队胜率回归
# =========================================================
def team_winrate_regression(team, output_dir):
    print("\n========== 3. 球队胜率回归 ==========")

    features = [
        "场均得分",
        "投篮命中率(%)",
        "三分命中率(%)",
        "罚球命中率(%)",
        "场均篮板",
        "场均助攻",
        "场均抢断",
        "场均盖帽",
        "场均失误",
        "三分出手占比",
        "助攻失误比",
        "罚球出手率",
    ]
    target = "胜率"
    team = ensure_numeric(team, features + [target])
    use = team[["球队", "联盟排名", target] + features].dropna().copy()
    X = use[features]
    y = use[target]

    # 5 折交叉验证：在 30 支球队的小样本条件下评估模型。
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    alphas = np.logspace(-3, 3, 60)

    models = {
        "均值基线": DummyRegressor(strategy="mean"),
        "线性回归": Pipeline([
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "岭回归": Pipeline([
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=alphas)),
        ]),
        "随机森林回归": RandomForestRegressor(
            n_estimators=120,
            max_depth=4,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
        "梯度提升回归": GradientBoostingRegressor(
            n_estimators=80,
            learning_rate=0.03,
            max_depth=2,
            random_state=RANDOM_STATE,
            loss="huber",
        ),
    }

    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }
    rows = []
    for name, model in models.items():
        score = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        rows.append({
            "模型": name,
            "CV_MAE均值": -score["test_MAE"].mean(),
            "CV_MAE标准差": score["test_MAE"].std(),
            "CV_RMSE均值": -score["test_RMSE"].mean(),
            "CV_RMSE标准差": score["test_RMSE"].std(),
            "CV_R2均值": score["test_R2"].mean(),
            "CV_R2标准差": score["test_R2"].std(),
        })
    compare = pd.DataFrame(rows).sort_values("CV_RMSE均值")
    save_csv(compare, output_dir / "14_胜率回归_模型交叉验证比较.csv")

    # 岭回归作为主解释模型：在全样本拟合后查看标准化系数。
    ridge_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", RidgeCV(alphas=alphas)),
    ])
    ridge_pipe.fit(X, y)
    ridge = ridge_pipe.named_steps["model"]
    coef = pd.DataFrame({
        "指标": features,
        "标准化岭回归系数": ridge.coef_,
        "系数绝对值": np.abs(ridge.coef_),
    }).sort_values("系数绝对值", ascending=False)
    coef["方向"] = np.where(coef["标准化岭回归系数"] >= 0, "正向", "负向")
    save_csv(coef, output_dir / "15_胜率回归_岭回归标准化系数.csv")

    # 5折 OOF 预测用于直观展示（每支球队仅被预测一次）
    cv_pred = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pred = cross_val_predict(ridge_pipe, X, y, cv=cv_pred, n_jobs=1)
    pred_df = use[["球队", "联盟排名", target]].copy()
    pred_df["交叉验证预测胜率"] = pred
    pred_df["预测误差"] = pred_df["交叉验证预测胜率"] - pred_df[target]
    pred_df["绝对误差"] = pred_df["预测误差"].abs()
    save_csv(pred_df.sort_values("绝对误差", ascending=False), output_dir / "16_胜率回归_岭回归OOF预测.csv")

    oof_mae = mean_absolute_error(y, pred)
    oof_rmse = np.sqrt(mean_squared_error(y, pred))
    oof_r2 = r2_score(y, pred)

    # 图：模型比较
    plt.figure(figsize=(9, 5))
    order = compare.sort_values("CV_RMSE均值")
    plt.barh(order["模型"], order["CV_RMSE均值"], xerr=order["CV_RMSE标准差"], alpha=0.8)
    plt.xlabel("5折交叉验证 RMSE（越低越好）")
    plt.title("球队胜率回归模型比较")
    save_figure(output_dir / "图07_胜率回归模型比较.png")

    # 图：实际 vs 预测
    plt.figure(figsize=(7, 6))
    plt.scatter(y, pred, s=70, alpha=0.8)
    lo = min(y.min(), pred.min())
    hi = max(y.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    for i, row in use.reset_index(drop=True).iterrows():
        plt.annotate(row["球队"], (y.iloc[i], pred[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.xlabel("真实胜率")
    plt.ylabel("交叉验证预测胜率")
    plt.title(f"岭回归：真实胜率 vs 预测胜率\nMAE={oof_mae:.3f}, RMSE={oof_rmse:.3f}, R²={oof_r2:.3f}")
    plt.grid(alpha=0.2)
    save_figure(output_dir / "图08_岭回归_真实胜率与预测胜率.png")

    # 图：系数
    plot_coef = coef.sort_values("标准化岭回归系数")
    plt.figure(figsize=(9, 6))
    plt.barh(plot_coef["指标"], plot_coef["标准化岭回归系数"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("标准化岭回归系数")
    plt.title("球队技术指标与胜率的岭回归系数（探索性）")
    save_figure(output_dir / "图09_胜率回归_岭回归系数.png")

    return {
        "compare": compare,
        "alpha": float(ridge.alpha_),
        "oof_mae": float(oof_mae),
        "oof_rmse": float(oof_rmse),
        "oof_r2": float(oof_r2),
        "coef": coef,
    }


# =========================================================
# 4. Top10 强队分类
# =========================================================
def top10_classification(team, output_dir):
    print("\n========== 4. Top10 强队分类 ==========")

    features = [
        "场均得分",
        "投篮命中率(%)",
        "三分命中率(%)",
        "罚球命中率(%)",
        "场均篮板",
        "场均助攻",
        "场均抢断",
        "场均盖帽",
        "场均失误",
        "三分出手占比",
        "助攻失误比",
        "罚球出手率",
    ]
    team = ensure_numeric(team, features + ["联盟排名", "胜率"])
    use = team[["球队", "联盟排名", "胜率"] + features].dropna().copy()
    use["是否Top10"] = (use["联盟排名"] <= 10).astype(int)
    X = use[features]
    y = use["是否Top10"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = {
        "多数类基线": DummyClassifier(strategy="most_frequent"),
        "逻辑回归": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        "决策树": DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "随机森林分类": RandomForestClassifier(
            n_estimators=120,
            max_depth=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    scoring = {
        "Accuracy": "accuracy",
        "BalancedAccuracy": "balanced_accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1": "f1",
        "ROC_AUC": "roc_auc",
    }
    rows = []
    for name, model in models.items():
        score = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        row = {"模型": name}
        for metric in scoring:
            row[f"CV_{metric}均值"] = score[f"test_{metric}"].mean()
            row[f"CV_{metric}标准差"] = score[f"test_{metric}"].std()
        rows.append(row)
    compare = pd.DataFrame(rows).sort_values("CV_F1均值", ascending=False)
    save_csv(compare, output_dir / "17_Top10分类_模型交叉验证比较.csv")

    # 逻辑回归为解释模型
    logit_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=RANDOM_STATE)),
    ])
    logit_pipe.fit(X, y)
    logit = logit_pipe.named_steps["model"]
    coef = pd.DataFrame({
        "指标": features,
        "逻辑回归标准化系数": logit.coef_[0],
        "系数绝对值": np.abs(logit.coef_[0]),
    }).sort_values("系数绝对值", ascending=False)
    coef["方向"] = np.where(coef["逻辑回归标准化系数"] >= 0, "更倾向Top10", "更倾向非Top10")
    save_csv(coef, output_dir / "18_Top10分类_逻辑回归标准化系数.csv")

    # 5折 OOF 概率，用于 ROC 和混淆矩阵
    cv_pred = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(logit_pipe, X, y, cv=cv_pred, method="predict_proba", n_jobs=1)[:, 1]
    pred = (proba >= 0.5).astype(int)
    pred_df = use[["球队", "联盟排名", "胜率", "是否Top10"]].copy()
    pred_df["Top10预测概率"] = proba
    pred_df["预测类别"] = pred
    save_csv(pred_df.sort_values("Top10预测概率", ascending=False), output_dir / "19_Top10分类_逻辑回归OOF预测.csv")

    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "BalancedAccuracy": balanced_accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y, proba),
    }
    metrics_df = pd.DataFrame([metrics])
    save_csv(metrics_df, output_dir / "20_Top10分类_逻辑回归OOF指标.csv")

    # ROC
    fpr, tpr, _ = roc_curve(y, proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"逻辑回归 AUC={metrics['ROC_AUC']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.6)
    plt.xlabel("假阳性率 FPR")
    plt.ylabel("真正率 TPR")
    plt.title("Top10 强队分类 ROC 曲线（5折OOF）")
    plt.legend()
    plt.grid(alpha=0.2)
    save_figure(output_dir / "图10_Top10分类_ROC曲线.png")

    # 混淆矩阵
    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(5, 4))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xticks([0, 1], ["预测非Top10", "预测Top10"])
    plt.yticks([0, 1], ["实际非Top10", "实际Top10"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=13)
    plt.title("Top10 分类混淆矩阵（5折OOF）")
    save_figure(output_dir / "图11_Top10分类_混淆矩阵.png")

    # 系数图
    plot_coef = coef.sort_values("逻辑回归标准化系数")
    plt.figure(figsize=(9, 6))
    plt.barh(plot_coef["指标"], plot_coef["逻辑回归标准化系数"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("标准化逻辑回归系数")
    plt.title("识别 Top10 球队的重要技术指标（探索性）")
    save_figure(output_dir / "图12_Top10分类_逻辑回归系数.png")

    return {
        "compare": compare,
        "metrics": metrics,
        "coef": coef,
    }


# =========================================================
# 5. 自动摘要
# =========================================================
def make_summary(output_dir, team_result, player_result, reg_result, cls_result):
    lines = []
    lines.append("NBA 2025-2026 数据挖掘项目：自动结果摘要")
    lines.append("=" * 60)
    lines.append("")
    lines.append("一、方法说明")
    lines.append("1. 球队和球员使用 K-Means 聚类，聚类数 K 由轮廓系数自动选择。")
    lines.append("2. 使用 PCA 提取主要变化方向并进行二维可视化。")
    lines.append("3. 胜率预测只使用球队技术统计，不使用排名、胜负场、净胜分等结果型变量，避免目标泄露。")
    lines.append("4. 球队仅 30 个样本，回归和分类采用 5 折交叉验证；所有预测结论均应视为探索性结果。")
    lines.append("")

    lines.append("二、球队聚类")
    lines.append(f"最佳 K = {team_result['best_k']}，轮廓系数 = {team_result['silhouette']:.3f}。")
    lines.append(f"前两个主成分累计解释 {team_result['pca2']*100:.1f}% 的标准化技术统计变异。")
    for _, r in team_result["profile"].iterrows():
        lines.append(
            f"聚类{int(r['聚类'])}：{int(r['球队数'])}支球队，平均胜率{r['平均胜率']:.3f}；"
            f"突出：{r['相对突出指标']}；偏低：{r['相对偏低指标']}。"
        )
    lines.append("")

    lines.append("三、球员聚类")
    lines.append(
        f"筛选出出场≥20场且场均时间≥10分钟的 {player_result['n_players']} 名球员。"
    )
    lines.append(f"最佳 K = {player_result['best_k']}，轮廓系数 = {player_result['silhouette']:.3f}。")
    lines.append(f"前两个主成分累计解释 {player_result['pca2']*100:.1f}% 的球员特征变异。")
    for _, r in player_result["profile"].iterrows():
        lines.append(
            f"聚类{int(r['聚类'])}：{int(r['球员数'])}人；突出：{r['相对突出指标']}；偏低：{r['相对偏低指标']}。"
        )
    lines.append("")

    lines.append("四、球队胜率回归")
    best_reg = reg_result["compare"].iloc[0]
    lines.append(
        f"按5折交叉验证 RMSE，当前表现最好的模型是“{best_reg['模型']}”："
        f"RMSE={best_reg['CV_RMSE均值']:.3f}，MAE={best_reg['CV_MAE均值']:.3f}，R²={best_reg['CV_R2均值']:.3f}。"
    )
    lines.append(
        f"岭回归 5 折 OOF：MAE={reg_result['oof_mae']:.3f}，RMSE={reg_result['oof_rmse']:.3f}，R²={reg_result['oof_r2']:.3f}；"
        f"全样本内部选择的 alpha={reg_result['alpha']:.4g}。"
    )
    top_coef = reg_result["coef"].head(5)
    lines.append("岭回归中绝对系数较大的指标：" + "、".join(top_coef["指标"].tolist()) + "。")
    lines.append("注意：系数反映控制其他变量后的统计关系，不等同于因果影响。")
    lines.append("")

    lines.append("五、Top10 强队分类")
    best_cls = cls_result["compare"].iloc[0]
    lines.append(
        f"按5折交叉验证 F1，当前表现最好的分类模型是“{best_cls['模型']}”："
        f"F1={best_cls['CV_F1均值']:.3f}，ROC-AUC={best_cls['CV_ROC_AUC均值']:.3f}。"
    )
    m = cls_result["metrics"]
    lines.append(
        f"逻辑回归 5 折 OOF：Accuracy={m['Accuracy']:.3f}，BalancedAccuracy={m['BalancedAccuracy']:.3f}，"
        f"Precision={m['Precision']:.3f}，Recall={m['Recall']:.3f}，F1={m['F1']:.3f}，ROC-AUC={m['ROC_AUC']:.3f}。"
    )
    top_cls_coef = cls_result["coef"].head(5)
    lines.append("逻辑回归中绝对系数较大的指标：" + "、".join(top_cls_coef["指标"].tolist()) + "。")
    lines.append("")

    lines.append("六、解释边界")
    lines.append("1. 本项目是单赛季、30支球队的横截面数据，不适合据此宣称稳定的长期因果规律。")
    lines.append("2. 聚类名称应依据聚类中心画像人工命名，例如“高组织进攻型”“内线防守型”等，不应只凭单一指标命名。")
    lines.append("3. 如果后续补充多个赛季、比赛级数据或球员所属球队字段，可显著提升预测模型与球员—球队联合分析的可信度。")

    with open(output_dir / "21_数据挖掘自动结果摘要.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================================================
# 6. 主程序
# =========================================================
def main():
    setup_chinese_font()
    base_dir = get_base_dir()
    output_dir = base_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NBA 2025-2026 数据挖掘项目")
    print("=" * 70)
    print(f"工作目录：{base_dir}")
    print(f"输出目录：{output_dir}")

    team, player = load_cleaned_or_raw(base_dir)
    print(f"球队样本数：{len(team)}")
    print(f"球员样本数：{len(player)}")

    team_result = team_clustering(team, output_dir)
    player_result = player_clustering(player, output_dir)
    reg_result = team_winrate_regression(team, output_dir)
    cls_result = top10_classification(team, output_dir)
    make_summary(output_dir, team_result, player_result, reg_result, cls_result)

    print("\n" + "=" * 70)
    print("数据挖掘完成。")
    print(f"结果已保存到：{output_dir}")
    print("建议优先查看：")
    print("  04_球队聚类画像.csv")
    print("  10_球员聚类画像.csv")
    print("  14_胜率回归_模型交叉验证比较.csv")
    print("  17_Top10分类_模型交叉验证比较.csv")
    print("  21_数据挖掘自动结果摘要.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
