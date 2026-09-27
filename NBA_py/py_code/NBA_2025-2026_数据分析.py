# -*- coding: utf-8 -*-
"""
NBA 2025-2026 数据分析项目（数据分析部分）

功能：
1. 自动读取三表中文版 Excel
2. 数据质量检查与预处理
3. 球队排名、胜率、净胜分分析
4. 强/中/弱球队比较
5. 球队技术统计与胜率相关性分析
6. 进攻、防守、三分、失误、篮板等专项分析
7. 主客场与东西部分区分析
8. 球员基础表现、位置差异、效率指标分析
9. 自动输出清洗后的数据、统计表、图片和文字摘要

运行环境：
Anaconda / Spyder / Jupyter 

依赖：
pandas, numpy, matplotlib, openpyxl
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


#============
# 0.全局设置
#============
warnings.filterwarnings("ignore", category=FutureWarning)  #忽略 FutureWarning 类型的警告
pd.set_option("display.max_columns", 100)  #设置输出规格
pd.set_option("display.width", 160)

OUTPUT_NAME = "NBA数据分析输出"

    #设置字体，涵盖win mac lin
def setup_chinese_font():
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
    #设置负号显示
    plt.rcParams["axes.unicode_minus"] = False


def find_data_file():
    """
    自动寻找包含“球员数据、排名数据、球队数据”三个工作表的 Excel
    优先查找脚本所在目录，其次查找当前工作目录
    增强解耦性与容错性，比起1直接read更稳妥
    """
    required_sheets = {"球员数据", "排名数据", "球队数据"}

    search_dirs = []
    try:
        search_dirs.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    search_dirs.append(Path.cwd())

    # 去重
    unique_dirs = []
    for d in search_dirs:
        if d not in unique_dirs:
            unique_dirs.append(d)

    preferred_names = [
        "NBA_2025-2026_三表中文版(1).xlsx",
        "NBA_2025-2026_三表中文版.xlsx",
    ]

    # 先找优先文件名
    for d in unique_dirs:
        for name in preferred_names:
            p = d / name
            if p.exists():
                return p

    # 再扫描 Excel
    for d in unique_dirs:
        for p in d.glob("*.xlsx"):
            try:
                sheets = set(pd.ExcelFile(p).sheet_names)
                if required_sheets.issubset(sheets):
                    return p
            except Exception:
                continue

    raise FileNotFoundError(  #未找到报错
        "未找到包含【球员数据、排名数据、球队数据】三个工作表的 Excel 文件。\n"
        "请把本代码与 NBA_2025-2026_三表中文版(1).xlsx 放在同一文件夹后重新运行。"
    )


def safe_divide(a, b):   #安全除法：分母为 0 时返回 NaN
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return np.where(b != 0, a / b, np.nan)


def parse_record(series):    #拆分胜场、负场、计算胜率
    s = series.astype(str).str.strip()
    parts = s.str.extract(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
    wins = pd.to_numeric(parts[0], errors="coerce")
    losses = pd.to_numeric(parts[1], errors="coerce")
    total = wins + losses
    rate = np.where(total > 0, wins / total, np.nan)
    return pd.DataFrame({"胜": wins, "负": losses, "胜率": rate}, index=series.index)


def parse_streak(series):    #连胜连败字段变量化
    s = series.astype(str).str.strip()
    n = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")
    sign = np.where(s.str.contains("胜", na=False), 1,
                    np.where(s.str.contains("败", na=False), -1, np.nan))
    return n * sign


def save_csv(df, output_dir, filename, index=False):
    df.to_csv(output_dir / filename, index=index, encoding="utf-8-sig")


def finish_plot(output_dir, filename):
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close()


def plot_heatmap(corr_df, title, output_path, decimals=2):    #绘制相关系数热力图
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(corr_df.values, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_df.index)
    ax.set_title(title)

    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            value = corr_df.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.{decimals}f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


# ============
# 1. 读取数据
# ============
def load_data(excel_path):
    players = pd.read_excel(excel_path, sheet_name="球员数据")
    standings = pd.read_excel(excel_path, sheet_name="排名数据")
    teams = pd.read_excel(excel_path, sheet_name="球队数据")

    # 去除列名前后空格
    players.columns = players.columns.astype(str).str.strip()
    standings.columns = standings.columns.astype(str).str.strip()
    teams.columns = teams.columns.astype(str).str.strip()

    return players, standings, teams


# ==============
# 2. 数据质量检查
# ==============
def data_quality_check(players, standings, teams, output_dir):
    rows = []

    for name, df in [
        ("球员数据", players),
        ("排名数据", standings),
        ("球队数据", teams),
    ]:
        rows.append({
            "数据表": name,
            "行数": df.shape[0],
            "列数": df.shape[1],
            "缺失值总数": int(df.isna().sum().sum()),
            "整行重复数": int(df.duplicated().sum()),
        })

    quality = pd.DataFrame(rows)
    save_csv(quality, output_dir, "01_数据质量概览.csv")

    # 球队名称匹配检查
    team_set_1 = set(standings["球队"].astype(str).str.strip())
    team_set_2 = set(teams["球队"].astype(str).str.strip())

    match_df = pd.DataFrame({
        "检查项目": [
            "排名表球队数",
            "球队技术统计表球队数",
            "两表共同球队数",
            "仅排名表存在球队数",
            "仅球队统计表存在球队数",
        ],
        "结果": [
            len(team_set_1),
            len(team_set_2),
            len(team_set_1 & team_set_2),
            len(team_set_1 - team_set_2),
            len(team_set_2 - team_set_1),
        ]
    })
    save_csv(match_df, output_dir, "02_球队名称匹配检查.csv")

    # 球员姓名重复
    duplicate_names = players.loc[
        players["球员姓名"].duplicated(keep=False),
        ["球员姓名", "位置", "出场次数"]
    ].sort_values("球员姓名")
    save_csv(duplicate_names, output_dir, "03_重复球员姓名检查.csv")

    return quality, match_df


# =========================
# 3. 预处理与特征工程
# =========================
def preprocess_standings(standings):
    df = standings.copy()

    numeric_cols = [
        "联盟排名", "分区排名", "胜场", "负场", "胜率",
        "场均得分", "对手场均得分", "场均净胜分"
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 战绩字符串拆分
    for source_col, prefix in [
        ("主场战绩", "主场"),
        ("客场战绩", "客场"),
        ("赛区内战绩", "赛区内"),
        ("分区内战绩", "分区内"),
        ("最近10场", "最近10场"),
    ]:
        parsed = parse_record(df[source_col])
        df[f"{prefix}胜场"] = parsed["胜"]
        df[f"{prefix}负场"] = parsed["负"]
        df[f"{prefix}胜率"] = parsed["胜率"]

    df["连胜连败数值"] = parse_streak(df["连胜/连败"])
    df["主客场胜率差"] = df["主场胜率"] - df["客场胜率"]

    # 竞争层级：严格按联盟排名分为 10 / 10 / 10
    df["竞争层级"] = pd.cut(
        df["联盟排名"],
        bins=[0, 10, 20, np.inf],
        labels=["Top 10", "Middle 10", "Bottom 10"]
    )

    return df


def preprocess_teams(teams):
    df = teams.copy()

    numeric_cols = [c for c in df.columns if c != "球队"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 自定义指标
    df["三分出手占比"] = safe_divide(df["场均三分出手数"], df["场均投篮出手数"])
    df["助攻失误比"] = safe_divide(df["场均助攻"], df["场均失误"])
    df["前场篮板占比"] = safe_divide(df["场均前场篮板"], df["场均篮板"])
    df["罚球出手率"] = safe_divide(df["场均罚球出手数"], df["场均投篮出手数"])

    return df


def preprocess_players(players):
    df = players.copy()

    # 保留原始位置；建立位置大类
    df["原始位置"] = df["位置"]
    position_map = {
        "大前锋": "前锋",
        "小前锋": "前锋",
        "控球后卫": "后卫",
        "得分后卫": "后卫",
    }
    df["位置大类"] = df["位置"].replace(position_map)

    numeric_cols = [c for c in df.columns if c not in ["球员姓名", "位置", "原始位置", "位置大类"]]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 每36分钟数据
    for stat, new_col in [
        ("场均得分", "每36分钟得分"),
        ("场均篮板", "每36分钟篮板"),
        ("场均助攻", "每36分钟助攻"),
        ("场均抢断", "每36分钟抢断"),
        ("场均盖帽", "每36分钟盖帽"),
        ("场均失误", "每36分钟失误"),
    ]:
        df[new_col] = safe_divide(df[stat] * 36, df["场均时间(分钟)"])

    # 进阶效率指标（由现有场均数据近似计算）
    df["有效投篮命中率eFG(%)"] = safe_divide(
        df["场均投篮命中数"] + 0.5 * df["场均三分命中数"],
        df["场均投篮出手数"]
    ) * 100

    df["真实命中率TS(%)"] = safe_divide(
        df["场均得分"],
        2 * (df["场均投篮出手数"] + 0.44 * df["场均罚球出手数"])
    ) * 100

    df["三分出手占比"] = safe_divide(df["场均三分出手数"], df["场均投篮出手数"])
    df["助攻失误比"] = safe_divide(df["场均助攻"], df["场均失误"])
    df["三分得分贡献率"] = safe_divide(3 * df["场均三分命中数"], df["场均得分"])
    df["罚球得分贡献率"] = safe_divide(df["场均罚球命中数"], df["场均得分"])

    # 排行榜分析过滤条件：至少20场、场均10分钟，减少偶然性
    df["进入效率榜"] = (df["出场次数"] >= 20) & (df["场均时间(分钟)"] >= 10)

    return df


def merge_team_data(standings_clean, teams_clean):
    # 排名表和球队表都有“场均得分”，合并后保留两份但明确命名
    stand = standings_clean.rename(columns={"场均得分": "排名表场均得分"})
    merged = pd.merge(stand, teams_clean, on="球队", how="inner", validate="one_to_one")

    # 检查两张表场均得分差异
    merged["两表场均得分差"] = merged["场均得分"] - merged["排名表场均得分"]

    return merged


# ===============
# 4. 描述性统计
# ===============
def descriptive_statistics(team_all, players_clean, output_dir):
    # 球队描述统计
    team_metrics = [
        "胜率", "场均净胜分", "场均得分", "对手场均得分",
        "投篮命中率(%)", "三分命中率(%)", "罚球命中率(%)",
        "场均篮板", "场均助攻", "场均抢断", "场均盖帽", "场均失误"
    ]
    team_desc = team_all[team_metrics].describe().T
    team_desc["中位数"] = team_all[team_metrics].median()
    team_desc["变异系数"] = team_desc["std"] / team_desc["mean"]
    save_csv(team_desc.reset_index().rename(columns={"index": "指标"}), output_dir,
             "04_球队描述性统计.csv")

    # 球员描述统计
    player_metrics = [
        "出场次数", "场均时间(分钟)", "场均得分", "场均篮板", "场均助攻",
        "场均抢断", "场均盖帽", "场均失误", "投篮命中率(%)",
        "三分命中率(%)", "罚球命中率(%)", "两双次数", "三双次数"
    ]
    player_desc = players_clean[player_metrics].describe().T
    player_desc["中位数"] = players_clean[player_metrics].median()
    save_csv(player_desc.reset_index().rename(columns={"index": "指标"}), output_dir,
             "05_球员描述性统计.csv")


# =========================
# 5. 球队分析
# =========================
def team_ranking_analysis(team_all, output_dir):
    # 胜率排行榜
    rank_table = team_all[
        ["联盟排名", "球队", "分区", "胜场", "负场", "胜率",
         "场均得分", "对手场均得分", "场均净胜分"]
    ].sort_values("联盟排名")
    save_csv(rank_table, output_dir, "06_球队排名核心指标.csv")

    plot_df = team_all.sort_values("胜率", ascending=True)
    plt.figure(figsize=(10, 9))
    plt.barh(plot_df["球队"], plot_df["胜率"])
    plt.xlabel("胜率")
    plt.ylabel("球队")
    plt.title("NBA球队胜率排名")
    finish_plot(output_dir, "图01_球队胜率排名.png")

    # 净胜分
    plot_df = team_all.sort_values("场均净胜分", ascending=True)
    plt.figure(figsize=(10, 9))
    plt.barh(plot_df["球队"], plot_df["场均净胜分"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("场均净胜分")
    plt.ylabel("球队")
    plt.title("NBA球队场均净胜分")
    finish_plot(output_dir, "图02_球队场均净胜分.png")

    # 胜率分布
    plt.figure(figsize=(8, 5))
    plt.hist(team_all["胜率"].dropna(), bins=10, edgecolor="black")
    plt.xlabel("胜率")
    plt.ylabel("球队数量")
    plt.title("NBA球队胜率分布")
    finish_plot(output_dir, "图03_球队胜率分布.png")

    # 探究净胜分和胜率相关性
    plt.figure(figsize=(8, 6))
    plt.scatter(team_all["场均净胜分"], team_all["胜率"])
    for _, r in team_all.iterrows():
        plt.annotate(r["球队"], (r["场均净胜分"]-0.5, r["胜率"]),
                     fontsize=5, alpha=0.8, rotation = 45)
    x = team_all["场均净胜分"].to_numpy()
    y = team_all["胜率"].to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() >= 2:
        coef = np.polyfit(x[valid], y[valid], 1)
        xs = np.linspace(x[valid].min(), x[valid].max(), 100)
        plt.plot(xs, np.polyval(coef, xs))
    plt.xlabel("场均净胜分")
    plt.ylabel("胜率")
    plt.title("场均净胜分与胜率关系")
    finish_plot(output_dir, "图04_净胜分与胜率散点图.png")


def strong_weak_team_analysis(team_all, output_dir):
    metrics = [
        "场均得分", "对手场均得分", "场均净胜分",
        "投篮命中率(%)", "三分命中率(%)", "罚球命中率(%)",
        "场均篮板", "场均助攻", "场均抢断", "场均盖帽",
        "场均失误", "三分出手占比", "助攻失误比"
    ]

    group_mean = team_all.groupby("竞争层级", observed=False)[metrics].mean()
    save_csv(group_mean.reset_index(), output_dir, "07_强中弱球队均值比较.csv")

    # Z-score 后比较，避免不同量纲混在一起
    z = team_all[metrics].copy()
    z = (z - z.mean()) / z.std(ddof=0)
    z["竞争层级"] = team_all["竞争层级"].values
    z_group = z.groupby("竞争层级", observed=False)[metrics].mean()

    plot_heatmap(
        z_group,
        "强/中/弱球队核心指标标准化均值（Z-score）",
        output_dir / "图05_强中弱球队标准化指标比较.png"
    )

    # 绘制箱线图，观测层级与离散程度
    for idx, metric in enumerate(
        ["投篮命中率(%)", "三分命中率(%)", "场均篮板", "场均助攻", "场均失误"],
        start=6
    ):
        groups = []
        labels = ["Top 10", "Middle 10", "Bottom 10"]
        for label in labels:
            groups.append(
                team_all.loc[team_all["竞争层级"].astype(str) == label, metric]
                .dropna()
                .to_numpy()
            )

        plt.figure(figsize=(8, 5))
        plt.boxplot(groups, tick_labels=labels)
        plt.ylabel(metric)
        plt.title(f"不同竞争层级球队的{metric}比较")
        finish_plot(output_dir, f"图{idx:02d}_不同层级_{metric.replace('/', '_')}.png")


def team_correlation_analysis(team_all, output_dir):
    metrics = [
        "胜率", "场均净胜分", "场均得分", "对手场均得分",
        "投篮命中率(%)", "三分命中率(%)", "罚球命中率(%)",
        "场均篮板", "场均助攻", "场均抢断", "场均盖帽",
        "场均失误", "三分出手占比", "助攻失误比",
        "主场胜率", "客场胜率", "最近10场胜率"
    ]

    corr = team_all[metrics].corr(method="pearson")
    save_csv(corr.reset_index().rename(columns={"index": "指标"}), output_dir,
             "08_球队指标相关系数矩阵.csv")
    plot_heatmap(
        corr,
        "球队核心指标 Pearson 相关系数",
        output_dir / "图11_球队相关系数热力图.png"
    )

    # 提取出与胜率的关系
    win_corr = (
        corr["胜率"]
        .drop("胜率")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .rename("与胜率Pearson相关系数")
        .reset_index()
        .rename(columns={"index": "指标"})
    )
    save_csv(win_corr, output_dir, "09_各指标与胜率相关性排序.csv")

    # 绘制各技术指标与胜率散点图
    selected = [
        "场均得分", "对手场均得分", "投篮命中率(%)",
        "三分命中率(%)", "场均篮板", "场均助攻", "场均失误"
    ]
    start_no = 12
    for i, metric in enumerate(selected, start=start_no):
        x = team_all[metric].to_numpy()
        y = team_all["胜率"].to_numpy()
        valid = np.isfinite(x) & np.isfinite(y)

        plt.figure(figsize=(8, 6))
        plt.scatter(x[valid], y[valid])

        if valid.sum() >= 2:
            coef = np.polyfit(x[valid], y[valid], 1)
            xs = np.linspace(x[valid].min(), x[valid].max(), 100)
            plt.plot(xs, np.polyval(coef, xs))

        plt.xlabel(metric)
        plt.ylabel("胜率")
        r = np.corrcoef(x[valid], y[valid])[0, 1] if valid.sum() >= 2 else np.nan
        plt.title(f"{metric}与胜率关系（r={r:.3f}）")
        finish_plot(output_dir, f"图{i:02d}_{metric.replace('/', '_')}_与胜率.png")

    return win_corr


def home_away_analysis(team_all, output_dir):
    cols = [
        "球队", "联盟排名", "胜率",
        "主场胜率", "客场胜率", "主客场胜率差", "最近10场胜率", "连胜连败数值"
    ]
    home_away = team_all[cols].sort_values("主客场胜率差", ascending=False)
    save_csv(home_away, output_dir, "10_主客场及近期状态分析.csv")

    # 主场 vs 客场
    plt.figure(figsize=(8, 6))
    plt.scatter(team_all["主场胜率"], team_all["客场胜率"])
    lo = np.nanmin([team_all["主场胜率"].min(), team_all["客场胜率"].min()])
    hi = np.nanmax([team_all["主场胜率"].max(), team_all["客场胜率"].max()])
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    for _, r in team_all.iterrows():
        plt.annotate(r["球队"], (r["主场胜率"], r["客场胜率"]), fontsize=7, alpha=0.8)
    plt.xlabel("主场胜率")
    plt.ylabel("客场胜率")
    plt.title("NBA球队主场与客场胜率")
    finish_plot(output_dir, "图19_主客场胜率散点图.png")

    # 主场优势 Top 10
    top = home_away.head(10).sort_values("主客场胜率差", ascending=True)
    plt.figure(figsize=(9, 6))
    plt.barh(top["球队"], top["主客场胜率差"])
    plt.xlabel("主场胜率 - 客场胜率")
    plt.ylabel("球队")
    plt.title("主场优势最明显的10支球队")
    finish_plot(output_dir, "图20_主场优势Top10.png")


def conference_analysis(team_all, output_dir):
    metrics = [
        "胜率", "场均得分", "对手场均得分", "场均净胜分",
        "投篮命中率(%)", "三分命中率(%)",
        "场均篮板", "场均助攻", "场均抢断", "场均盖帽", "场均失误"
    ]

    summary = team_all.groupby("分区")[metrics].agg(["mean", "median", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    save_csv(summary.reset_index(), output_dir, "11_东西部统计比较.csv")

    conferences = list(team_all["分区"].dropna().unique())
    groups = [team_all.loc[team_all["分区"] == c, "胜率"].dropna().to_numpy() for c in conferences]

    plt.figure(figsize=(7, 5))
    plt.boxplot(groups, tick_labels=conferences)
    plt.ylabel("胜率")
    plt.title("东西部球队胜率分布比较")
    finish_plot(output_dir, "图21_东西部胜率箱线图.png")


# =========================
# 6. 球员分析
# =========================
def player_basic_analysis(players_clean, output_dir):
    # 位置人数
    position_counts = (
        players_clean["位置大类"]
        .value_counts()
        .rename_axis("位置")
        .reset_index(name="人数")
    )
    save_csv(position_counts, output_dir, "12_球员位置人数.csv")

    plt.figure(figsize=(7, 5))
    plt.bar(position_counts["位置"], position_counts["人数"])
    plt.xlabel("位置")
    plt.ylabel("人数")
    plt.title("球员位置分布")
    finish_plot(output_dir, "图22_球员位置分布.png")

    # 基础指标分布
    for no, metric in enumerate(
        ["场均时间(分钟)", "场均得分", "场均篮板", "场均助攻"],
        start=23
    ):
        plt.figure(figsize=(8, 5))
        plt.hist(players_clean[metric].dropna(), bins=20, edgecolor="black")
        plt.xlabel(metric)
        plt.ylabel("球员人数")
        plt.title(f"球员{metric}分布")
        finish_plot(output_dir, f"图{no:02d}_球员{metric.replace('/', '_')}分布.png")

    # 出场时间 vs 得分
    plt.figure(figsize=(8, 6))
    plt.scatter(players_clean["场均时间(分钟)"], players_clean["场均得分"], alpha=0.65)
    plt.xlabel("场均时间(分钟)")
    plt.ylabel("场均得分")
    r = players_clean[["场均时间(分钟)", "场均得分"]].corr().iloc[0, 1]
    plt.title(f"球员上场时间与得分关系（r={r:.3f}）")
    finish_plot(output_dir, "图27_球员时间与得分关系.png")


def player_position_analysis(players_clean, output_dir):
    metrics = [
        "场均得分", "场均篮板", "场均助攻",
        "场均抢断", "场均盖帽", "投篮命中率(%)",
        "三分命中率(%)", "真实命中率TS(%)"
    ]

    position_summary = (
        players_clean.groupby("位置大类")[metrics]
        .agg(["count", "mean", "median", "std"])
    )
    position_summary.columns = [f"{a}_{b}" for a, b in position_summary.columns]
    save_csv(position_summary.reset_index(), output_dir, "13_不同位置球员统计比较.csv")

    positions = [p for p in ["后卫", "前锋", "中锋"] if p in set(players_clean["位置大类"])]

    for no, metric in enumerate(
        ["场均得分", "场均篮板", "场均助攻", "场均盖帽"],
        start=28
    ):
        groups = [
            players_clean.loc[players_clean["位置大类"] == p, metric].dropna().to_numpy()
            for p in positions
        ]
        plt.figure(figsize=(8, 5))
        plt.boxplot(groups, tick_labels=positions, showfliers=False)
        plt.xlabel("位置")
        plt.ylabel(metric)
        plt.title(f"不同位置球员{metric}比较")
        finish_plot(output_dir, f"图{no:02d}_不同位置_{metric}.png")


def player_leaderboards(players_clean, output_dir):
    eligible = players_clean.loc[players_clean["进入效率榜"]].copy()

    base_cols = ["球员姓名", "位置大类", "出场次数", "场均时间(分钟)"]

    leaderboards = {
        "14_场均得分Top20.csv": ("场均得分", False),
        "15_场均篮板Top20.csv": ("场均篮板", False),
        "16_场均助攻Top20.csv": ("场均助攻", False),
        "17_每36分钟得分Top20.csv": ("每36分钟得分", True),
        "18_真实命中率Top20.csv": ("真实命中率TS(%)", True),
        "19_有效投篮命中率Top20.csv": ("有效投篮命中率eFG(%)", True),
        "20_两双次数Top20.csv": ("两双次数", False),
        "21_三双次数Top20.csv": ("三双次数", False),
    }

    for filename, (metric, use_eligible) in leaderboards.items():
        src = eligible if use_eligible else players_clean
        cols = base_cols + [metric]
        out = src[cols].sort_values(metric, ascending=False).head(20)
        save_csv(out, output_dir, filename)

    # 得分 Top 15 图
    top = players_clean.sort_values("场均得分", ascending=False).head(15).sort_values("场均得分")
    plt.figure(figsize=(9, 7))
    plt.barh(top["球员姓名"], top["场均得分"])
    plt.xlabel("场均得分")
    plt.ylabel("球员")
    plt.title("场均得分 Top 15")
    finish_plot(output_dir, "图32_球员场均得分Top15.png")

    # 每36分钟得分 Top 15
    top36 = eligible.sort_values("每36分钟得分", ascending=False).head(15).sort_values("每36分钟得分")
    plt.figure(figsize=(9, 7))
    plt.barh(top36["球员姓名"], top36["每36分钟得分"])
    plt.xlabel("每36分钟得分")
    plt.ylabel("球员")
    plt.title("每36分钟得分 Top 15（至少20场且场均10分钟）")
    finish_plot(output_dir, "图33_每36分钟得分Top15.png")


def player_efficiency_analysis(players_clean, output_dir):
    # 相关矩阵
    metrics = [
        "场均时间(分钟)", "场均得分", "场均篮板", "场均助攻",
        "场均抢断", "场均盖帽", "场均失误",
        "投篮命中率(%)", "三分命中率(%)",
        "有效投篮命中率eFG(%)", "真实命中率TS(%)"
    ]
    corr = players_clean[metrics].corr()
    save_csv(corr.reset_index().rename(columns={"index": "指标"}), output_dir,
             "22_球员核心指标相关系数矩阵.csv")
    plot_heatmap(
        corr,
        "球员核心指标 Pearson 相关系数",
        output_dir / "图34_球员相关系数热力图.png"
    )

    # FGA vs PTS
    plt.figure(figsize=(8, 6))
    plt.scatter(players_clean["场均投篮出手数"], players_clean["场均得分"], alpha=0.65)
    x = players_clean["场均投篮出手数"].to_numpy()
    y = players_clean["场均得分"].to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() >= 2:
        coef = np.polyfit(x[valid], y[valid], 1)
        xs = np.linspace(x[valid].min(), x[valid].max(), 100)
        plt.plot(xs, np.polyval(coef, xs))
    r = np.corrcoef(x[valid], y[valid])[0, 1]
    plt.xlabel("场均投篮出手数")
    plt.ylabel("场均得分")
    plt.title(f"投篮出手数与得分关系（r={r:.3f}）")
    finish_plot(output_dir, "图35_投篮出手与得分关系.png")

    # 3PA vs 3P%
    three = players_clean.loc[players_clean["场均三分出手数"] >= 1].copy()
    plt.figure(figsize=(8, 6))
    plt.scatter(three["场均三分出手数"], three["三分命中率(%)"], alpha=0.65)
    plt.xlabel("场均三分出手数")
    plt.ylabel("三分命中率(%)")
    plt.title("三分出手量与三分命中率（场均至少1次三分出手）")
    finish_plot(output_dir, "图36_三分出手与命中率.png")


# =========================
# 7. 自动生成文字摘要
# =========================
def generate_summary(players_clean, team_all, win_corr, quality, output_dir):
    lines = []
    lines.append("NBA 2025-2026 数据分析自动摘要")
    lines.append("=" * 50)
    lines.append("")
    lines.append("一、数据规模与质量")
    lines.append(f"球员数据：{len(players_clean)} 名球员。")
    lines.append(f"球队数据：{len(team_all)} 支球队。")
    lines.append(
        "三张原始表缺失值总数分别为：" +
        "、".join(f"{r['数据表']}={r['缺失值总数']}" for _, r in quality.iterrows()) + "。"
    )
    lines.append(
        "三张原始表整行重复数分别为：" +
        "、".join(f"{r['数据表']}={r['整行重复数']}" for _, r in quality.iterrows()) + "。"
    )

    lines.append("")
    lines.append("二、球队表现")
    best = team_all.sort_values("胜率", ascending=False).iloc[0]
    worst = team_all.sort_values("胜率", ascending=True).iloc[0]
    diff_best = team_all.sort_values("场均净胜分", ascending=False).iloc[0]

    lines.append(
        f"当前胜率最高球队：{best['球队']}，胜率={best['胜率']:.3f}，"
        f"战绩={int(best['胜场'])}-{int(best['负场'])}。"
    )
    lines.append(
        f"当前胜率最低球队：{worst['球队']}，胜率={worst['胜率']:.3f}。"
    )
    lines.append(
        f"场均净胜分最高球队：{diff_best['球队']}，场均净胜分={diff_best['场均净胜分']:.1f}。"
    )

    lines.append("")
    lines.append("三、与胜率关系较强的指标")
    for _, r in win_corr.head(6).iterrows():
        lines.append(f"{r['指标']}：r={r['与胜率Pearson相关系数']:.3f}")
    lines.append("注意：相关系数描述统计关联，不等同于因果关系。")

    lines.append("")
    lines.append("四、主客场")
    home_adv = team_all.sort_values("主客场胜率差", ascending=False).iloc[0]
    away_stable = team_all.sort_values("客场胜率", ascending=False).iloc[0]
    lines.append(
        f"主客场胜率差最大球队：{home_adv['球队']}，"
        f"主场胜率={home_adv['主场胜率']:.3f}，客场胜率={home_adv['客场胜率']:.3f}。"
    )
    lines.append(
        f"客场胜率最高球队：{away_stable['球队']}，客场胜率={away_stable['客场胜率']:.3f}。"
    )

    lines.append("")
    lines.append("五、球员表现")
    scorer = players_clean.sort_values("场均得分", ascending=False).iloc[0]
    rebounder = players_clean.sort_values("场均篮板", ascending=False).iloc[0]
    passer = players_clean.sort_values("场均助攻", ascending=False).iloc[0]
    lines.append(f"场均得分最高：{scorer['球员姓名']}，{scorer['场均得分']:.1f} 分。")
    lines.append(f"场均篮板最高：{rebounder['球员姓名']}，{rebounder['场均篮板']:.1f} 个。")
    lines.append(f"场均助攻最高：{passer['球员姓名']}，{passer['场均助攻']:.1f} 次。")

    eligible = players_clean.loc[players_clean["进入效率榜"]]
    if not eligible.empty:
        per36 = eligible.sort_values("每36分钟得分", ascending=False).iloc[0]
        lines.append(
            f"满足至少20场且场均10分钟条件的球员中，每36分钟得分最高："
            f"{per36['球员姓名']}，{per36['每36分钟得分']:.2f} 分。"
        )

    lines.append("")
    lines.append("六、解释注意")
    lines.append("1. 本部分属于描述性/探索性数据分析，不直接做因果推断。")
    lines.append("2. 球员数据没有球队字段，因此球员与球队暂不直接联结。")
    lines.append("3. eFG% 与 TS% 由现有场均数据近似计算，适合本项目内部比较。")
    lines.append("4. 排名表与球队统计表的场均得分存在极小口径差异时，两者均保留并单独检查。")

    text = "\n".join(lines)
    (output_dir / "23_自动分析摘要.txt").write_text(text, encoding="utf-8")


# =========================
# 8. 主程序
# =========================
def main():
    setup_chinese_font()

    excel_path = find_data_file()
    output_dir = excel_path.parent / OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NBA 2025-2026 数据分析项目")
    print("=" * 70)
    print(f"读取文件：{excel_path}")
    print(f"输出目录：{output_dir}")

    players, standings, teams = load_data(excel_path)

    print("\n[1/8] 数据质量检查...")
    quality, match_df = data_quality_check(players, standings, teams, output_dir)

    print("[2/8] 数据预处理与特征工程...")
    standings_clean = preprocess_standings(standings)
    teams_clean = preprocess_teams(teams)
    players_clean = preprocess_players(players)
    team_all = merge_team_data(standings_clean, teams_clean)

    # 保存清洗数据
    save_csv(players_clean, output_dir, "清洗后_球员数据.csv")
    save_csv(standings_clean, output_dir, "清洗后_排名数据.csv")
    save_csv(teams_clean, output_dir, "清洗后_球队数据.csv")
    save_csv(team_all, output_dir, "清洗后_球队综合数据.csv")

    # 两表得分一致性检查
    ppg_check = team_all[
        ["球队", "排名表场均得分", "场均得分", "两表场均得分差"]
    ].copy()
    save_csv(ppg_check, output_dir, "24_两表场均得分一致性检查.csv")

    print("[3/8] 描述性统计...")
    descriptive_statistics(team_all, players_clean, output_dir)

    print("[4/8] 球队排名、强弱对比与相关性分析...")
    team_ranking_analysis(team_all, output_dir)
    strong_weak_team_analysis(team_all, output_dir)
    win_corr = team_correlation_analysis(team_all, output_dir)

    print("[5/8] 主客场与东西部分析...")
    home_away_analysis(team_all, output_dir)
    conference_analysis(team_all, output_dir)

    print("[6/8] 球员基础表现与位置差异分析...")
    player_basic_analysis(players_clean, output_dir)
    player_position_analysis(players_clean, output_dir)

    print("[7/8] 球员排行榜与效率分析...")
    player_leaderboards(players_clean, output_dir)
    player_efficiency_analysis(players_clean, output_dir)

    print("[8/8] 生成自动分析摘要...")
    generate_summary(players_clean, team_all, win_corr, quality, output_dir)


    print("\n结果已保存到：")
    print(output_dir)


if __name__ == "__main__":
    main()
