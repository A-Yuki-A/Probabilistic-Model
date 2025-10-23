import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from math import comb

# -------------------------------
# ページ設定
# -------------------------------
st.set_page_config(
    page_title="確率シミュレーション・ラボ",
    page_icon="🎲",
    layout="wide",
)

# Altair設定（軸/余白を広めに）
CHART_H = 380
PADDING = {"left": 70, "right": 20, "top": 10, "bottom": 60}
alt.themes.enable("default")

# -------------------------------
# ユーティリティ
# -------------------------------
def set_seed(seed: int):
    return np.random.default_rng(seed)

def binom_pmf(n: int, p: float):
    k = np.arange(n + 1)
    probs = np.array([comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in k])
    return k, probs

def int_uniform_prob_ge(lo: int, hi: int, threshold: int):
    if lo > hi:
        lo, hi = hi, lo
    if threshold <= lo:
        return 1.0
    if threshold > hi:
        return 0.0
    return (hi - threshold + 1) / (hi - lo + 1)

def run_button(label: str, key: str) -> bool:
    pressed = st.button(label, key=key, type="primary")
    if pressed:
        st.session_state[key] = True
    return st.session_state.get(key, False)

# -------------------------------
# サイドバー（共通設定）
# -------------------------------
st.sidebar.header("共通設定")
seed = st.sidebar.number_input("乱数シード（同じ結果を再現）", min_value=0, max_value=10**9, value=42, step=1)
sim_repeats = st.sidebar.slider("シミュレーション反復（平均を安定化）", 1, 2000, 500, help="同じ条件で繰り返す回数（大きいほど誤差が小さくなります）")
rng = set_seed(seed)

st.sidebar.markdown("---")
st.sidebar.markdown("**用語メモ**")
st.sidebar.caption(
    "・理論値：確率に基づく“こうなるはず”の分布や割合\n"
    "・期待値：分布の平均（例：コイン10回の表の期待値は 10×1/2=5）\n"
    "・経験的確率：実験（シミュレーション）から得た割合"
)

st.title("確率シミュレーション・ラボ（問1〜問4）")
st.caption("パラメータを調整 → 「実行」ボタンでグラフ更新。理論とシミュレーションの収束を体験しよう。")

tab1, tab2, tab3, tab4 = st.tabs(["問1：コイン10回×人数分布", "問2：サイコロの割合", "問3：RPG（命中×ダメージ）", "問4：ガチャ（1回以上当たる）"])

# -------------------------------
# 問1：コイン10回×人数分布
# -------------------------------
with tab1:
    st.subheader("問1：10回コインを投げた“表の回数”の分布（人数を増やす）")
    col_a, col_b = st.columns([1,2])
    with col_a:
        n = st.number_input("1人あたりの試行回数（回）", 1, 200, 10)
        p = st.slider("表の出る確率 p", 0.0, 1.0, 0.5, 0.01)
        people_choices = [1, 10, 50, 100]
        people_list = st.multiselect("人数の候補（複数選択可）", options=people_choices, default=[10, 50, 100])
        show_theory = st.checkbox("理論（二項分布）を重ねて表示", value=True)
        st.caption("横軸：表の回数（0〜n）／縦軸：その回数が出た人数")

        ready = run_button("実行（問1）", key="run_q1")

    with col_b:
        if not people_list:
            st.info("人数の候補を少なくとも1つ選んでください。")
        elif ready:
            k_vals, pmf = binom_pmf(n, p)
            layers = []
            for people in people_list:
                counts_accum = np.zeros(n+1, dtype=float)
                for _ in range(sim_repeats):
                    heads = rng.binomial(n=n, p=p, size=people)
                    binc = np.bincount(heads, minlength=n+1)
                    counts_accum += binc
                counts_mean = counts_accum / sim_repeats
                df_sim = pd.DataFrame({
                    "表の回数": np.arange(n+1),
                    "人数": counts_mean.astype(float),
                    "凡例": f"{people}人"
                })
                bars = alt.Chart(df_sim).mark_bar(clip=False).encode(
                    x=alt.X("表の回数:O", title="表の回数"),
                    y=alt.Y("人数:Q", title="その回数が出た人数"),
                    color=alt.Color("凡例:N", legend=alt.Legend(title="人数"))
                ).properties(width="container", height=CHART_H, padding=PADDING)
                layers.append(bars)

                if show_theory:
                    df_theory = pd.DataFrame({
                        "表の回数": k_vals.astype(str),
                        "期待人数（理論）": people * pmf,
                        "凡例": f"{people}人（理論）"
                    })
                    line = alt.Chart(df_theory).mark_line(point=True, clip=False).encode(
                        x=alt.X("表の回数:O", title="表の回数"),
                        y=alt.Y("期待人数（理論）:Q", title="その回数が出た人数"),
                        color=alt.Color("凡例:N", legend=alt.Legend(title="人数"))
                    ).properties(width="container", height=CHART_H, padding=PADDING)
                    layers.append(line)

            st.altair_chart(alt.layer(*layers).resolve_scale(color='independent'), use_container_width=True)
        else:
            st.info("パラメータを設定して「実行（問1）」を押してください。")

    with st.expander("学習ポイント（要約）"):
        st.markdown(
            "- 人数が少ないと分布にムラが出やすい。\n"
            "- 人数を増やすと二項分布（理論）に近づく（大数の法則）。\n"
            "- **理論値**＝確率に基づく“こうなるはず”の分布、**期待値**＝その平均。"
        )

# -------------------------------
# 問2：サイコロの割合
# -------------------------------
with tab2:
    st.subheader("問2：サイコロの各目の“割合”（試行回数の影響）")
    col_a, col_b = st.columns([1,2])
    with col_a:
        dice_trials = st.number_input("サイコロを振る回数（1セット）", 1, 100000, 1000, step=10)
        show_ratio = st.radio("表示（回数 or 割合）", ["割合", "回数"], index=0)
        st.caption("横軸：出目（1〜6）／縦軸：出目の出た割合（または回数）")
        ready = run_button("実行（問2）", key="run_q2")

    with col_b:
        if ready:
            counts_accum = np.zeros(6, dtype=float)
            for _ in range(sim_repeats):
                rolls = rng.integers(1, 7, size=dice_trials)
                binc = np.bincount(rolls, minlength=7)[1:]
                counts_accum += binc
            counts_mean = counts_accum / sim_repeats

            if show_ratio == "割合":
                yvals = counts_mean / counts_mean.sum()
                ytitle = "割合"
                theory_line_val = 1/6
            else:
                yvals = counts_mean
                ytitle = "回数"
                theory_line_val = dice_trials / 6

            df = pd.DataFrame({"出目": [str(i) for i in range(1,7)], ytitle: yvals})
            bars = alt.Chart(df).mark_bar(clip=False).encode(
                x=alt.X("出目:O", title="出目"),
                y=alt.Y(f"{ytitle}:Q", title=ytitle),
            ).properties(width="container", height=CHART_H, padding=PADDING)
            chart = bars

            if show_ratio == "割合":
                line_df = pd.DataFrame({"出目": [str(i) for i in range(1,7)],
                                        "理論（1/6）": [theory_line_val]*6})
                line = alt.Chart(line_df).mark_rule().encode(y="理論（1/6):Q")
                chart = chart + line

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("パラメータを設定して「実行（問2）」を押してください。")

    with st.expander("学習ポイント（要約）"):
        st.markdown(
            "- 回数が少ないと割合にムラが出やすい。\n"
            "- 回数を増やすと各目の割合は **1/6** に近づく（大数の法則）。\n"
            "- “確率（理論）”は一定、変わるのは“実験から得られる割合”。"
        )

# -------------------------------
# 問3：RPG（命中×ダメージ）
# -------------------------------
with tab3:
    st.subheader("問3：RPG（命中率×ダメージ）で敵を倒せる確率")
    col_a, col_b = st.columns([1,2])
    with col_a:
        hit_p = st.slider("命中率（%）", 0.0, 100.0, 65.0, 0.5)
        dmg_lo = st.number_input("ダメージ下限（整数）", 0, 9999, 90)
        dmg_hi = st.number_input("ダメージ上限（整数）", 0, 9999, 99)
        hp = st.number_input("敵のHP（整数）", 1, 99999, 95)
        sims = st.number_input("戦闘を繰り返す回数（1セット）", 1, 1000000, 1000, step=100)
        show_converge = st.checkbox("収束のようす（累積推定）も表示", value=True)
        st.caption("倒せる条件：攻撃が命中 かつ ダメージ≥HP（ダメージは整数一様）")
        ready = run_button("実行（問3）", key="run_q3")

    with col_b:
        if ready:
            p_hit = hit_p / 100.0
            p_dmg = int_uniform_prob_ge(int(dmg_lo), int(dmg_hi), int(hp))
            p_theory = p_hit * p_dmg

            success_rate_accum = 0.0
            if show_converge:
                cum_estimates = np.zeros(sims, dtype=float)

            for _ in range(sim_repeats):
                hits = rng.random(sims) < p_hit
                dmg = rng.integers(int(min(dmg_lo,dmg_hi)), int(max(dmg_lo,dmg_hi))+1, size=sims)
                success = hits & (dmg >= hp)
                sr = success.mean()
                success_rate_accum += sr
                if show_converge:
                    cum_estimates += np.cumsum(success) / (np.arange(sims) + 1)

            success_rate = success_rate_accum / sim_repeats

            c1, c2, c3 = st.columns(3)
            c1.metric("理論：倒せる確率", f"{p_theory*100:.2f}%")
            c2.metric("実測：倒せる確率（平均）", f"{success_rate*100:.2f}%")
            c3.metric("ダメージ≥HPの確率", f"{p_dmg*100:.2f}%")

            if show_converge:
                est = cum_estimates / sim_repeats
                df_conv = pd.DataFrame({
                    "試行回": np.arange(1, sims+1),
                    "累積推定（実測）": est,
                    "理論値": np.full(sims, p_theory)
                })
                line1 = alt.Chart(df_conv).mark_line(clip=False).encode(
                    x=alt.X("試行回:Q", title="試行回"),
                    y=alt.Y("累積推定（実測）:Q", title="倒せる確率"),
                ).properties(width="container", height=CHART_H, padding=PADDING)
                line2 = alt.Chart(df_conv).mark_line(strokeDash=[4,4], clip=False).encode(
                    x="試行回:Q",
                    y="理論値:Q",
                ).properties(width="container", height=CHART_H, padding=PADDING)
                st.altair_chart(line1 + line2, use_container_width=True)
        else:
            st.info("パラメータを設定して「実行（問3）」を押してください。")

    with st.expander("学習ポイント（要約）"):
        st.markdown(
            "- 倒せる確率＝ **命中の確率 ×（ダメージがHP以上になる確率）** の積。\n"
            "- 試行回数を増やすと、実測の割合は理論値に近づく（大数の法則）。\n"
            "- “命中率＝倒せる確率”ではないことに注意。"
        )

# -------------------------------
# 問4：ガチャ（1回以上当たる確率）
# -------------------------------
with tab4:
    st.subheader("問4：ガチャで“1回以上当たる”確率")
    col_a, col_b = st.columns([1,2])
    with col_a:
        p_ssr = st.slider("1回あたりの当選確率（%）", 0.0, 100.0, 1.0, 0.1)
        max_n = st.slider("横軸の上限（引く回数 n の最大値）", 1, 1000, 200, step=10,
                          help="グラフの横軸上限です。1〜この値までの回数 n を描画します。")
        show_sim = st.checkbox("シミュレーション（実測線）も重ねる", value=True)
        n_for_point = st.number_input("指定回数 n での当選確率（数値表示）", 1, 100000, 100, step=10,
                                      help="ここで指定した n 回引いたときの“1回以上当たる確率”を数値で表示します。")
        st.caption("理論：当たる確率 = 1 - (1-p)^n（pは1回あたりの当選確率）")
        ready = run_button("実行（問4）", key="run_q4")

    with col_b:
        if ready:
            p = p_ssr / 100.0
            n_axis = np.arange(1, max_n + 1)
            theory = 1 - (1 - p) ** n_axis
            df = pd.DataFrame({"回数": n_axis, "1回以上当たる確率（理論）": theory})

            chart = alt.Chart(df).mark_line(clip=False).encode(
                x=alt.X("回数:Q", title="引いた回数 n"),
                y=alt.Y("1回以上当たる確率（理論）:Q", axis=alt.Axis(format=".%"), title="確率"),
            ).properties(width="container", height=CHART_H, padding=PADDING)

            if show_sim:
                step = max(1, max_n // 100)
                sim_ns = np.arange(1, max_n + 1, step)
                sim_vals = []
                for n in sim_ns:
                    success = 0
                    for _ in range(sim_repeats):
                        trials = rng.random(n) < p
                        if trials.any():
                            success += 1
                    sim_vals.append(success / sim_repeats)
                df_sim = pd.DataFrame({"回数": sim_ns, "1回以上当たる確率（実測）": sim_vals})
                sim_line = alt.Chart(df_sim).mark_line(point=True, clip=False).encode(
                    x="回数:Q",
                    y=alt.Y("1回以上当たる確率（実測）:Q", axis=alt.Axis(format=".%")),
                    color=alt.value("#888")
                ).properties(width="container", height=CHART_H, padding=PADDING)
                chart = chart + sim_line

            st.altair_chart(chart, use_container_width=True)

            # スポット（指定回数 n の数値表示）
            point_prob = 1 - (1 - p) ** n_for_point
            c1, c2 = st.columns(2)
            c1.metric(f"n={n_for_point} 回：理論", f"{point_prob*100:.2f}%")
            success = 0
            for _ in range(sim_repeats):
                trials = rng.random(n_for_point) < p
                if trials.any():
                    success += 1
            emp = success / sim_repeats
            c2.metric(f"n={n_for_point} 回：実測（平均）", f"{emp*100:.2f}%")
            st.caption("※「指定回数 n での当選確率（数値表示）」は、上の曲線の特定点を数値で確認するための機能です。")
        else:
            st.info("パラメータを設定して「実行（問4）」を押してください。")

st.markdown("---")
st.caption("© 確率シミュレーション・ラボ｜高校生向け可視化教材（Streamlit）")
