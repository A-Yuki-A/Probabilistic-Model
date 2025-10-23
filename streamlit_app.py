import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# -------------------------------
# 共通ページ設定
# -------------------------------
st.set_page_config(
    page_title="確率シミュレーションラボ",
    page_icon="🎲",
    layout="wide",
)

# -------------------------------
# ユーティリティ
# -------------------------------
def set_seed(seed: int):
    rng = np.random.default_rng(seed)
    return rng

def binom_pmf(n: int, p: float):
    # 二項分布の理論分布（0..n）
    k = np.arange(n + 1)
    # nCk * p^k * (1-p)^(n-k)
    from math import comb
    probs = np.array([comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in k])
    return k, probs

def int_uniform_prob_ge(lo: int, hi: int, threshold: int):
    # [lo..hi]（整数一様）で ">= threshold" となる確率
    if threshold <= lo:
        return 1.0
    if threshold > hi:
        return 0.0
    count_ok = hi - threshold + 1
    count_all = hi - lo + 1
    return count_ok / count_all

# Altairテーマ：日本語ラベルが見やすいようフォントサイズ少し大きめ
alt.themes.enable("default")

# -------------------------------
# サイドバー（共通設定）
# -------------------------------
st.sidebar.header("共通設定")
seed = st.sidebar.number_input("乱数シード（同じ結果を再現）", min_value=0, max_value=10**9, value=42, step=1)
rng = set_seed(seed)
sim_repeats = st.sidebar.slider("シミュレーション反復（平均を安定化）", 1, 2000, 500, help="同じ条件で繰り返す回数（大きいほど誤差が小さくなります）")

st.sidebar.markdown("---")
st.sidebar.markdown("**用語メモ**")
st.sidebar.caption(
    "・理論値：確率に基づく“こうなるはず”の分布や割合\n"
    "・期待値：分布の平均（例：コイン10回の表の期待値は 10×1/2=5）\n"
    "・経験的確率：実験（シミュレーション）から得た割合"
)

st.title("確率シミュレーション・ラボ（問1〜問4）")
st.caption("パラメータを動かして、理論とシミュレーションの差や収束を体験しよう。")

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
        people_list_txt = st.text_input("人数の候補（カンマ区切り）", value="10,40,200,1000")
        try:
            people_list = [int(s.strip()) for s in people_list_txt.split(",") if s.strip()]
            people_list = [v for v in people_list if v > 0]
        except:
            people_list = [10,40,200,1000]
        show_theory = st.checkbox("理論（二項分布）を重ねて表示", value=True)
        st.caption("横軸：表の回数（0〜n）／縦軸：その回数が出た人数")

    with col_b:
        # 理論分布（人数ではなく確率）。重ねるときは人数に合わせて期待人数へスケーリングして表示
        k_vals, pmf = binom_pmf(n, p)

        # シミュレーション：人数ごとに、表の回数のヒストグラムを描く
        charts = []
        for idx, people in enumerate(people_list):
            # 反復して平均化（各反復で people×n の二項実験を people 人ぶん）
            counts_accum = np.zeros(n+1, dtype=float)
            for _ in range(sim_repeats):
                # people 人それぞれについて表の回数をサンプル
                heads = rng.binomial(n=n, p=p, size=people)
                # 0..n の度数を集計
                binc = np.bincount(heads, minlength=n+1)
                counts_accum += binc
            counts_mean = counts_accum / sim_repeats

            df_sim = pd.DataFrame({
                "表の回数": np.arange(n+1),
                "人数": counts_mean.astype(float),
                "グラフ": f"{people}人"
            })
            bars = alt.Chart(df_sim).mark_bar().encode(
                x=alt.X("表の回数:O", title="表の回数"),
                y=alt.Y("人数:Q", title="その回数が出た人数"),
                color=alt.Color("グラフ:N", legend=alt.Legend(title="人数"))
            )
            charts.append(bars)

            if show_theory:
                # 期待“人数” = people * pmf
                df_theory = pd.DataFrame({
                    "表の回数": k_vals.astype(str),
                    "期待人数（理論）": people * pmf,
                    "グラフ": f"{people}人（理論）"
                })
                line = alt.Chart(df_theory).mark_line(point=True).encode(
                    x=alt.X("表の回数:O", title="表の回数"),
                    y=alt.Y("期待人数（理論）:Q", title="その回数が出た人数"),
                    color=alt.Color("グラフ:N", legend=alt.Legend(title="人数"))
                )
                charts.append(line)

        if charts:
            st.altair_chart(alt.layer(*charts).resolve_scale(color='independent'), use_container_width=True)

    with st.expander("学習ポイント（要約）"):
        st.markdown(
            "- 人数が少ないと分布の形にムラ（ばらつき）が出やすい。\n"
            "- 人数を増やすと二項分布（理論）に近い滑らかな形に近づく（大数の法則）。\n"
            "- **理論値**＝確率に基づく“こうなるはず”の分布、**期待値**＝その分布の平均。"
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

    with col_b:
        # 反復して平均化
        counts_accum = np.zeros(6, dtype=float)
        for _ in range(sim_repeats):
            rolls = rng.integers(1, 7, size=dice_trials)
            binc = np.bincount(rolls, minlength=7)[1:]  # index 1..6
            counts_accum += binc
        counts_mean = counts_accum / sim_repeats

        if show_ratio == "割合":
            yvals = counts_mean / counts_mean.sum()
            ytitle = "割合"
            theory_line_val = 1/6
        else:
            yvals = counts_mean
            ytitle = "回数"
            theory_line_val = dice_trials / 6  # 1セットの期待回数（参考）

        df = pd.DataFrame({"出目": [str(i) for i in range(1,7)], ytitle: yvals})
        bars = alt.Chart(df).mark_bar().encode(
            x=alt.X("出目:O"),
            y=alt.Y(f"{ytitle}:Q", title=ytitle),
        )
        chart = bars

        if show_ratio == "割合":
            line_df = pd.DataFrame({"出目": [str(i) for i in range(1,7)],
                                    "理論（1/6）": [theory_line_val]*6})
            line = alt.Chart(line_df).mark_rule().encode(y="理論（1/6):Q")
            chart = chart + line

        st.altair_chart(chart, use_container_width=True)

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

        st.caption("倒せる条件：攻撃が命中 かつ ダメージ≥HP")
        st.caption("注）ダメージは指定範囲の **整数一様分布** を仮定")

    with col_b:
        p_hit = hit_p / 100.0
        # 理論：P(倒す) = p_hit * P(damage >= HP)
        p_dmg = int_uniform_prob_ge(int(dmg_lo), int(dmg_hi), int(hp))
        p_theory = p_hit * p_dmg

        # シミュレーション（反復平均）
        success_rate_accum = 0.0
        # 収束表示用
        if show_converge:
            cum_estimates = np.zeros(sims, dtype=float)

        for r in range(sim_repeats):
            # 1セットの sims 回
            hits = rng.random(sims) < p_hit
            # ダメージ（整数一様）
            if dmg_lo > dmg_hi:
                dmg_lo, dmg_hi = dmg_hi, dmg_lo
            dmg = rng.integers(int(dmg_lo), int(dmg_hi)+1, size=sims)
            success = hits & (dmg >= hp)
            sr = success.mean()
            success_rate_accum += sr

            if show_converge:
                # ランニング平均
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
            line1 = alt.Chart(df_conv).mark_line().encode(
                x=alt.X("試行回:Q"),
                y=alt.Y("累積推定（実測）:Q", title="倒せる確率"),
            )
            line2 = alt.Chart(df_conv).mark_line(strokeDash=[4,4]).encode(
                x="試行回:Q",
                y="理論値:Q",
            )
            st.altair_chart(line1 + line2, use_container_width=True)

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
        max_n = st.slider("最大回数 n（横軸の範囲）", 1, 1000, 200, step=10)
        show_sim = st.checkbox("シミュレーション（実測線）も重ねる", value=True)
        n_for_point = st.number_input("スポット表示：何回引く？", 1, 100000, 100, step=10)
        st.caption("理論：当たる確率 = 1 - (1-p)^n（pは1回あたりの当選確率）")

    with col_b:
        p = p_ssr / 100.0
        n_axis = np.arange(1, max_n + 1)
        theory = 1 - (1 - p) ** n_axis
        df = pd.DataFrame({"回数": n_axis, "1回以上当たる確率（理論）": theory})

        chart = alt.Chart(df).mark_line().encode(
            x=alt.X("回数:Q"),
            y=alt.Y("1回以上当たる確率（理論）:Q", axis=alt.Axis(format="%"), title="確率"),
        )

        if show_sim:
            # シミュレーション：各nでsim_repeatsセットやると重いので、等間引き（高速化）
            step = max(1, max_n // 100)  # 点数を抑えて軽量化
            sim_ns = np.arange(1, max_n + 1, step)
            sim_vals = []
            for n in sim_ns:
                # 1セット＝“n回引く”を sim_repeats 回
                # 1回も当たらない確率を直接サンプルするより、乱数で当たり判定
                success = 0
                for _ in range(sim_repeats):
                    trials = rng.random(n) < p
                    if trials.any():
                        success += 1
                sim_vals.append(success / sim_repeats)
            df_sim = pd.DataFrame({"回数": sim_ns, "1回以上当たる確率（実測）": sim_vals})
            sim_line = alt.Chart(df_sim).mark_line(point=True).encode(
                x="回数:Q",
                y=alt.Y("1回以上当たる確率（実測）:Q", axis=alt.Axis(format="%")),
                color=alt.value("#999")
            )
            chart = chart + sim_line

        # スポット値（n_for_point）
        point_prob = 1 - (1 - p) ** n_for_point
        st.altair_chart(chart, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric(f"n={n_for_point} 回：理論", f"{point_prob*100:.2f}%")
        # 参考：同nでの実測（1点）
        success = 0
        for _ in range(sim_repeats):
            trials = rng.random(n_for_point) < p
            if trials.any():
                success += 1
        emp = success / sim_repeats
        c2.metric(f"n={n_for_point} 回：実測（平均）", f"{emp*100:.2f}%")

    with st.expander("学習ポイント（要約）"):
        st.markdown(
            "- 各回の当選確率 p は一定でも、**“1回以上当たる”** 確率は回数 n が増えるほど上がる。\n"
            "- 理論式：**1 − (1 − p)^n**。n→大で1に近づくが、“必ず”ではない。"
        )

st.markdown("---")
st.caption("© 確率シミュレーション・ラボ｜高校生向け可視化教材（Streamlit）")
