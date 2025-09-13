# project_app.py
# 실행: streamlit run project_app.py

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis, shapiro, spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler

# ====== 스타일 & 한글 설정 ======
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'NanumBarunGothic'
plt.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False

# ====== 페이지 설정 ======
st.set_page_config(page_title="건설장비 RUL 탐색 분석 리포트", layout="wide")
st.title("건설장비 잔존수명(RUL) 데이터 탐색 분석 리포트")
st.caption("기업 보고서 수준의 설명과 시각화 · Streamlit 대시보드")

# ====== 사이드바 ======
st.sidebar.title("탐색 메뉴")
step = st.sidebar.radio(
    "분석 STEP 선택",
    options=[
        "개요 (ALL)",
        "STEP 1. 미션/비즈니스/데이터 이해",
        "STEP 2-1. 데이터 로드/구조 파악",
        "STEP 2-2. Y(목표) 및 분석 계획",
        "STEP 3-1. Y(잔존수명) 특성 분석",
        "STEP 3-2. X(설명변수) 특성 분석",
        "STEP 4. 현장 기반 가설 수립",
        "STEP 5-1. 탐색/분석 계획 재확인",
        "STEP 5-2. 단변량/이변량 시각화",
        "STEP 5-3. 2변수 조합 + risk_score",
        "STEP 6. 분석 정리/인사이트"
    ],
    index=0
)

# 보조 옵션
with st.sidebar.expander("표시 옵션"):
    show_sigma_classes = st.checkbox("표준편차(z-score) 기반 등급도 함께 표시", value=True)
    show_component_type_plots = st.checkbox("유형(Component_Type) 관련 시각화 표시", value=True)

# ====== 데이터 로드 ======
@st.cache_data
def load_data():
    df = pd.read_csv("./construction_machine_data.csv")
    # Arrow 변환 호환성: object → str 변환
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)
    return df

df = load_data()

# 공통 컬럼 변수
num_cols = ['Vibration', 'Temperature', 'Pressure', 'Operating_Hours', 'Remaining_Useful_Life']
x_cols  = ['Vibration', 'Temperature', 'Pressure', 'Operating_Hours']
y_col   = 'Remaining_Useful_Life'

# ====== STEP 1 ======
if step in ["개요 (ALL)", "STEP 1. 미션/비즈니스/데이터 이해"]:
    st.subheader("STEP 1. 미션/비즈니스/데이터 이해")
    st.markdown("""
**미션**  
- 건설장비 예지보전 시스템 개발을 위한 **데이터 탐색·전처리·시각화** 수행

**비즈니스 배경**  
- 월 평균 3~4건의 돌발 장비 고장 → 공사 지연·손실  
- 예방정비 과도 집행 → 불필요 비용 발생

**데이터 활용 목표**  
- 센서/운용 데이터를 바탕으로 **잔존유효수명(RUL)** 패턴 이해  
- 현장에서 즉시 활용 가능한 **인사이트 및 파생특성** 도출
""")
    st.info("데이터 파일: construction_machine_data.csv  |  컬럼: Component_ID, Component_Type, Vibration, Temperature, Pressure, Operating_Hours, Remaining_Useful_Life")

# ====== STEP 2-1 ======
if step in ["개요 (ALL)", "STEP 2-1. 데이터 로드/구조 파악"]:
    st.subheader("STEP 2-1. 데이터 로드 및 전체 구조 파악")
    col1, col2 = st.columns([1,1])
    with col1:
        st.write("데이터 크기 (행, 열):", df.shape)
        st.write("컬럼 목록:", df.columns.tolist())
        st.markdown("""
**컬럼 설명**
- Component_ID: 부품 고유 ID  
- Component_Type: 부품 유형 (Engine, Hydraulic Cylinder, Gear)  
- Vibration: 진동 수준(임의 단위)  
- Temperature: 작동 온도(℃)  
- Pressure: 가해지는 압력(psi)  
- Operating_Hours: 누적 작동 시간(hours)  
- Remaining_Useful_Life: 잔존 유효 수명(hours)
""")
    with col2:
        st.write("상위 5행 미리보기:")
        # DataFrame Arrow 변환 오류 방지를 위해 object 컬럼을 string으로 변환
        df_preview = df.head(5).astype({col: "string" for col in df.select_dtypes(include="object").columns})
        st.dataframe(df_preview, width="stretch")

    st.write("데이터 타입 정보:")
    # PyArrow 직렬화 오류 방지: dtype 객체를 문자열로 변환하여 표시
    dtypes_df = pd.DataFrame({'dtype': df.dtypes.astype(str)})
    st.dataframe(dtypes_df, width="stretch")
    st.write("결측치 개수:")
    st.write(df.isnull().sum())

    st.write("기본 통계(수치형):")
    st.write("기본 통계(수치형):")
    df_desc = df.describe().astype("float")
    st.dataframe(df_desc, width="stretch")

    st.write("범주 분포(Component_Type):")
    # Series → DataFrame 변환으로 Arrow 직렬화 안정화
    comp_counts = df['Component_Type'].value_counts().rename('count').to_frame()
    st.dataframe(comp_counts, width="content")
    st.caption("데이터의 구조와 품질(결측/타입/분포)을 먼저 확인하여 이후 분석의 신뢰도를 확보합니다.")

# ====== STEP 2-2 ======
if step in ["개요 (ALL)", "STEP 2-2. Y(목표) 및 분석 계획"]:
    st.subheader("STEP 2-2. 예측 대상(Y) 및 분석/전처리/시각화 계획")
    st.markdown("""
**대상(Y)**: Remaining_Useful_Life (RUL)

**분석 계획 요약**
- Y의 분포/형태/이상치 점검(기술통계, 정규성, IQR 기준 등)
- X(진동/온도/압력/누적시간, +유형) 특성: 범위/분포/이상치/상관/다변량 관계
- 고급 통계: 피어슨/스피어만 상관, 그룹 차이(박스플롯), 상호작용 확인
- **파생변수**: X만 활용하여 **잔존수명평가점수(risk_score)** 생성 및 5단계/3단계 **risk_class** 부여  
  - 가중치 = |Spearman(Y, X)| (단조 관계 강도), **부호 반전**(점수↑ ⇒ 위험↑ ⇒ RUL↓)  
  - 정보누수 방지: score 계산은 X만 사용
- 시각화: 테이블 + seaborn (hist/box/scatter/reg/heatmap/pair/joint/Facet 등)
""")

# ====== STEP 3-1 ======
if step in ["개요 (ALL)", "STEP 3-1. Y(잔존수명) 특성 분석"]:
    st.subheader("STEP 3-1. Y(잔존수명) 특성 분석")
    y = df[y_col]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RUL 최소~최대", f"{int(y.min())} ~ {int(y.max())}h")
    with col2:
        st.metric("평균 / 중앙값", f"{y.mean():.1f} / {y.median():.1f}h")
    with col3:
        st.metric("표준편차", f"{y.std():.1f}h")

    W, p_norm = shapiro(y)
    Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
    IQR = Q3 - Q1
    low_thr, high_thr = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outlier_cnt = ((y < low_thr) | (y > high_thr)).sum()

    st.write(f"정규성 검정(Shapiro): p-value={p_norm:.3f}")
    st.write(f"IQR 이상치 개수: {outlier_cnt} (하한={low_thr:.1f}, 상한={high_thr:.1f})")
    st.write(f"RUL ≤ 100h 개수: {(y<=100).sum()},  RUL ≥ 900h 개수: {(y>=900).sum()}")

    fig, axes = plt.subplots(1,2, figsize=(12,4))
    sns.histplot(y, kde=True, ax=axes[0])
    axes[0].set_title("RUL 분포")
    axes[0].set_xlabel("Remaining Useful Life (hours)")
    sns.boxplot(x=y, ax=axes[1])
    axes[1].set_title("RUL 박스플롯")
    axes[1].set_xlabel("Remaining Useful Life (hours)")
    st.pyplot(fig)
    st.caption("도표 목적: RUL의 분포 형태와 이상치 존재 여부를 확인하여 이후 모델링 가정의 타당성을 점검")

# ====== STEP 3-2 ======
if step in ["개요 (ALL)", "STEP 3-2. X(설명변수) 특성 분석"]:
    st.subheader("STEP 3-2. X(설명변수) 특성 분석: 분포/범위/상관")
    # 범위 요약
    st.markdown("**수치형 변수 범위 요약**")
    st.write(pd.DataFrame({
        c: [df[c].min(), df[c].max(), df[c].mean(), df[c].std()]
        for c in x_cols
    }, index=['min','max','mean','std']).T)

    # 분포
    fig, axes = plt.subplots(2, 2, figsize=(10,8))
    for ax, c in zip(axes.flatten(), x_cols):
        sns.histplot(df[c], kde=True, ax=ax)
        ax.set_title(f"{c} 분포")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("도표 목적: 각 센서/운용 변수의 분포 형태를 파악하여 이상 패턴 또는 왜도 확인")

    # 피어슨 상관행렬
    corr = df[num_cols].corr(numeric_only=True, method='pearson')
    st.write("피어슨 상관행렬:")
    fig2, ax2 = plt.subplots(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax2)
    ax2.set_title("상관행렬 (피어슨)")
    st.pyplot(fig2)
    st.caption("도표 목적: 변수 간 선형 상관 구조를 요약적으로 파악")

# ====== STEP 4 ======
if step in ["개요 (ALL)", "STEP 4. 현장 기반 가설 수립"]:
    st.subheader("STEP 4. 현장 기반 가설 수립 및 우선순위")
    st.markdown("""
**가설**
- H1(고우선): Operating_Hours ↑ → RUL ↓
- H2(고우선): Temperature/Pressure ↑ → RUL ↓
- H3(중우선): Component_Type에 따른 RUL 분포 차이 존재
- H4(중우선): Vibration ↑ & Operating_Hours ↑ 결합 시 RUL 급감 (상호작용)

**검증 기준**
- 비즈니스 임팩트(다운타임/비용 절감), 데이터 적합성(측정·관리 가능성), 통계적 근거
""")

# ====== STEP 5-1 ======
if step in ["개요 (ALL)", "STEP 5-1. 탐색/분석 계획 재확인"]:
    st.subheader("STEP 5-1. 탐색/분석 계획(업데이트)")
    st.markdown("""
- RUL(종속)과 각 X(연속/범주) 간 관계 시각화/통계 점검  
- 고급 통계: 피어슨/스피어만 상관, 분포/이상치, 그룹 차이  
- **파생변수**: X만 활용한 표준화-가중합 기반 **risk_score** 생성  
  - 가중치 = |Spearman(RUL, X)|, 부호 반전(점수↑=위험↑=RUL↓)  
  - 분위(5·3단계)와 **표준편차(z-score) 기반** 등급 동시 제공  
- 시각화: risk_score 분포·등급 분포, RUL~risk_class 박스플롯(단조성)  
""")

# ====== STEP 5-2 ======
if step in ["개요 (ALL)", "STEP 5-2. 단변량/이변량 시각화"]:
    st.subheader("STEP 5-2. 단변량/이변량 시각화")
    # 산점도 + 회귀선
    fig, axes = plt.subplots(2, 2, figsize=(12,10))
    pairs = [('Operating_Hours',y_col),
             ('Temperature',y_col),
             ('Pressure',y_col),
             ('Vibration',y_col)]
    for ax, (xv, yv) in zip(axes.flatten(), pairs):
        sns.regplot(x=xv, y=yv, data=df, ax=ax, scatter_kws={'alpha':0.5})
        ax.set_title(f"{xv} vs {yv}")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("도표 목적: 각 설명변수와 RUL의 1:1 관계(선형 경향)를 빠르게 점검")

    if show_component_type_plots:
        fig2, ax2 = plt.subplots(figsize=(7,4))
        sns.boxplot(x='Component_Type', y=y_col, data=df, ax=ax2)
        ax2.set_title("Component_Type별 RUL 분포")
        ax2.set_xlabel("Component_Type"); ax2.set_ylabel("RUL (hours)")
        st.pyplot(fig2)
        st.caption("도표 목적: 유형별 RUL 분포 차이를 시각적으로 비교")

        g = sns.pairplot(df, vars=['Vibration','Temperature','Pressure','Operating_Hours',y_col],
                         hue='Component_Type', corner=True, diag_kind='hist')
        g.fig.suptitle("특징 & RUL Pairplot (유형 구분)", y=1.02)
        st.pyplot(g.fig)
        st.caption("도표 목적: 다변량 관계를 개괄적으로 살피고 유형별 패턴을 식별")

# ====== STEP 5-3 ======
if step in ["개요 (ALL)", "STEP 5-3. 2변수 조합 + risk_score"]:
    st.subheader("STEP 5-3. 2변수 조합 시각화 + risk_score 생성/등급화")

    # 예시 조합 산점도
    fig0, ax0 = plt.subplots(figsize=(6,5))
    sns.scatterplot(data=df, x='Temperature', y='Vibration', hue='Component_Type', alpha=0.7, ax=ax0)
    ax0.set_title("Temperature vs Vibration (Component_Type별)")
    st.pyplot(fig0)
    st.caption("도표 목적: 두 변수 조합에서 유형별 클러스터링·패턴 확인")

    # risk_score 생성 (Spearman 기반 가중치, 부호 반전)
    spearman_vals = {}
    for c in x_cols:
        rho, pval = spearmanr(df[c], df[y_col])
        spearman_vals[c] = {'rho': rho, 'pval': pval}

    abs_rhos = np.array([abs(spearman_vals[c]['rho']) for c in x_cols])
    weights = np.ones_like(abs_rhos) / len(abs_rhos) if abs_rhos.sum()==0 else abs_rhos / abs_rhos.sum()
    signs = np.array([-np.sign(spearman_vals[c]['rho']) for c in x_cols])  # 점수↑=위험↑=RUL↓ 되도록

    scaler = StandardScaler()
    Xz = scaler.fit_transform(df[x_cols])
    risk_score = (Xz * (weights * signs)).sum(axis=1)
    df['risk_score'] = risk_score

    st.markdown("**risk_score 생성 요약(가중치=|Spearman|, 부호 반전)**")
    st.write(pd.DataFrame({
        'feature': x_cols,
        '|rho|': [abs(spearman_vals[c]['rho']) for c in x_cols],
        'rho_sign': [np.sign(spearman_vals[c]['rho']) for c in x_cols],
        'sign(inverted)': list(map(int, signs)),
        'weight': list(weights)
    }))

    # 5단계/3단계 분위 기반 등급
    q5 = np.percentile(df['risk_score'], [20,40,60,80])
    def to_5class(v):
        if v <= q5[0]: return '매우 낮음'
        elif v <= q5[1]: return '낮음'
        elif v <= q5[2]: return '보통'
        elif v <= q5[3]: return '높음'
        else: return '매우 높음'
    df['risk_class_5'] = df['risk_score'].apply(to_5class)

    q3 = np.percentile(df['risk_score'], [33.33, 66.67])
    def to_3class(v):
        if v <= q3[0]: return '낮음'
        elif v <= q3[1]: return '보통'
        else: return '높음'
    df['risk_class_3'] = df['risk_score'].apply(to_3class)

    st.write("**리스크 등급 분포 (분위 기반)**")
    c1, c2 = st.columns(2)
    with c1:
        st.write(df['risk_class_5'].value_counts().reindex(['매우 낮음','낮음','보통','높음','매우 높음']))
        fig1, ax1 = plt.subplots(figsize=(7,3.8))
        sns.countplot(x='risk_class_5', data=df, order=['매우 낮음','낮음','보통','높음','매우 높음'], ax=ax1)
        ax1.set_title("risk_class(5단계) 분포")
        st.pyplot(fig1)
    with c2:
        st.write(df['risk_class_3'].value_counts().reindex(['낮음','보통','높음']))
        fig2, ax2 = plt.subplots(figsize=(6,3.6))
        sns.countplot(x='risk_class_3', data=df, order=['낮음','보통','높음'], ax=ax2)
        ax2.set_title("risk_class(3단계) 분포")
        st.pyplot(fig2)
    st.caption("도표 목적: 분위 기반 등급은 원리상 구간별 개수가 유사하게 배분됩니다(표본 1000 → 각 200).")

    # z-score 기반 등급(옵션)
    if show_sigma_classes:
        mu = df['risk_score'].mean()
        sigma = df['risk_score'].std(ddof=0) if df['risk_score'].std(ddof=0) > 0 else 1e-9

        def to_5class_sigma(v):
            z = (v - mu) / sigma
            if z <= -1.0:   return '매우 낮음'
            elif z <= -0.5: return '낮음'
            elif z <=  0.5: return '보통'
            elif z <=  1.0: return '높음'
            else:           return '매우 높음'

        def to_3class_sigma(v):
            z = (v - mu) / sigma
            if z <= -0.5:   return '낮음'
            elif z <=  0.5: return '보통'
            else:           return '높음'

        df['risk_class_5_sigma'] = df['risk_score'].apply(to_5class_sigma)
        df['risk_class_3_sigma'] = df['risk_score'].apply(to_3class_sigma)

        st.write("**리스크 등급 분포 (표준편차 기반)**")
        c3, c4 = st.columns(2)
        with c3:
            st.write(df['risk_class_5_sigma'].value_counts().reindex(['매우 낮음','낮음','보통','높음','매우 높음']))
            fig3, ax3 = plt.subplots(figsize=(7,3.8))
            sns.countplot(x='risk_class_5_sigma', data=df, order=['매우 낮음','낮음','보통','높음','매우 높음'], ax=ax3)
            ax3.set_title("risk_class_5 (표준편차 기반) 분포")
            st.pyplot(fig3)
        with c4:
            st.write(df['risk_class_3_sigma'].value_counts().reindex(['낮음','보통','높음']))
            fig4, ax4 = plt.subplots(figsize=(6,3.6))
            sns.countplot(x='risk_class_3_sigma', data=df, order=['낮음','보통','높음'], ax=ax4)
            ax4.set_title("risk_class_3 (표준편차 기반) 분포")
            st.pyplot(fig4)
        st.caption("도표 목적: 데이터의 실제 분포(평균±표준편차)를 반영한 자연스러운 등급 분포 확인")

    # risk_score 분포
    fig5, ax5 = plt.subplots(figsize=(6,4))
    sns.histplot(df['risk_score'], kde=True, ax=ax5)
    ax5.set_title("risk_score 분포 (높을수록 위험↑ / RUL↓ 경향)")
    ax5.set_xlabel("risk_score")
    st.pyplot(fig5)
    st.caption("도표 목적: 조합 지표 risk_score의 전반적 분포 확인")

    # RUL vs risk_score (joint, reg)
    g = sns.jointplot(data=df, x='risk_score', y=y_col, kind='scatter', height=5, space=0)
    g.ax_joint.set_xlabel("risk_score (높을수록 위험↑)")
    g.ax_joint.set_ylabel("RUL (hours)")
    g.fig.suptitle("RUL vs risk_score (jointplot)", y=1.02)
    st.pyplot(g.fig)
    st.caption("도표 목적: risk_score와 RUL의 단조/비선형 관계 및 분포 특성 확인")

    fig6, ax6 = plt.subplots(figsize=(6,4))
    sns.regplot(data=df, x='risk_score', y=y_col, scatter_kws={'alpha':0.5}, ax=ax6)
    ax6.set_title("RUL vs risk_score (regplot)")
    ax6.set_xlabel("risk_score (높을수록 위험↑)"); ax6.set_ylabel("RUL (hours)")
    st.pyplot(fig6)
    st.caption("도표 목적: risk_score가 증가할수록 RUL이 감소하는 경향(단조성) 점검")

    pr, pp = pearsonr(df['risk_score'], df[y_col])
    sr, sp = spearmanr(df['risk_score'], df[y_col])
    st.write(f"[검증] risk_score ↔ RUL 상관  |  Pearson={pr:.3f} (p={pp:.3f}),  Spearman={sr:.3f} (p={sp:.3f})")

    # 등급별 RUL 분포 (분위 & z-score)
    fig7, ax7 = plt.subplots(figsize=(7,4))
    order5 = ['매우 낮음','낮음','보통','높음','매우 높음']
    sns.boxplot(data=df, x='risk_class_5', y=y_col, order=order5, ax=ax7)
    ax7.set_title("RUL vs risk_class(5단계, 분위 기반)")
    ax7.set_xlabel("risk_class_5"); ax7.set_ylabel("RUL (hours)")
    st.pyplot(fig7)

    if show_sigma_classes:
        fig8, ax8 = plt.subplots(figsize=(7,4))
        sns.boxplot(data=df, x='risk_class_5_sigma', y=y_col, order=order5, ax=ax8)
        ax8.set_title("RUL vs risk_class_5_sigma (표준편차 기반)")
        ax8.set_xlabel("risk_class_5_sigma"); ax8.set_ylabel("RUL (hours)")
        st.pyplot(fig8)

    # 보조: 상관 히트맵
    corr_aux = df[['risk_score'] + x_cols + [y_col]].corr(numeric_only=True)
    fig9, ax9 = plt.subplots(figsize=(6,5))
    sns.heatmap(corr_aux, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax9)
    ax9.set_title("risk_score 및 특징과 RUL 상관")
    st.pyplot(fig9)
    st.caption("도표 목적: 조합 지표(risk_score)를 포함한 변수군과 RUL의 상관 구조 요약")

    # 상호작용 예시: 고사용량∩고진동
    high_usage = df['Operating_Hours'] > df['Operating_Hours'].quantile(0.9)
    high_vib   = df['Vibration'] > df['Vibration'].quantile(0.9)
    mean_rul_combo = df[high_usage & high_vib][y_col].mean()
    mean_rul_else  = df[~(high_usage & high_vib)][y_col].mean()
    st.write(f"[상호작용 체크] 상위 10% 고사용량∩고진동 평균 RUL: {mean_rul_combo:.1f} | 그 외: {mean_rul_else:.1f}")

    fig10, ax10 = plt.subplots(figsize=(6,4))
    sns.scatterplot(x='Operating_Hours', y=y_col, hue=high_vib.map({True:'고진동',False:'일반'}),
                    data=df, alpha=0.7, ax=ax10)
    ax10.set_title("Operating_Hours vs RUL (진동 수준별)")
    ax10.set_xlabel("Operating_Hours"); ax10.set_ylabel("RUL (hours)")
    ax10.legend(title='진동 수준')
    st.pyplot(fig10)
    st.caption("도표 목적: 고사용량·고진동 조합에서 RUL 저하가 나타나는지 시각적으로 확인")

# ====== STEP 6 ======
if step in ["개요 (ALL)", "STEP 6. 분석 정리/인사이트"]:
    st.subheader("STEP 6. 분석 정리 및 인사이트")
    st.markdown("""
**주요 관찰**
- 단일 변수와 RUL의 강한 선형 관계는 약함(피어슨 상관 ≈ 0)
- 고온/고압일수록 RUL이 다소 낮아지는 경향(약한 음의 관계)
- Component_Type 중 **Gear**가 다소 낮은 RUL 경향 가능(겹침 큼)
- **Vibration 단독** 효과는 약하나 **고사용량과 결합** 시 RUL 저하 신호 강화
- **risk_score**는 RUL과 단조 관계를 보일 가능성 → 유지보수 우선순위/경보 기준에 유용

**실무적 시사점**
1) 복합 스트레스(고사용량∩고진동/고온/고압) 구간에 대한 집중 점검/교체 전략  
2) risk_class 상위 등급(‘높음’, ‘매우 높음’) 비율 모니터링 및 알람  
3) 유형별 취약 패턴(예: Gear)에 대한 조건부 유지보수 정책

**다음 단계 제안**
- 특징 엔지니어링: 상호작용/비선형(제곱·로그), 누적·극단 빈도, 이동통계량
- 지도학습: RUL 회귀 + risk_class 분류의 멀티태스킹/앙상블
- 운영 룰: risk_class 상위 구간·고사용량∩고진동 구간 우선 점검/교체
""")

st.success("대시보드 준비 완료: 사이드바에서 STEP을 선택해 각 섹션의 분석 결과와 차트를 확인하세요.")


