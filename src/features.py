"""피처 생성 — cycle 2~100 정보만 사용 (예측 시점 이후 정보 누수 방지)

인덱스 규칙: summary / Qdlin 배열 index j = cycle − 1  →  cycle 10 = j9, cycle 100 = j99
"""
import re

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# DAY 1 Feature Selection 결과 (README '피처' 참고)
FEATURE_SETS = {
    'A': ['dQ_logvar'],                                            # 원논문 variance 모델 (베이스라인)
    'B': ['dQ_logvar', 'QD_slope_91_100'],                         # 배치 일관성 기준 통과 피처 (주력 후보)
    'C': ['dQ_logvar', 'QD_slope_91_100', 'dQ_logmin', 'dQ_mean',
          'QD_2', 'QD_slope_2_100', 'QD_icpt_2_100'],              # 원논문 discharge 계열 재현 (가설 검증용, 선택 대상 제외)
}

_POLICY_RE = re.compile(r'([\d.]+)C\(([\d.]+)%\)-([\d.]+)C')


def parse_policy(p):
    """충전 정책 문자열 → C1, Q1, C2, 0→80% SOC 평균 C-rate"""
    c1, q1, c2 = map(float, _POLICY_RE.search(p).groups())
    t = min(q1, 80) / 100 / c1 + max(0, 80 - q1) / 100 / c2
    return dict(C1=c1, Q1=q1, C2=c2, avgC_80=0.8 / t, newstruct=int('newstructure' in p))


def delta_q(c, hi=100, lo=10):
    """ΔQ(V) = Q_hi(V) − Q_lo(V)  (Ah, 1000 points on 3.5→2.0V)"""
    return c['qdlin'][hi - 1] - c['qdlin'][lo - 1]


def make_features(c):
    s = c['summary']
    dq = delta_q(c)
    cyc = np.arange(2, 101).astype(float)
    qd = c['qd_s'][1:100]                                   # cycle 2..100 (median-filtered)
    ir = s['IR'][1:100]
    ir = np.where(ir > 0, ir, np.nan)                       # IR = 0 은 측정 누락
    p_all = np.polyfit(cyc, qd, 1)
    p_last = np.polyfit(cyc[-10:], qd[-10:], 1)
    f = dict(batch=c['batch'], key=c['key'], policy=c['policy'], life=c['life'], log_life=np.log10(c['life']),
             life_corrected=c.get('life_corrected', False),
             # ΔQ(V)
             dQ_logvar=np.log10(np.var(dq)), dQ_logmin=np.log10(abs(dq.min())), dQ_mean=dq.mean() * 1000,
             dQ_skew=skew(dq), dQ_kurt=np.log10(abs(kurtosis(dq))),
             # 용량
             QD_2=qd[0], QD_100_m10=(c['qd_s'][99] - c['qd_s'][9]) * 1000,
             QD_slope_2_100=p_all[0] * 1e6, QD_icpt_2_100=p_all[1], QD_slope_91_100=p_last[0] * 1e6,
             # 저항 / 온도 / 충전
             IR_2=ir[0], IR_min=np.nanmin(ir) if np.isfinite(ir).any() else np.nan, IR_100_m2=(ir[-1] - ir[0]) * 1e4,
             Tavg_mean=np.nanmean(s['Tavg'][1:100]), Tmax_max=np.nanmax(s['Tmax'][1:100]),
             Trange_mean=np.nanmean(s['Tmax'][1:100] - s['Tmin'][1:100]),
             chargetime_2_6=np.nanmean(s['chargetime'][1:6]))
    f.update(parse_policy(c['policy']))
    return f


def build_feature_table(cells):
    return pd.DataFrame([make_features(c) for c in cells])


if __name__ == '__main__':
    from preprocess import get_cells
    cells, _ = get_cells()
    F = build_feature_table(cells)
    print(F.groupby('batch').size())
    print(F[FEATURE_SETS['C']].describe().round(3))
