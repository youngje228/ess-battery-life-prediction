"""모델 학습 · 선택 · 평가 파이프라인 (notebooks/02, 03과 같은 규칙을 스크립트로 재현)

실행 :  python src/train.py          (data/cells_raw.pkl 필요 → load_data.py 먼저 실행)
출력 :  results/model_performance.csv  (과제 리포팅 포맷, Batch 3 포함)
        results/model_selection_20seeds.csv, results/residual_by_structure.csv, results/top_error_cells.csv,
        results/sensitivity.csv, results/dq_voltage_window.csv, results/offset_calibration.csv,
        results/test_predictions.csv

설계 (DAY 1 모델링 전략)
- 타깃 log10(cycle_life), 평가 지표는 역변환 후 MAPE (%)
- Valid : Batch 1 안에서 '충전 정책' 그룹 단위 hold-out 20% (GroupShuffleSplit), seed 0~19 반복
- Train : hold-out을 뺀 Batch 1에서 정책 단위 GroupKFold(5) CV 평균 (하이퍼파라미터 탐색 포함)
- Test  : 선택된 설정으로 Batch 1 전체 재학습 → Batch 2 (필수), Batch 3 (추가)
- 선택  : 피처셋 A·B × 모델 7종 중 (CV + Valid)/2 의 20 seed 평균이 최소인 후보,
          1위와의 차이가 1위의 표준편차보다 작으면 피처 수가 적고 단순한 모델. Test는 선택에 쓰지 않는다.
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURE_SETS, build_feature_table  # noqa: E402
from preprocess import get_cells                           # noqa: E402

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
TARGET_MAPE = 9.1           # 원논문 Regression 테스트 오차 (%)
N_SEEDS = 20
SELECTABLE = ['A', 'B']     # 셋 C는 배치마다 부호가 뒤집히는 피처 포함 → 가설 검증용으로만 사용

# name : (estimator, param grid, 복잡도 순위)  — notebooks/03_modeling.ipynb 와 동일
MODELS = {
    'Linear':     (LinearRegression(), {}, 0),
    'Ridge':      (Ridge(), {'m__alpha': np.logspace(-3, 2, 11)}, 1),
    'Lasso':      (Lasso(max_iter=100000), {'m__alpha': np.logspace(-4, -0.5, 8)}, 1),
    'ElasticNet': (ElasticNet(max_iter=100000), {'m__alpha': np.logspace(-4, 0, 9), 'm__l1_ratio': [.1, .5, .9]}, 2),
    'GPR':        (GaussianProcessRegressor(ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e2))
                                            + WhiteKernel(1e-2, (1e-3, 1.0)),
                                            normalize_y=True, n_restarts_optimizer=3, random_state=0), {}, 3),
    'SVR':        (SVR(), {'m__C': [0.3, 1, 3, 10], 'm__gamma': ['scale', 0.1, 1], 'm__epsilon': [0.01, 0.03]}, 3),
    'RF':         (RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1),
                   {'m__max_depth': [2, 3], 'm__min_samples_leaf': [2, 4]}, 4),
}


def mape(y_log, p_log):
    """log10 값을 역변환한 뒤 MAPE (%)"""
    y, p = 10 ** np.asarray(y_log, float), 10 ** np.asarray(p_log, float)
    return float(np.mean(np.abs(p - y) / y) * 100)


SCORER = make_scorer(lambda y, p: -mape(y, p))


def tune(df, feats, name):
    """정책 단위 GroupKFold로 하이퍼파라미터 탐색 → df 전체로 재학습된 best 모델"""
    est, grid, _ = MODELS[name]
    pipe = Pipeline([('s', StandardScaler()), ('m', clone(est))])
    gs = GridSearchCV(pipe, grid or [{}], cv=GroupKFold(min(5, df.policy.nunique())), scoring=SCORER, n_jobs=-1)
    gs.fit(df[feats], df.log_life, groups=df.policy)
    return gs


def make_splits(TR, n_seeds=N_SEEDS):
    out = []
    for seed in range(n_seeds):
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(TR, groups=TR.policy))
        assert not set(TR.policy.iloc[tr]) & set(TR.policy.iloc[va])      # 같은 정책이 train/valid에 나뉘지 않음
        out.append((seed, TR.iloc[tr], TR.iloc[va]))
    return out


def run_selection(TR, splits):
    rows = []
    for seed, tr, va in splits:
        for s, feats in FEATURE_SETS.items():
            for name in MODELS:
                gs = tune(tr, feats, name)
                rows.append(dict(seed=seed, set=s, model=name, n_feat=len(feats), cv=-gs.best_score_,
                                 valid=mape(va.log_life, gs.predict(va[feats])), params=str(gs.best_params_)))
    R = pd.DataFrame(rows)
    R['score'] = (R.cv + R.valid) / 2
    S = R.groupby(['set', 'model']).agg(n_feat=('n_feat', 'first'), cv_mean=('cv', 'mean'), cv_std=('cv', 'std'),
                                        valid_mean=('valid', 'mean'), valid_std=('valid', 'std'),
                                        score_mean=('score', 'mean'), score_std=('score', 'std'))
    return R, S


def select(S):
    cand = S.loc[SELECTABLE].reset_index().sort_values('score_mean').reset_index(drop=True)
    best = cand.iloc[0]
    near = cand[cand.score_mean - best.score_mean < best.score_std].copy()
    near['complexity'] = near.model.map({k: v[2] for k, v in MODELS.items()})
    return near.sort_values(['n_feat', 'complexity', 'score_mean']).iloc[0]


def report_table(sel, test_b2, test_b3):
    """과제 리포팅 포맷 (노션 'Reporting format for Batch 3' 구조). Gap = 뒤 − 앞, (+) = 오차 증가"""
    sel = sel.round(2)        # 노트북과 같이 반올림한 평균으로 Gap 계산
    test_b2, test_b3 = round(test_b2, 2), round(test_b3, 2)
    rows = [('Train (Batch 1 CV)', '', sel.cv_mean, f'20 seed 평균 ± {sel.cv_std:.2f}'),
            ('Valid (Batch 1 Hold-out)', '', sel.valid_mean, f'20 seed 평균 ± {sel.valid_std:.2f}'),
            ('Test (Batch 2)', '', test_b2, 'Batch 1 전체(36셀)로 재학습'),
            ('', 'Gap (Train-Valid)', sel.valid_mean - sel.cv_mean, '(+) : 과적합 의심'),
            ('', 'Gap (Valid-Test)', test_b2 - sel.valid_mean, '(+) : 배치간 일반화 저하 의심'),
            ('', 'Gap (Target-Test)', test_b2 - TARGET_MAPE, f'Target : 원논문 {TARGET_MAPE}%'),
            ('Test (Batch 3)', '', test_b3, ''),
            ('', 'Gap (Batch2-Batch3)', test_b3 - test_b2, 'Test 성능 간 비교 (−) : Batch 3가 더 잘 맞음'),
            ('', 'Gap (Target-Test)', test_b3 - TARGET_MAPE, 'Batch 3 기준, 원논문 성능 비교')]
    out = pd.DataFrame(rows, columns=['구분', 'Gap', 'MAPE (%)', '비고'])
    out['MAPE (%)'] = out['MAPE (%)'].round(2)
    return out


def offset_calibration(d, ks=(1, 3, 5, 10), n_draw=500, seed=0):
    """(추가 분석) 새 배치에서 수명이 확인된 셀 k개로 절편만 보정했을 때 나머지 셀의 MAPE.
    테스트 배치의 실제 수명 일부를 쓰는 운영 시나리오 분석이므로 본 성능표와 분리해서 보고한다."""
    rng = np.random.default_rng(seed)
    y, p = d.log_life.values, d.pred_log.values
    rows = []
    for k in ks:
        m = []
        for _ in range(n_draw):
            idx = rng.choice(len(y), k, replace=False)
            rest = np.setdiff1d(np.arange(len(y)), idx)
            m.append(mape(y[rest], p[rest] + np.mean(y[idx] - p[idx])))
        rows.append(dict(k_cells=k, mape_mean=np.mean(m), mape_std=np.std(m)))
    return pd.DataFrame(rows)


def dq_window_check(cells, F, windows=(('2.0~3.5V (기본, 전체)', 3.5), ('2.0~3.1V', 3.1), ('2.0~3.0V', 3.0))):
    """(검증) Batch 3 경고 'Qdlin 시작점이 배치별로 달라 단순 비교 시 왜곡' 대응.
    ΔQ 계산 전압 구간의 상단을 잘라 dQ_logvar를 다시 만들고, 셋 A · Linear로 같은 방식(b1 전체 재학습) 평가.
    피처 정의는 DAY 1에서 고정했으므로 선택에는 쓰지 않는 사후 확인이다."""
    V = np.linspace(3.5, 2.0, 1000)                         # Qdlin 전압 격자 (3.5 → 2.0V)
    G = F.set_index('key')
    rows = []
    for name, vmax in windows:
        m = V <= vmax + 1e-9
        G['w'] = pd.Series({c['key']: np.log10(np.var((c['qdlin'][99] - c['qdlin'][9])[m])) for c in cells})
        tr, b2, b3 = [G[G.batch == b].reset_index() for b in ('b1', 'b2', 'b3')]
        gs = tune(tr, ['w'], 'Linear')
        p3 = gs.predict(b3[['w']])
        rows.append({'dQ_window': name, **{f'r_{b}': np.corrcoef(d.w, d.log_life)[0, 1] for b, d in (('b1', tr), ('b2', b2), ('b3', b3))},
                     'train_cv_b1_all': -gs.best_score_, 'test_b2': mape(b2.log_life, gs.predict(b2[['w']])),
                     'test_b3': mape(b3.log_life, p3), 'bias_b3_pct': np.mean((10 ** p3 - b3.life) / b3.life * 100)})
    return pd.DataFrame(rows).round(2)


def main():
    RESULTS.mkdir(exist_ok=True)
    cells, _ = get_cells()
    F = build_feature_table(cells)
    TR, B2, B3 = [F[F.batch == b].reset_index(drop=True) for b in ['b1', 'b2', 'b3']]

    # 1) 20 seed × 피처셋 3 × 모델 7 = 420회 학습 → 선택
    R, S = run_selection(TR, make_splits(TR))
    S.round(3).to_csv(RESULTS / 'model_selection_20seeds.csv')
    sel = select(S)
    sel_row = S.loc[(sel.set, sel.model)]
    feats = FEATURE_SETS[sel.set]
    print(S.sort_values('score_mean').round(2).to_string())
    print(f'\n선택 : 셋 {sel.set} {feats} · {sel.model}')

    # 2) 최종 모델 : Batch 1 전체로 재학습 → Test
    final = tune(TR, feats, sel.model)
    for d in (B2, B3):
        d['pred_log'] = final.predict(d[feats])
        d['pred'] = 10 ** d.pred_log
        d['err_pct'] = (d.pred - d.life) / d.life * 100
    test_b2, test_b3 = mape(B2.log_life, B2.pred_log), mape(B3.log_life, B3.pred_log)
    rep = report_table(sel_row, test_b2, test_b3)
    rep.to_csv(RESULTS / 'model_performance.csv', index=False, encoding='utf-8-sig')
    print(rep.to_string(index=False))

    T = pd.concat([B2, B3], ignore_index=True)
    T[['batch', 'key', 'policy', 'newstruct', 'life', 'pred', 'err_pct', 'dQ_logvar']].round(3) \
        .to_csv(RESULTS / 'test_predictions.csv', index=False)

    # 3) 오차 구조 : 무작위 오차인가, 일괄 편향인가
    T['group'] = T.batch + ' · ' + np.where(T.newstruct == 1, 'newstructure', 'normal')
    res = T.groupby('group').agg(n=('err_pct', 'size'), bias_pct=('err_pct', 'mean'),
                                 MAPE=('err_pct', lambda x: x.abs().mean()),
                                 over_pred_share=('err_pct', lambda x: (x > 0).mean() * 100)).round(1)
    res.to_csv(RESULTS / 'residual_by_structure.csv')
    print(res)

    # 4) 가장 크게 틀린 셀
    lo, hi = TR.life.min(), TR.life.max()
    T['life_vs_b1'] = np.where(T.life < lo, 'b1 최소 미만', np.where(T.life > hi, 'b1 최대 초과', 'b1 범위 안'))
    top = T.reindex(T.err_pct.abs().sort_values(ascending=False).index).head(12)
    top[['batch', 'key', 'policy', 'newstruct', 'life', 'pred', 'err_pct', 'life_vs_b1', 'dQ_logvar']].round(2) \
        .to_csv(RESULTS / 'top_error_cells.csv', index=False, encoding='utf-8-sig')
    print(top[['batch', 'key', 'policy', 'life', 'pred', 'err_pct', 'life_vs_b1']].round(1).to_string(index=False))

    # 5) 민감도 : b1 c0~c4 보정 수명 포함 / b3 원논문 노이즈 채널 제거
    sens = [dict(setting='기본 (b1 36셀, b3 44셀)', test_b2=test_b2, test_b3=test_b3)]
    cs, _ = get_cells(include_b1_corrected=True)
    Fs = build_feature_table(cs)
    ms = tune(Fs[Fs.batch == 'b1'].reset_index(drop=True), feats, sel.model)
    sens.append(dict(setting='b1 c0~c4 원논문 보정 수명 포함 (b1 41셀)', test_b2=mape(B2.log_life, ms.predict(B2[feats])),
                     test_b3=mape(B3.log_life, ms.predict(B3[feats]))))
    B3n = B3[~B3.key.isin(['b3c2', 'b3c37', 'b3c42', 'b3c43'])]
    sens.append(dict(setting='b3 원논문 노이즈 채널 4셀 제거 (b3 40셀)', test_b2=test_b2,
                     test_b3=mape(B3n.log_life, B3n.pred_log)))
    pd.DataFrame(sens).round(2).to_csv(RESULTS / 'sensitivity.csv', index=False, encoding='utf-8-sig')

    # 5-2) Batch 3 경고 대응 : ΔQ 전압 구간 제한 (Qdlin 시작점 왜곡 확인)
    dqw = dq_window_check(cells, F)
    dqw.to_csv(RESULTS / 'dq_voltage_window.csv', index=False, encoding='utf-8-sig')
    print(dqw.to_string(index=False))

    # 6) (추가) 배치별 소량 재보정
    cal = pd.concat([offset_calibration(d).assign(batch=b) for b, d in (('b2', B2), ('b3', B3))]).round(2)
    cal.to_csv(RESULTS / 'offset_calibration.csv', index=False)
    print(cal.to_string(index=False))
    return F, R, S, sel, final, T, cal


if __name__ == '__main__':
    main()
