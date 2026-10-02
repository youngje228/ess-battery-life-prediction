"""데이터 로드 및 정제

- 원본: MIT-Stanford Battery Dataset (Kaggle errorcorrect 버전, MATLAB v7.3 = HDF5)
- 원본 로드는 루트의 `load_data.py` (h5py 선택 로드 → data/cells_raw.pkl). 캐시가 없으면 여기서 같은 방식으로 만든다
- 정제 기준은 DAY 1 EDA에서 확정한 팀 기준을 따른다 (README '데이터 정제' 참고)
"""
from pathlib import Path
import os
import pickle

import numpy as np
import pandas as pd
from scipy.signal import medfilt

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.path.expanduser(os.environ.get('MAT_DIR', '~/Downloads/archive')))   # load_data.py와 같은 규칙
CACHE = ROOT / 'data' / 'cells_raw.pkl'

FILES = {'b1': '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
         'b2': '2018-02-20_batchdata_updated_struct_errorcorrect.mat',
         'b3': '2018-04-12_batchdata_updated_struct_errorcorrect.mat'}
EOL_AH = 0.88            # 공칭 1.1Ah × 80% (SOH 80%)

# 원논문 코드의 b1 c0~c4 수명 보정값 (원논문 2017-06-30 배치로 이어진 실험이라고 보고 더한 사이클 수, 이번 데이터로는 검증 불가)
# 이번 데이터 버전의 b2에는 이어진 셀이 없어(barcode 대조) 검증 불가 → 기본 제외, 민감도 분석에만 사용
PAPER_B1_ADD = {'b1c0': 662, 'b1c1': 981, 'b1c2': 1060, 'b1c3': 208, 'b1c4': 482}
# 원논문 b3 노이즈 채널 — 이번 데이터의 스파이크 점검에는 걸리지 않아 기본 유지, 민감도 분석에서만 제거
PAPER_B3_NOISY = ['b3c2', 'b3c37', 'b3c42', 'b3c43']


def load_raw(force=False):
    """원본 .mat → 셀 리스트 dict {'b1': [...], 'b2': [...], 'b3': [...]}

    캐시(data/cells_raw.pkl)가 있으면 읽고, 없으면 루트의 load_data.py로 만든다 (로더는 한 곳에서만 관리)"""
    if CACHE.exists() and not force:
        with open(CACHE, 'rb') as fp:
            return pickle.load(fp)
    import sys
    sys.path.insert(0, str(ROOT))
    import load_data
    out = {t: load_data.load_batch(t, str(RAW_DIR / fn)) for t, fn in FILES.items()}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, 'wb') as fp:
        pickle.dump(out, fp)
    return out


def smooth_qd(qd):
    """단발성 QD 스파이크 보정 (median filter k=5). index 0 (b1의 cycle 1 더미)은 그대로 둔다"""
    out = qd.astype(float).copy()
    out[1:] = medfilt(qd[1:], 5)
    return out


def removal_reason(c):
    """셀 제거 사유 ('' = 사용)"""
    if 'VarCharge' in c['policy'] or 'SLOWCYCLE' in c['policy']:
        return '다른 실험 프로토콜 (cycle_life 없음)'
    qd_s = smooth_qd(c['summary']['QDischarge'])
    reached = (qd_s[1:] <= EOL_AH + 0.0035).any()      # 기록은 0.880~0.881Ah에서 종료되는 셀이 많음
    if not np.isfinite(c['cycle_life_raw']) or not reached:
        return 'EOL 미도달 (수명 미관측)'
    return ''


def get_cells(include_b1_corrected=False, drop_b3_paper_noisy=False):
    """정제된 셀 리스트와 정제 로그 반환

    include_b1_corrected=True 이면 b1 c0~c4를 원논문 보정 수명으로 포함 (민감도 분석용)
    drop_b3_paper_noisy=True 이면 원논문 b3 노이즈 채널 4셀 제거 (민감도 분석용)
    """
    raw = load_raw()
    cells, log = [], []
    for tag in ['b1', 'b2', 'b3']:
        for c in raw[tag]:
            c = dict(c)
            c['qd_s'] = smooth_qd(c['summary']['QDischarge'])
            reason = removal_reason(c)
            if include_b1_corrected and c['key'] in PAPER_B1_ADD:
                c['life'] = c['cycle_life_raw'] + PAPER_B1_ADD[c['key']]
                c['life_corrected'] = True
                reason = ''
            else:
                c['life'] = c['cycle_life_raw']
                c['life_corrected'] = False
            if drop_b3_paper_noisy and c['key'] in PAPER_B3_NOISY:
                reason = '원논문 노이즈 채널 (민감도 분석)'
            log.append(dict(batch=tag, key=c['key'], policy=c['policy'], cycle_life_raw=c['cycle_life_raw'],
                            QD_last=round(float(c['summary']['QDischarge'][-1]), 3), removed=reason != '', reason=reason))
            if reason == '':
                cells.append(c)
    return cells, pd.DataFrame(log)


if __name__ == '__main__':
    cells, log = get_cells()
    print(log.groupby('batch').agg(original=('key', 'size'), removed=('removed', 'sum')).assign(
        final=lambda d: d.original - d.removed))
    print(log[log.removed][['batch', 'key', 'policy', 'QD_last', 'reason']].to_string(index=False))
