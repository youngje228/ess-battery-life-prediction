"""원본 .mat(HDF5) → data/cells_raw.pkl

h5py로 셀별 summary 전체와 cycle 1~120의 Qdlin만 골라 읽는다 (전체 로드는 메모리 부족).
인덱스 규칙: summary / cycles 배열의 index j = cycle - 1 (j=0의 cycle 필드 = 1)
원본 위치는 환경변수 MAT_DIR로 바꿀 수 있다 (기본 ~/Downloads/archive)
"""
import os, pickle
import numpy as np
import h5py

DATA_DIR = os.path.expanduser(os.environ.get('MAT_DIR', '~/Downloads/archive'))
FILES = {
    'b1': '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    'b2': '2018-02-20_batchdata_updated_struct_errorcorrect.mat',
    'b3': '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
}
SUMMARY_KEYS = ['QDischarge', 'QCharge', 'IR', 'Tmax', 'Tavg', 'Tmin', 'chargetime', 'cycle']
N_QDLIN = 120
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'cells_raw.pkl')


def read_str(f, ref):
    return f[ref][()].tobytes()[::2].decode()


def read_vec(f, ref):
    return np.ravel(f[ref][()]).astype(float)


def load_batch(tag, path):
    cells = []
    with h5py.File(path, 'r') as f:
        b = f['batch']
        n = b['summary'].shape[0]
        for i in range(n):
            s = f[b['summary'][i, 0]]
            summary = {k: np.ravel(s[k][()]).astype(float) for k in SUMMARY_KEYS if k in s}
            cyc = f[b['cycles'][i, 0]]
            qdlin = {}
            for j in range(min(N_QDLIN, cyc['Qdlin'].shape[0])):
                try:
                    a = read_vec(f, cyc['Qdlin'][j, 0])
                    qdlin[j] = a if a.size == 1000 else None
                except Exception:
                    qdlin[j] = None
            try:
                vdlin = read_vec(f, b['Vdlin'][i, 0])
                vdlin = vdlin if vdlin.size == 1000 else None
            except Exception:
                vdlin = None
            cells.append(dict(batch=tag, key=f'{tag}c{i}', policy=read_str(f, b['policy_readable'][i, 0]),
                              cycle_life_raw=float(read_vec(f, b['cycle_life'][i, 0])[0]),
                              summary=summary, qdlin=qdlin, vdlin=vdlin))
            print(f'\r[{tag}] {i + 1}/{n}', end='', flush=True)
    print()
    return cells


if __name__ == '__main__':
    raw = {t: load_batch(t, os.path.join(DATA_DIR, fn)) for t, fn in FILES.items()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'wb') as fp:
        pickle.dump(raw, fp)
    print('saved', OUT, {t: len(v) for t, v in raw.items()})
