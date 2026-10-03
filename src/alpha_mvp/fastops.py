from __future__ import annotations
import numpy as np
from numba import njit, prange

@njit(cache=True)
def _rank_cs_2d_impl(x):
    n, k = x.shape
    out = np.empty_like(x)
    for i in range(n):
        valid_count = 0
        for j in range(k):
            if np.isfinite(x[i, j]):
                valid_count += 1
        if valid_count == 0:
            for j in range(k):
                out[i, j] = np.nan
            continue
        ranks = np.empty(valid_count, dtype=np.float64)
        vals = np.empty(valid_count, dtype=np.float64)
        idx = 0
        for j in range(k):
            if np.isfinite(x[i, j]):
                vals[idx] = x[i, j]
                idx += 1
        sort_idx = np.argsort(vals)
        start = 0
        while start < valid_count:
            end = start + 1
            while end < valid_count and vals[sort_idx[end]] == vals[sort_idx[start]]:
                end += 1
            average_rank = ((start + 1.0) + end) / 2.0 / valid_count
            for j in range(start, end):
                ranks[sort_idx[j]] = average_rank
            start = end
        idx = 0
        for j in range(k):
            if np.isfinite(x[i, j]):
                out[i, j] = ranks[idx]
                idx += 1
            else:
                out[i, j] = np.nan
    return out

@njit(cache=True)
def _rank_cs_1d_impl(x):
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    valid_count = 0
    for i in range(n):
        if np.isfinite(x[i]):
            valid_count += 1
    if valid_count == 0:
        for i in range(n):
            out[i] = np.nan
        return out
    vals = np.empty(valid_count, dtype=np.float64)
    idx = 0
    for i in range(n):
        if np.isfinite(x[i]):
            vals[idx] = x[i]
            idx += 1
    sort_idx = np.argsort(vals)
    ranks = np.empty(valid_count, dtype=np.float64)
    start = 0
    while start < valid_count:
        end = start + 1
        while end < valid_count and vals[sort_idx[end]] == vals[sort_idx[start]]:
            end += 1
        average_rank = ((start + 1.0) + end) / 2.0 / valid_count
        for j in range(start, end):
            ranks[sort_idx[j]] = average_rank
        start = end
    idx = 0
    for i in range(n):
        if np.isfinite(x[i]):
            out[i] = ranks[idx]
            idx += 1
        else:
            out[i] = np.nan
    return out

def rank_cs(x):
    if x.ndim == 1:
        return _rank_cs_1d_impl(x)
    return _rank_cs_2d_impl(x)

@njit(cache=True, parallel=True)
def _daily_corr_impl(x, y, rank_flag, min_valid):
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        xi = x[i]
        yi = y[i]
        valid = 0
        for j in range(len(xi)):
            if np.isfinite(xi[j]) and np.isfinite(yi[j]):
                valid += 1
        if valid < min_valid:
            out[i] = np.nan
            continue
        fx = np.empty(valid, dtype=np.float64)
        fy = np.empty(valid, dtype=np.float64)
        cnt = 0
        for j in range(len(xi)):
            if np.isfinite(xi[j]) and np.isfinite(yi[j]):
                fx[cnt] = xi[j]
                fy[cnt] = yi[j]
                cnt += 1
        if rank_flag:
            sort_idx_x = np.argsort(fx)
            sort_idx_y = np.argsort(fy)
            rx = np.empty(valid, dtype=np.float64)
            ry = np.empty(valid, dtype=np.float64)
            start = 0
            while start < valid:
                end = start + 1
                while end < valid and fx[sort_idx_x[end]] == fx[sort_idx_x[start]]:
                    end += 1
                average_rank = ((start + 1.0) + end) / 2.0 / valid
                for j in range(start, end):
                    rx[sort_idx_x[j]] = average_rank
                start = end
            start = 0
            while start < valid:
                end = start + 1
                while end < valid and fy[sort_idx_y[end]] == fy[sort_idx_y[start]]:
                    end += 1
                average_rank = ((start + 1.0) + end) / 2.0 / valid
                for j in range(start, end):
                    ry[sort_idx_y[j]] = average_rank
                start = end
            mx, my = 0.0, 0.0
            for j in range(valid):
                mx += rx[j]
                my += ry[j]
            mx /= valid
            my /= valid
            num = 0.0
            dx, dy = 0.0, 0.0
            for j in range(valid):
                dx += (rx[j] - mx) * (rx[j] - mx)
                dy += (ry[j] - my) * (ry[j] - my)
                num += (rx[j] - mx) * (ry[j] - my)
            denom = np.sqrt(dx * dy)
            out[i] = num / denom if denom > 1e-10 else np.nan
        else:
            mx, my = 0.0, 0.0
            for j in range(valid):
                mx += fx[j]
                my += fy[j]
            mx /= valid
            my /= valid
            num = 0.0
            dx, dy = 0.0, 0.0
            for j in range(valid):
                dx += (fx[j] - mx) * (fx[j] - mx)
                dy += (fy[j] - my) * (fy[j] - my)
                num += (fx[j] - mx) * (fy[j] - my)
            denom = np.sqrt(dx * dy)
            out[i] = num / denom if denom > 1e-10 else np.nan
    return out

def daily_corr(x, y, rank=False, min_valid=10):
    return _daily_corr_impl(x, y, rank, min_valid)

@njit(cache=True, parallel=True)
def _rolling_corr_impl(x, y, w):
    n, k = x.shape
    out = np.full((n, k), np.nan, dtype=np.float64)
    for j in prange(k):
        count = 0
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        minimum = max(2, w // 2)
        for i in range(n):
            xv = x[i, j]
            yv = y[i, j]
            if np.isfinite(xv) and np.isfinite(yv):
                count += 1
                sx += xv
                sy += yv
                sxx += xv * xv
                syy += yv * yv
                sxy += xv * yv
            if i >= w:
                old_x = x[i - w, j]
                old_y = y[i - w, j]
                if np.isfinite(old_x) and np.isfinite(old_y):
                    count -= 1
                    sx -= old_x
                    sy -= old_y
                    sxx -= old_x * old_x
                    syy -= old_y * old_y
                    sxy -= old_x * old_y
            if i >= w - 1 and count >= minimum:
                covariance = sxy - sx * sy / count
                variance_x = max(0.0, sxx - sx * sx / count)
                variance_y = max(0.0, syy - sy * sy / count)
                denominator = np.sqrt(variance_x * variance_y)
                if denominator > 1e-10:
                    out[i, j] = covariance / denominator
    return out

def rolling_corr(x, y, w):
    return _rolling_corr_impl(x, y, w)

def fast_rank_cs(x):
    return rank_cs(x)

def fast_daily_corr(x, y, rank=False, min_valid=10):
    return daily_corr(x, y, rank, min_valid)

def fast_rolling_corr(x, y, w):
    return rolling_corr(x, y, w)

@njit(cache=True, parallel=True)
def fast_rolling_mean(x, w):
    n, k = x.shape
    out = np.full((n, k), np.nan, dtype=np.float64)
    for j in prange(k):
        asum = 0.0
        count = 0
        for i in range(n):
            val = x[i, j]
            if np.isfinite(val):
                asum += val
                count += 1
            
            if i >= w:
                prev_val = x[i - w, j]
                if np.isfinite(prev_val):
                    asum -= prev_val
                    count -= 1
            
            if i >= w - 1 and count >= w // 2 + 1:
                out[i, j] = asum / count if count > 0 else np.nan
    return out

@njit(cache=True, parallel=True)
def fast_rolling_std(x, w):
    n, k = x.shape
    out = np.full((n, k), np.nan, dtype=np.float64)
    for j in prange(k):
        asum = 0.0
        asq_sum = 0.0
        count = 0
        for i in range(n):
            value = x[i, j]
            if np.isfinite(value):
                asum += value
                asq_sum += value * value
                count += 1
            if i >= w:
                old = x[i - w, j]
                if np.isfinite(old):
                    asum -= old
                    asq_sum -= old * old
                    count -= 1
            if i >= w - 1 and count >= w // 2 + 1:
                mean = asum / count
                out[i, j] = np.sqrt(max(0.0, asq_sum / count - mean * mean))
    return out

@njit(cache=True, parallel=True)
def fast_rolling_zscore(x, w):
    """Per-asset rolling z-score using at least half-window observations."""
    n, k = x.shape
    out = np.full((n, k), np.nan, dtype=np.float64)
    for j in prange(k):
        asum = 0.0
        asq_sum = 0.0
        count = 0
        for i in range(n):
            value = x[i, j]
            if np.isfinite(value):
                asum += value
                asq_sum += value * value
                count += 1
            if i >= w:
                old = x[i - w, j]
                if np.isfinite(old):
                    asum -= old
                    asq_sum -= old * old
                    count -= 1
            if i >= w - 1 and count >= w // 2 + 1 and np.isfinite(value):
                mean = asum / count
                variance = max(0.0, asq_sum / count - mean * mean)
                std = np.sqrt(variance)
                if std > 1e-12:
                    out[i, j] = (value - mean) / std
    return out

@njit(cache=True, parallel=True)
def fast_rolling_sum(x, w):
    n, k = x.shape
    out = np.full((n, k), np.nan, dtype=np.float64)
    for j in prange(k):
        asum = 0.0
        count = 0
        for i in range(n):
            val = x[i, j]
            if np.isfinite(val):
                asum += val
                count += 1
            if i >= w:
                prev_val = x[i - w, j]
                if np.isfinite(prev_val):
                    asum -= prev_val
                    count -= 1
            if i >= w - 1:
                out[i, j] = asum if count >= w // 2 + 1 else np.nan
    return out

@njit(cache=True, parallel=True)
def fast_rolling_max(x, w):
    n, k = x.shape
    out = np.full((n, k), np.nan, dtype=np.float64)
    for j in prange(k):
        for i in range(w - 1, n):
            cur_max = -np.inf
            found = False
            for t in range(i - w + 1, i + 1):
                val = x[t, j]
                if np.isfinite(val):
                    if val > cur_max:
                        cur_max = val
                    found = True
            if found:
                out[i, j] = cur_max
    return out

@njit(cache=True, parallel=True)
def fast_rolling_min(x, w):
    n, k = x.shape
    out = np.full((n, k), np.nan, dtype=np.float64)
    for j in prange(k):
        for i in range(w - 1, n):
            cur_min = np.inf
            found = False
            for t in range(i - w + 1, i + 1):
                val = x[t, j]
                if np.isfinite(val):
                    if val < cur_min:
                        cur_min = val
                    found = True
            if found:
                out[i, j] = cur_min
    return out

@njit(cache=True, parallel=True)
def fast_rolling_rank(x, w):
    n, k = x.shape
    out = np.full((n, k), np.nan, dtype=np.float64)
    for j in prange(k):
        for i in range(w - 1, n):
            val = x[i, j]
            if not np.isfinite(val):
                continue
            count = 0
            less_count = 0
            tie_count = 0
            for t in range(i - w + 1, i + 1):
                cur = x[t, j]
                if np.isfinite(cur):
                    count += 1
                    if cur < val:
                        less_count += 1
                    elif cur == val:
                        tie_count += 1
            if count >= w // 2 + 1:
                out[i, j] = (
                    less_count + (tie_count + 1.0) / 2.0
                ) / count
    return out

@njit(cache=True, parallel=True)
def fast_rolling_wma(x, w):
    n, k = x.shape
    out = np.full((n, k), np.nan, dtype=np.float64)
    weights = np.arange(1, w + 1, dtype=np.float64)
    w_sum = weights.sum()
    for j in prange(k):
        for i in range(w - 1, n):
            asum = 0.0
            found = 0
            for t in range(w):
                val = x[i - w + 1 + t, j]
                if np.isfinite(val):
                    asum += val * weights[t]
                    found += 1
            if found == w:
                out[i, j] = asum / w_sum
    return out
