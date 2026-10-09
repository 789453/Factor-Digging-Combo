"""Versioned mathematical primitives, independent of search and market labels."""
from __future__ import annotations

from functools import lru_cache
from itertools import product
import numpy as np
from numba import njit
from scipy.linalg import helmert
from scipy.special import logsumexp

PRIMITIVE_VERSION = 'complex-primitives-20261008-v1'


def rolling_sum(x: np.ndarray, width: int) -> np.ndarray:
    """Trailing sum including current; full finite support required."""
    if width < 1:
        raise ValueError('positive rolling support required')
    a = np.asarray(x, dtype=float)
    bad = ~np.isfinite(a)
    p = np.concatenate((np.zeros_like(a[:1]), np.cumsum(np.where(bad, 0., a), axis=0)), axis=0)
    b = np.concatenate((np.zeros_like(a[:1], dtype=int), np.cumsum(bad, axis=0)), axis=0)
    result = np.full_like(a, np.nan)
    if len(a) >= width:
        result[width-1:] = p[width:] - p[:-width]
        result[width-1:][(b[width:] - b[:-width]) > 0] = np.nan
    return result


def causal_standardize(x: np.ndarray, width: int, min_count: int,
                       center: bool = True, cap: float = 8.) -> tuple[np.ndarray, np.ndarray]:
    """Previous observations only. Mean/RMS, explicitly not a median scale."""
    import pandas as pd
    if not 2 <= min_count <= width or cap <= 0:
        raise ValueError('invalid normalization support')
    s = pd.Series(np.asarray(x, float)).shift(1)
    mean = s.rolling(width, min_periods=min_count).mean().to_numpy() if center else np.zeros(len(s))
    variance = (s.rolling(width, min_periods=min_count).var(ddof=0).to_numpy()
                if center else s.pow(2).rolling(width, min_periods=min_count).mean().to_numpy())
    scale = np.sqrt(np.maximum(variance, 0))
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (x - mean) / scale
    z[~np.isfinite(scale) | (scale <= 1e-12)] = np.nan
    clipped = np.isfinite(z) & (np.abs(z) > cap)
    return np.clip(z, -cap, cap), clipped


def signed_log_bins(x: np.ndarray) -> np.ndarray:
    """13 states: |x|<.25; six signed geometric bins; exact edge goes up."""
    a = np.asarray(x, float)
    mag = np.searchsorted([.25, .5, 1., 2., 4., 8.], np.abs(a), side='right')
    out = np.where(mag == 0, 0, np.where(a >= 0, mag, 6 + mag)).astype(int)
    out[~np.isfinite(a)] = -1
    return out


def ilr_counts(counts: np.ndarray, pseudocount: float = .5) -> np.ndarray:
    """Fixed Helmert orthonormal contrasts. Empty counts remain missing."""
    x = np.asarray(counts, float)
    if x.shape[-1] < 2 or pseudocount <= 0 or np.any(x < 0):
        raise ValueError('invalid probability counts')
    result = np.log(x + pseudocount) @ helmert(x.shape[-1], full=False).T
    result[np.sum(x, axis=-1) == 0] = np.nan
    return result


def tilt_partition(z: np.ndarray, theta: np.ndarray,
                   weight: np.ndarray | None = None) -> tuple[float, float, float]:
    """Log partition, effective sample size, maximum weight; empty -> NaN."""
    a = np.asarray(z, float)
    theta = np.asarray(theta, float)
    if a.ndim != 2 or theta.shape != (a.shape[1],):
        raise ValueError('tilt dimension mismatch')
    w = np.ones(len(a)) if weight is None else np.asarray(weight, float)
    if w.shape != (len(a),) or np.any(w < 0) or not np.isfinite(w).all():
        raise ValueError('invalid tilt weights')
    good = np.isfinite(a).all(axis=1) & (w > 0)
    if not good.any():
        return np.nan, np.nan, np.nan
    w = w[good] / w[good].sum()
    v = a[good] @ theta + np.log(w)
    k = logsumexp(v)
    probability = np.exp(v-k)
    return float(k), float(1/np.sum(probability**2)), float(probability.max())


def causal_haar(x: np.ndarray, half_width: int) -> np.ndarray:
    mean = rolling_sum(x, half_width) / half_width
    past = np.full_like(mean, np.nan)
    past[half_width:] = mean[:-half_width]
    return (mean - past) / np.sqrt(2.)


def matrix_log_spd(x: np.ndarray) -> np.ndarray:
    """Matrix log, not elementwise. Invalid or non-positive matrices fail."""
    a = np.asarray(x, float)
    if not np.isfinite(a).all() or not np.allclose(a, a.swapaxes(-1,-2), atol=1e-10):
        raise ValueError('finite symmetric SPD matrix required')
    val, vec = np.linalg.eigh(a)
    if np.any(val <= 0):
        raise ValueError('matrix is not positive definite')
    return (vec * np.log(val)[..., None, :]) @ vec.swapaxes(-1,-2)


def relative_covariance_log(short: np.ndarray, long: np.ndarray) -> np.ndarray:
    val, vec = np.linalg.eigh(long)
    if not np.isfinite(short).all() or not np.isfinite(long).all() or np.any(val <= 0):
        raise ValueError('valid positive-definite covariance required')
    inv = (vec * (1/np.sqrt(val))[..., None, :]) @ vec.swapaxes(-1,-2)
    x = inv @ short @ inv
    return matrix_log_spd((x+x.swapaxes(-1,-2))/2)


def transition_flux(states: np.ndarray, nstate: int, lag: int = 1,
                    pseudocount: float = .5) -> tuple[np.ndarray, float]:
    s = np.asarray(states, int)
    if nstate < 3 or lag < 1 or pseudocount <= 0:
        raise ValueError('invalid transition declaration')
    a, b = s[:-lag], s[lag:]
    good = (a>=0)&(b>=0)&(a<nstate)&(b<nstate)
    if not good.any():
        return np.full((nstate,nstate),np.nan), np.nan
    q = np.bincount(a[good]*nstate+b[good], minlength=nstate*nstate).reshape(nstate,nstate)+pseudocount
    q /= q.sum()
    return q-q.T, float(np.sum(q*np.log(q/q.T)))


def _is_lyndon(w: tuple) -> bool:
    return all(w < w[k:]+w[:k] for k in range(1,len(w)))


def _bracket(w: tuple) -> dict:
    if len(w)==1:
        return {w:1.}
    # Longest proper Lyndon suffix: standard bracketing.
    k = next(k for k in range(1,len(w)) if _is_lyndon(w[k:]))
    a,b = _bracket(w[:k]),_bracket(w[k:])
    out={}
    for x,c in a.items():
        for y,d in b.items():
            out[x+y]=out.get(x+y,0)+c*d
            out[y+x]=out.get(y+x,0)-c*d
    return out


@lru_cache(maxsize=8)
def lie_basis(dimension: int, order: int) -> tuple[tuple, np.ndarray]:
    words=tuple(w for w in product(range(dimension),repeat=order) if _is_lyndon(w))
    tensor_words=list(product(range(dimension),repeat=order))
    matrix=np.array([[ _bracket(w).get(t,0.) for w in words] for t in tensor_words])
    return words,np.linalg.pinv(matrix)


@njit(cache=True)
def _signature_prefix(dx):
    n,d=dx.shape
    a=np.zeros((n+1,d)); b=np.zeros((n+1,d,d)); c=np.zeros((n+1,d,d,d))
    for t in range(n):
        for i in range(d):
            a[t+1,i]=a[t,i]+dx[t,i]
            for j in range(d):
                b[t+1,i,j]=b[t,i,j]+a[t,i]*dx[t,j]+dx[t,i]*dx[t,j]/2
                for k in range(d):
                    c[t+1,i,j,k]=c[t,i,j,k]+b[t,i,j]*dx[t,k]+a[t,i]*dx[t,j]*dx[t,k]/2+dx[t,i]*dx[t,j]*dx[t,k]/6
    return a,b,c


def window_logsignatures(dx: np.ndarray, ends: np.ndarray, width: int,
                         chunk_size: int = 16384) -> np.ndarray:
    """Piecewise-linear logsignature, independent Lyndon basis, <=order3.

    Prefixes rebased per bounded chunk to control cancellation. No missing
    increments bridged. End is exclusive; full [end-width,end) required.
    """
    x=np.asarray(dx,float); ends=np.asarray(ends,int); d=x.shape[1]
    _,p2=lie_basis(d,2); _,p3=lie_basis(d,3)
    result=np.full((len(ends),d+len(p2)+len(p3)),np.nan)
    for start in range(0,len(x),chunk_size):
        stop=min(start+chunk_size,len(x))
        pick=np.flatnonzero((ends>start)&(ends<=stop)&(ends>=width))
        if not len(pick): continue
        origin=max(0,start-width)
        block=x[origin:stop]
        bad=np.r_[0,np.cumsum(~np.isfinite(block).all(axis=1))]
        a,b,c=_signature_prefix(np.nan_to_num(block,nan=0.))
        hi=ends[pick]-origin; lo=hi-width
        good=(lo>=0)&((bad[hi]-bad[lo])==0)
        pick=pick[good]; hi=hi[good]; lo=lo[good]
        if not len(pick): continue
        s1=a[hi]-a[lo]
        s2=b[hi]-b[lo]-a[lo,:,None]*s1[:,None,:]
        s3=c[hi]-c[lo]-b[lo,:,:,None]*s1[:,None,None,:]-a[lo,:,None,None]*s2[:,None,:,:]
        l2=s2-.5*s1[:,:,None]*s1[:,None,:]
        l3=s3-.5*(s1[:,:,None,None]*s2[:,None,:,:]+s2[:,:,:,None]*s1[:,None,None,:])+s1[:,:,None,None]*s1[:,None,:,None]*s1[:,None,None,:]/3
        result[pick]=np.column_stack((s1,l2.reshape(len(pick),-1)@p2.T,l3.reshape(len(pick),-1)@p3.T))
    return result


def fractional_kernel(d: float, support: int) -> np.ndarray:
    if not 0<=d<=1 or support<2:
        raise ValueError('fractional support/domain invalid')
    w=np.ones(support)
    for k in range(1,support): w[k]=-w[k-1]*(d-k+1)/k
    return w
