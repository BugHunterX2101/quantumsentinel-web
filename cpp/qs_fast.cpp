/**
 * qs_fast.cpp — QuantumSentinel high-performance kernels
 *
 * Three kernels bound via pybind11:
 *   1. rolling_corr  — O(N²T) rolling Pearson correlation matrix
 *   2. hmm_forward   — HMM scaled forward algorithm (Baum-Welch inner loop)
 *   3. backtest_loop — Bar-by-bar backtest fill simulation
 *
 * Build (MinGW / Linux / macOS):
 *   cd cpp && pip install -e .
 *
 * Build (MSVC):
 *   cd cpp && pip install -e . --no-build-isolation
 *
 * Runtime: if _qs_fast.pyd/.so not found, backend/services/cpp_ext.py
 * falls back to pure-NumPy implementations automatically.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <stdexcept>

#ifndef M_PI
static constexpr double M_PI = 3.14159265358979323846;
#endif

namespace py = pybind11;
using arr_d = py::array_t<double, py::array::c_style | py::array::forcecast>;

// ---------------------------------------------------------------------------
// 1. rolling_corr — rolling Pearson correlation matrix
//    Input : X (T, N) double array
//    Output: corr_series (T, N, N) — corr[t] = corr matrix at bar t
//            (bars 0..window-2 are identity)
// ---------------------------------------------------------------------------
py::array_t<double> rolling_corr(arr_d X, int window) {
    auto buf = X.request();
    if (buf.ndim != 2)
        throw std::invalid_argument("X must be 2-D (T, N)");

    const int T = static_cast<int>(buf.shape[0]);
    const int N = static_cast<int>(buf.shape[1]);
    const double* x = static_cast<double*>(buf.ptr);

    if (window < 2 || window > T)
        throw std::invalid_argument("window must satisfy 2 <= window <= T");

    // Output: (T, N, N) — result[t] is the N×N correlation matrix at bar t
    std::vector<ssize_t> shape = {T, N, N};
    py::array_t<double> result(shape);
    auto rbuf = result.request();
    double* r = static_cast<double*>(rbuf.ptr);

    // Initialise all to identity
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r[t * N * N + i * N + j] = (i == j) ? 1.0 : 0.0;

    // Rolling Pearson correlation using incremental updates
    for (int t = window - 1; t < T; ++t) {
        // Compute means for columns in window [t-window+1 .. t]
        std::vector<double> means(N, 0.0);
        for (int k = t - window + 1; k <= t; ++k)
            for (int j = 0; j < N; ++j)
                means[j] += x[k * N + j];
        for (int j = 0; j < N; ++j) means[j] /= window;

        // Covariance matrix
        std::vector<double> cov(N * N, 0.0);
        for (int k = t - window + 1; k <= t; ++k) {
            for (int i = 0; i < N; ++i) {
                double di = x[k * N + i] - means[i];
                for (int j = i; j < N; ++j) {
                    double dj = x[k * N + j] - means[j];
                    cov[i * N + j] += di * dj;
                }
            }
        }
        // Scale and fill lower triangle
        for (int i = 0; i < N; ++i)
            for (int j = i; j < N; ++j) {
                cov[i * N + j] /= (window - 1);
                cov[j * N + i] = cov[i * N + j];
            }

        // Convert covariance → correlation
        std::vector<double> stds(N);
        for (int i = 0; i < N; ++i)
            stds[i] = std::sqrt(std::max(cov[i * N + i], 1e-12));

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double denom = stds[i] * stds[j];
                r[t * N * N + i * N + j] = (denom > 1e-14)
                    ? std::min(std::max(cov[i * N + j] / denom, -1.0), 1.0)
                    : (i == j ? 1.0 : 0.0);
            }
        }
    }

    return result;
}


// ---------------------------------------------------------------------------
// 2. hmm_forward — scaled forward algorithm for 2-state Gaussian HMM
//    Inputs:
//      obs  : (T,) observation sequence
//      pi   : (K,) initial state probabilities
//      A    : (K, K) transition matrix (row-stochastic)
//      means: (K,) Gaussian emission means
//      stds : (K,) Gaussian emission standard deviations
//    Returns:
//      alpha: (T, K) scaled forward probabilities
//      log_likelihood: scalar
// ---------------------------------------------------------------------------
py::tuple hmm_forward(arr_d obs_in, arr_d pi_in, arr_d A_in,
                       arr_d means_in, arr_d stds_in) {
    auto obs_buf  = obs_in.request();
    auto pi_buf   = pi_in.request();
    auto A_buf    = A_in.request();
    auto mu_buf   = means_in.request();
    auto sig_buf  = stds_in.request();

    const int T = static_cast<int>(obs_buf.shape[0]);
    const int K = static_cast<int>(pi_buf.shape[0]);

    const double* obs   = static_cast<double*>(obs_buf.ptr);
    const double* pi    = static_cast<double*>(pi_buf.ptr);
    const double* A     = static_cast<double*>(A_buf.ptr);
    const double* mu    = static_cast<double*>(mu_buf.ptr);
    const double* sigma = static_cast<double*>(sig_buf.ptr);

    // Gaussian emission probability
    auto emit = [&](int t, int k) -> double {
        double z = (obs[t] - mu[k]) / std::max(sigma[k], 1e-9);
        return std::exp(-0.5 * z * z) / (sigma[k] * std::sqrt(2.0 * M_PI));
    };

    py::array_t<double> alpha_arr({T, K});
    auto alpha_buf = alpha_arr.request();
    double* alpha = static_cast<double*>(alpha_buf.ptr);

    std::vector<double> scale(T, 0.0);

    // t = 0
    double s0 = 0.0;
    for (int k = 0; k < K; ++k) {
        alpha[k] = pi[k] * emit(0, k);
        s0 += alpha[k];
    }
    scale[0] = std::max(s0, 1e-300);
    for (int k = 0; k < K; ++k) alpha[k] /= scale[0];

    // t > 0
    std::vector<double> tmp(K);
    for (int t = 1; t < T; ++t) {
        double st = 0.0;
        for (int j = 0; j < K; ++j) {
            double sum = 0.0;
            for (int i = 0; i < K; ++i)
                sum += alpha[(t-1)*K + i] * A[i*K + j];
            tmp[j] = sum * emit(t, j);
            st += tmp[j];
        }
        scale[t] = std::max(st, 1e-300);
        for (int j = 0; j < K; ++j)
            alpha[t*K + j] = tmp[j] / scale[t];
    }

    // Log-likelihood
    double ll = 0.0;
    for (int t = 0; t < T; ++t)
        ll += std::log(scale[t]);

    return py::make_tuple(alpha_arr, ll);
}


// ---------------------------------------------------------------------------
// 3. backtest_loop — vectorised fill simulation
//    Inputs:
//      prices    : (T,) close price series
//      signals   : (T,) position signal (-1, 0, +1)
//      commission: commission rate per trade (fraction of notional)
//      spread_bps: bid-ask half-spread in basis points
//    Returns:
//      equity_curve: (T,) equity curve starting at 1.0
//      daily_returns: (T-1,) daily return series
//      n_trades: number of signal changes
//      total_cost: total transaction cost fraction
// ---------------------------------------------------------------------------
py::dict backtest_loop(arr_d prices_in, arr_d signals_in,
                        double commission, double spread_bps) {
    auto p_buf = prices_in.request();
    auto s_buf = signals_in.request();

    const int T = static_cast<int>(p_buf.shape[0]);
    const double* prices  = static_cast<double*>(p_buf.ptr);
    const double* signals = static_cast<double*>(s_buf.ptr);

    const double half_spread = spread_bps / 1e4;

    py::array_t<double> equity_arr(T);
    py::array_t<double> returns_arr(T > 1 ? T - 1 : 1);

    double* equity  = static_cast<double*>(equity_arr.request().ptr);
    double* returns = static_cast<double*>(returns_arr.request().ptr);

    equity[0] = 1.0;
    double position = 0.0;    // current position (+1 long, -1 short, 0 flat)
    double total_cost = 0.0;
    int n_trades = 0;

    for (int t = 1; t < T; ++t) {
        double sig_prev = signals[t - 1];  // signal known at bar t-1
        double price_prev = prices[t - 1];
        double price_curr = prices[t];

        // Price return
        double price_ret = (price_curr - price_prev) / std::max(price_prev, 1e-9);

        // Position change: execute at bar t (1-bar delay from signal)
        double target = sig_prev;  // target from prior signal
        double delta  = target - position;

        double cost = 0.0;
        if (std::abs(delta) > 1e-6) {
            // Commission on the change in position
            cost += commission * std::abs(delta);
            // Spread cost in direction of trade
            cost += half_spread * std::abs(delta);
            position = target;
            n_trades++;
        }

        total_cost += cost;

        // Portfolio return = position * price_ret − transaction cost
        double port_ret = position * price_ret - cost;
        equity[t] = equity[t - 1] * (1.0 + port_ret);
        returns[t - 1] = port_ret;
    }
    // Fill last returns element if T==1
    if (T == 1) returns[0] = 0.0;

    py::dict result;
    result["equity_curve"]  = equity_arr;
    result["daily_returns"] = returns_arr;
    result["n_trades"]      = n_trades;
    result["total_cost"]    = total_cost;
    return result;
}


// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_qs_fast, m) {
    m.doc() = "QuantumSentinel high-performance C++ kernels";

    m.def("rolling_corr", &rolling_corr,
          py::arg("X"), py::arg("window"),
          R"(
rolling_corr(X, window) -> ndarray (T, N, N)

Compute rolling Pearson correlation matrices over a (T, N) return matrix.
Each output slice corr[t] is the N×N correlation matrix computed over
bars [t-window+1 .. t]. Bars 0..window-2 return identity matrices.

Parameters
----------
X      : (T, N) float64 return matrix
window : int, lookback window (>= 2)
)");

    m.def("hmm_forward", &hmm_forward,
          py::arg("obs"), py::arg("pi"), py::arg("A"),
          py::arg("means"), py::arg("stds"),
          R"(
hmm_forward(obs, pi, A, means, stds) -> (alpha, log_likelihood)

Scaled forward algorithm for a K-state Gaussian HMM.

Parameters
----------
obs   : (T,)   observation sequence
pi    : (K,)   initial state distribution
A     : (K, K) row-stochastic transition matrix
means : (K,)   Gaussian emission means
stds  : (K,)   Gaussian emission standard deviations (> 0)

Returns
-------
alpha          : (T, K) scaled forward probabilities
log_likelihood : float  log P(obs | model)
)");

    m.def("backtest_loop", &backtest_loop,
          py::arg("prices"), py::arg("signals"),
          py::arg("commission") = 0.001, py::arg("spread_bps") = 5.0,
          R"(
backtest_loop(prices, signals, commission=0.001, spread_bps=5.0) -> dict

Bar-by-bar backtest with 1-bar execution delay, commission and spread cost.

Parameters
----------
prices     : (T,)  close price series
signals    : (T,)  position signal at each bar (-1, 0, +1)
commission : float commission rate (fraction of notional)
spread_bps : float bid-ask half-spread in basis points

Returns
-------
dict with keys:
  equity_curve  : (T,)   portfolio value starting at 1.0
  daily_returns : (T-1,) daily return series
  n_trades      : int    number of position changes
  total_cost    : float  cumulative transaction cost fraction
)");

    m.attr("__version__") = "4.0.0";
}
