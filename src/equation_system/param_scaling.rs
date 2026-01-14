use ad_trait::AD;
use nalgebra::ComplexField;

/// Logarithmic mapping from constrained model space (lb, +inf) to unconstrained optimization space (-inf, +inf).
///
/// scaled with respect to a "prior" and a lower bound such that:
/// (1) the output is shifted by `lb` to ensure positivity
/// (2) the logarithm is normalized by the prior to center the mapping around the prior (so that an input near the prior will be near 0.0 in the unconstrained space)
pub fn scaled_log_link<T: AD>(p: T, prior: T, lb: T) -> T {
    debug_assert!(lb >= T::zero(), "lb must be non-negative, got {}", lb);
    debug_assert!(p > lb, "p must be greater than lb, got p={} lb={}", p, lb);
    debug_assert!(
        prior > lb,
        "prior must be greater than lb, got prior={} lb={}",
        prior,
        lb
    );
    ComplexField::ln((p - lb) / (prior - lb))
}

/// Inverse of `scaled_log_link`, mapping from unconstrained optimization space (-inf, +inf) to constrained model space (lb, +inf).
pub fn scaled_log_link_inv<T: AD>(x: T, prior: T, lb: T) -> T {
    debug_assert!(lb >= T::zero(), "lb must be non-negative, got {}", lb);
    debug_assert!(
        prior > lb,
        "prior must be greater than lb, got prior={} lb={}",
        prior,
        lb
    );
    ComplexField::exp(x) * (prior - lb) + lb
}

/// Builds model_to_opt and opt_to_model functions using default_exp_link and its inverse.
/// This assumes all priors are non-zero. If any priors can be zero, a different scaling strategy is needed.
///
/// For negative priors, signs are flipped appropriately.
pub fn default_link_fns_builder<T: AD, const N: usize>(
    priors_vec: [T; N],
) -> (impl Fn([T; N]) -> [T; N], impl Fn([T; N]) -> [T; N]) {
    let model_to_opt = move |p_model: [T; N]| {
        std::array::from_fn(|i| {
            // signs of p and prior must match
            debug_assert!(
                p_model[i].signum() == priors_vec[i].signum(),
                "sign of model param and prior must match, got p_model={} prior={}",
                p_model[i],
                priors_vec[i]
            );
            // lower bound is 1% of absolute value of prior.
            let lb = priors_vec[i].abs() * T::constant(0.01);
            // need to take abs value of prior here to handle negative priors

            scaled_log_link(p_model[i].abs(), priors_vec[i].abs(), lb)
        })
    };
    let opt_to_model = move |p_opt: [T; N]| {
        std::array::from_fn(|i| {
            // p_opt can be any real number.
            // if the prior is negative, we need to:
            // (a) use its absolute value to compute the softplus_default
            // (b) flip the sign of the output to get a negative model param

            // lower bound is 1% of absolute value of prior.
            let lb = priors_vec[i].abs() * T::constant(0.01);
            scaled_log_link_inv(p_opt[i], priors_vec[i].abs(), lb) * priors_vec[i].signum()
        })
    };
    (opt_to_model, model_to_opt)
}
