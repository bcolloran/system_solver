use ad_trait::forward_ad::adfn::adfn;
use anyhow::bail;
use argmin::{
    core::{CostFunction, Error as ArgminError, Gradient, Jacobian, Operator},
    solver::simulatedannealing::Anneal,
};
use nalgebra::DVector;

use crate::prelude::*;
use rand::{distr, prelude::*};

/// Note that in the case of residual aggregation to scalar, the Operator returns a 1×1 matrix because aggregation to scalar happens *within* the FunctionEngine call (as it must in order to compute derivatives correctly).
///
/// Thus, we can just use the Operator implementation to get the scalar cost value as a 1×1 matrix, and then extract the scalar from that matrix in the CostFunction implementation.
impl<G64, U64, Gadfn, Uadfn, R, A, const N: usize> CostFunction for SubProblem<G64, U64, Gadfn, Uadfn, R, A, N>
where
    G64: GivenParamsFor<f64, N>,
    U64: UnknownParamsFor<f64, N>,
    Gadfn: GivenParamsFor<adfn<1>, N>,
    Uadfn: UnknownParamsFor<adfn<1>, N>,
    R: ResidTransHOF,
    A: ResidAggFnToScalarGen,
{
    type Param = nalgebra::DVector<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, ArgminError> {
        let operator_result = self.apply(p)?;
        Ok(operator_result[0])
    }
}

impl<G64, U64, Gadfn, Uadfn, R, A, const N: usize> Operator for SubProblem<G64, U64, Gadfn, Uadfn, R, A, N>
where
    G64: GivenParamsFor<f64, N>,
    U64: UnknownParamsFor<f64, N>,
    Gadfn: GivenParamsFor<adfn<1>, N>,
    Uadfn: UnknownParamsFor<adfn<1>, N>,
    R: ResidTransHOF,
    A: ResidAggHOF,
{
    type Param = nalgebra::DVector<f64>;
    type Output = nalgebra::DVector<f64>;

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, ArgminError> {
        if p.len() != self.block.unknown_idxs.len() {
            bail!(
                "Parameter vector length ({}) for subproblem cost function did not match number subproblem unknowns ({})",
                p.len(),
                self.block.unknown_idxs.len()
            );
        }

        let p_vec: Vec<f64> = p.as_slice().to_vec();
        let p_opt = self.optspace_fullprob_input_from_subprob_input(&p_vec);
        // println!(
        //     "SubProblem::cost called with full opt space params: {:?}",
        //     p_opt
        // );
        let result = self.loss_fn_engine.call(&p_opt);
        Ok(nalgebra::DVector::from_vec(result))
    }
}

impl<G64, U64, Gadfn, Uadfn, R, A, const N: usize> Gradient for SubProblem<G64, U64, Gadfn, Uadfn, R, A, N>
where
    G64: GivenParamsFor<f64, N>,
    U64: UnknownParamsFor<f64, N>,
    Gadfn: GivenParamsFor<adfn<1>, N>,
    Uadfn: UnknownParamsFor<adfn<1>, N>,
    R: ResidTransHOF,
    A: ResidAggFnToScalarGen,
{
    type Param = nalgebra::DVector<f64>;
    type Gradient = nalgebra::DVector<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, ArgminError> {
        if p.len() != self.block.unknown_idxs.len() {
            bail!(
                "Parameter vector length ({}) for subproblem gradient function did not match number subproblem unknowns ({})",
                p.len(),
                self.block.unknown_idxs.len()
            );
        }

        let p_vec: Vec<f64> = p.as_slice().to_vec();
        let p_full = self.optspace_fullprob_input_from_subprob_input(&p_vec);

        let (_values, full_jacobian) = self.loss_fn_engine.derivative(&p_full);

        // Select columns, then convert 1×N matrix to N×1 vector
        let gradient_matrix = self.select_subprob_jacobian(&full_jacobian);
        if gradient_matrix.nrows() != 1 {
            bail!(
                "Expected gradient to have 1 row (scalar function output), but got {} rows",
                gradient_matrix.nrows()
            );
        }
        Ok(gradient_matrix.row(0).transpose())
    }
}

impl<G64, U64, Gadfn, Uadfn, R, const N: usize> Jacobian for SubProblem<G64, U64, Gadfn, Uadfn, R, ResidNoOpGaussNewton, N>
where
    G64: GivenParamsFor<f64, N>,
    U64: UnknownParamsFor<f64, N>,
    Gadfn: GivenParamsFor<adfn<1>, N>,
    Uadfn: UnknownParamsFor<adfn<1>, N>,
    R: ResidTransHOF,
{
    type Param = nalgebra::DVector<f64>;
    type Jacobian = nalgebra::DMatrix<f64>;

    fn jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, ArgminError> {
        if p.len() != self.block.unknown_idxs.len() {
            bail!(
                "Parameter vector length ({}) for subproblem jacobian function did not match number subproblem unknowns ({})",
                p.len(),
                self.block.unknown_idxs.len()
            );
        }

        let p_vec: Vec<f64> = p.as_slice().to_vec();
        let p_full = self.optspace_fullprob_input_from_subprob_input(&p_vec);

        let (_values, full_jacobian) = self.loss_fn_engine.derivative(&p_full);

        Ok(self.select_subprob_jacobian(&full_jacobian))
    }
}

impl<G64, U64, Gadfn, Uadfn, R, A, const N: usize> Anneal for SubProblem<G64, U64, Gadfn, Uadfn, R, A, N>
where
    G64: GivenParamsFor<f64, N>,
    U64: UnknownParamsFor<f64, N>,
    Gadfn: GivenParamsFor<adfn<1>, N>,
    Uadfn: UnknownParamsFor<adfn<1>, N>,
    R: ResidTransHOF,
    A: ResidAggHOF,
{
    type Param = DVector<f64>;
    type Output = DVector<f64>;
    type Float = f64;

    /// Proposes a neighbor by modifying exactly one coordinate.
    ///
    /// - Uses a temperature-scaled mixture:
    ///   - small uniform step (local exploration)
    ///   - occasional Cauchy (heavy-tailed) big jump (order-of-magnitude moves in log-space)
    /// - Persists RNG state across calls (required for meaningful SA behavior).
    fn anneal(&self, p: &Self::Param, temp: Self::Float) -> Result<Self::Output, ArgminError> {
        if p.len() != self.block.unknown_idxs.len() {
            bail!(
                "Parameter vector length ({}) for subproblem anneal function did not match number subproblem unknowns ({})",
                p.len(),
                self.block.unknown_idxs.len()
            );
        }

        let sa_cfg = self
            .sa_cfg
            .as_ref()
            .expect("Simulated annealing config (sa_cfg) not set on annealing SubProblem");

        // Normalize temperature into [0, 1] fraction of initial temp.
        // tau ~ 1 => "hot" => larger steps & more frequent big jumps
        // tau ~ 0 => "cold" => smaller steps & rarer big jumps
        let tau = if sa_cfg.init_temp > 0.0 {
            (temp / sa_cfg.init_temp).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Linear interpolation helper: lerp(lo, hi, tau)
        #[inline]
        fn lerp(lo: f64, hi: f64, t: f64) -> f64 {
            lo + (hi - lo) * t
        }

        let small_step = lerp(sa_cfg.small_step_min, sa_cfg.small_step_init, tau);
        let big_step = lerp(sa_cfg.big_step_min, sa_cfg.big_step_init, tau);
        let p_big = lerp(sa_cfg.p_big_min, sa_cfg.p_big_init, tau).clamp(0.0, 1.0);

        let mut out = p.clone();

        // Persistent RNG (shared across clones of SubProblem via Arc).
        let mut rng = self.rng.lock().expect("SubProblem.sa_rng mutex poisoned");

        // Choose one coordinate to modify.
        let idx = rng.random_range(0..p.len());

        // Mixture proposal: local vs occasional heavy-tailed jump
        let mut delta = if rng.random_bool(p_big) {
            // Standard Cauchy sample from u ~ (0, 1): tan(pi*(u - 0.5))
            // Heavy tails => rare but large moves (useful for "orders of magnitude" rescue).
            let u: f64 = rng.sample(distr::Open01);
            let cauchy = (std::f64::consts::PI * (u - 0.5)).tan();
            big_step * cauchy
        } else {
            // Local move (uniform is fine; you can swap to normal if you prefer).
            rng.random_range(-small_step..small_step)
        };

        // Safety clamp (limits extreme tails causing overflow / NaN in downstream exp/link funcs)
        delta = delta.clamp(-sa_cfg.max_abs_step, sa_cfg.max_abs_step);

        // // --- Optional: gradient-biased drift (compile-time gated) -----------------------------
        // //
        // // Enable with:
        // //   [features]
        // //   sa_grad = []
        // //
        // // Then implement `grad_optspace(&self, p: &DVector<f64>) -> Result<DVector<f64>, ArgminError>`
        // // returning the gradient of your *cost* in opt-space.
        // //
        // // This uses a *bounded* bias: normalize gradient to avoid exploding gradients dominating.
        // // Drift is stronger when colder (1 - tau), but still small compared to big jumps.
        // #[cfg(feature = "sa_grad")]
        // {
        //     if sa_cfg.grad_drift_max > 0.0 {
        //         let g = self.grad_optspace(p)?; // <- you provide this
        //         let gnorm = g.norm();
        //         if gnorm.is_finite() && gnorm > 1e-12 {
        //             let g_unit = g[idx] / gnorm; // in [-1, 1]
        //             let drift = sa_cfg.grad_drift_max * (1.0 - tau);
        //             delta += -drift * g_unit;
        //             delta = delta.clamp(-sa_cfg.max_abs_step, sa_cfg.max_abs_step);
        //         }
        //     }
        // }
        // // -------------------------------------------------------------------------------------

        out[idx] += delta;
        Ok(out)
    }
}
