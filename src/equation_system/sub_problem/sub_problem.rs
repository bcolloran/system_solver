use std::rc::Rc;
use std::sync::{Arc, Mutex};

use ad_trait::{
    differentiable_function::ForwardAD, forward_ad::adfn::adfn, function_engine::FunctionEngine,
};
use argmin::core::{Error as ArgminError, Operator};
use nalgebra::{DVector, Dyn, Matrix, VecStorage};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::equation_system::sub_problem::solve_subproblem::simulated_annealing::SimulatedAnnealingConfig;
use crate::prelude::*;

pub struct ToScalar;
pub struct ToVector;

/// A sub-problem within an equation system optimization problem.
///
/// Type parameters:
/// - `R`: The residual transformation function generator.
/// - `A`: The residual aggregation function generator.
#[derive(Clone)]
pub struct SubProblem<R, A>
where
    R: ResidTransHOF,
    A: ResidAggHOF,
{
    pub loss_fn_engine: Rc<
        FunctionEngine<ObjectiveFunction<f64, R, A>, ObjectiveFunction<adfn<1>, R, A>, ForwardAD>,
    >,
    pub block: SolutionBlock,
    pub param_scaler: Option<ParamScaler<f64>>,
    pub initial_unknowns: DynamicsDerivedParams<f64>,
    pub residual_agg_fn_gen: A,
    pub rng: Arc<Mutex<StdRng>>,
    pub sa_cfg: Option<SimulatedAnnealingConfig>,
}

impl<R, A> SubProblem<R, A>
where
    R: ResidTransHOF,
    A: ResidAggHOF,
{
    /// Creates a new SubProblem for the given solution block.
    pub fn new(
        super_prob_resid_fn: &ResidualFns,
        solution_block: &SolutionBlock,
        givens: &DynamicsGivenParams<f64>,
        initial_unknowns: &DynamicsDerivedParams<f64>,
        residual_scaling: R,
        residual_agg_fn_gen: A,
        use_scaling: bool,
    ) -> Self {
        // Filter the residual functions to only those relevant to this sub-problem
        let sub_prob_res_fns = super_prob_resid_fn.filter_res_fns_to_block(solution_block);

        let loss_f64 = ObjectiveFunction::new(
            givens,
            &sub_prob_res_fns.f64,
            residual_scaling.clone(),
            residual_agg_fn_gen.clone(),
            if use_scaling {
                Some(ParamScaler::new_link_fns_from_priors(initial_unknowns))
            } else {
                None
            },
        );

        let loss_adfn = ObjectiveFunction::new(
            givens,
            &sub_prob_res_fns.adfn_1,
            residual_scaling,
            residual_agg_fn_gen.clone(),
            if use_scaling {
                Some(ParamScaler::new_link_fns_from_priors(initial_unknowns))
            } else {
                None
            },
        );

        let loss_fn_engine = FunctionEngine::new(loss_f64, loss_adfn, ForwardAD::new());

        let param_scaler = if use_scaling {
            Some(ParamScaler::new_link_fns_from_priors(initial_unknowns))
        } else {
            None
        };

        // // Extract only the active parameters from initial_unknowns
        // let full_params_opt_space = (param_scaler.model_to_opt)(initial_unknowns.to_arr());

        SubProblem {
            loss_fn_engine: Rc::new(loss_fn_engine),
            // equation_idxs: solution_block.equation_idxs.clone(),
            // unknown_idxs: solution_block.unknown_idxs.clone(),
            block: solution_block.clone(),
            param_scaler,
            residual_agg_fn_gen,
            initial_unknowns: initial_unknowns.clone(),
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(0))),
            sa_cfg: None,
        }
    }

    pub fn with_simulated_annealing_config(mut self, sa_config: SimulatedAnnealingConfig) -> Self {
        self.sa_cfg = Some(sa_config);
        self
    }

    /// Converts a full-problem parameter vector from optimization space to model space
    pub fn optspace_to_modspace(&self, opt_params: &[f64; N_UNKNOWNS]) -> [f64; N_UNKNOWNS] {
        if let Some(param_scaling) = &self.param_scaler {
            param_scaling.opt_to_model(*opt_params)
        } else {
            *opt_params
        }
    }

    /// Converts a full-problem parameter vector from model space to optimization space
    pub fn modspace_to_optspace(&self, model_params: &[f64; N_UNKNOWNS]) -> [f64; N_UNKNOWNS] {
        if let Some(param_scaling) = &self.param_scaler {
            param_scaling.model_to_opt(*model_params)
        } else {
            *model_params
        }
    }

    /// Converts a full-problem model-space parameter vector to a param struct.
    pub fn modspace_to_params(
        &self,
        model_params: &[f64; N_UNKNOWNS],
    ) -> DynamicsDerivedParams<f64> {
        DynamicsDerivedParams::from_arr(*model_params)
    }

    pub fn select_subprob_items(&self, items: &[f64]) -> Vec<f64> {
        debug_assert!(
            items.len() == N_UNKNOWNS,
            "Items length ({}) does not match expected number of unknowns ({})",
            items.len(),
            N_UNKNOWNS
        );
        self.block
            .unknown_idxs
            .iter()
            .map(|&idx| items[idx])
            .collect()
    }

    /// Selects the columns from a jacobian matrix that correspond to this subproblem's unknowns
    /// and returns them as a DMatrix. Works for both multi-row jacobians and single-row gradients.
    pub(super) fn select_subprob_jacobian(
        &self,
        full_jacobian: &Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>,
    ) -> nalgebra::DMatrix<f64> {
        nalgebra::DMatrix::from_fn(
            full_jacobian.nrows(),
            self.block.unknown_idxs.len(),
            |i, j| full_jacobian[(i, self.block.unknown_idxs[j])],
        )
    }

    /// makes a vector of the initial parameters relevant to this sub-problem in opt space
    pub fn subprob_initial_params_optspace(&self) -> DVector<f64> {
        DVector::from_iterator(
            self.block.unknown_idxs.len(),
            self.block
                .unknown_idxs
                .iter()
                .map(|&idx| self.fullprob_initial_params_optspace()[idx]),
        )
    }

    pub fn fullprob_initial_params_optspace(&self) -> [f64; N_UNKNOWNS] {
        let p_model = self.initial_unknowns.to_arr();
        self.modspace_to_optspace(&p_model)
    }

    /// Reconstructs unknown params struct by patching in optimized sub-problem parameters into initial parameter set.
    pub fn params_with_subprob_optimizer_result(
        &self,
        p_opt: &Vec<f64>,
    ) -> DynamicsDerivedParams<f64> {
        let full_optspace = self.optspace_fullprob_input_from_subprob_input(p_opt);

        let full_modspace = self.optspace_to_modspace(&full_optspace);

        DynamicsDerivedParams::from_arr(full_modspace)
    }

    /// The `argmin` optimizer uses only the number of active parameters for this sub-problem. Before handing inputs fromt the optimizer into the residual functions, we need to reconstruct the full opt space parameter vector. This function does that.
    pub fn optspace_fullprob_input_from_subprob_input(
        &self,
        opt_space_inputs: &Vec<f64>,
    ) -> [f64; N_UNKNOWNS] {
        debug_assert!(
            opt_space_inputs.len() == self.block.unknown_idxs.len(),
            "Parameter vector length ({}) for reconstruction did not match number subproblem unknowns ({})",
            opt_space_inputs.len(),
            self.block.unknown_idxs.len()
        );

        let mut full_params = self.fullprob_initial_params_optspace();

        // overwrite the intial opt space params with the inputs relevant to this sub-problem
        for (i, &idx) in self.block.unknown_idxs.iter().enumerate() {
            full_params[idx] = opt_space_inputs[i].clone();
        }
        full_params
    }
}

impl<R, A> SubProblem<R, A>
where
    R: ResidTransHOF,
    A: ResidAggHOF,
{
    pub fn initial_params_cost(&self) -> Result<f64, ArgminError> {
        let init_params = self.subprob_initial_params_optspace();
        let resids = self.apply(&init_params)?;
        Ok(self
            .residual_agg_fn_gen
            .scalar_cost_f64(resids.as_slice().to_vec()))
    }

    pub fn print_initial_loss(&self) {
        let subprob_params = self.subprob_initial_params_optspace();
        let full_params =
            self.optspace_fullprob_input_from_subprob_input(&subprob_params.as_slice().to_vec());
        let loss = self.loss_fn_engine.derivative(&full_params);
        println!("Loss and gradient for first sub-problem: {:?}", loss);
        match self.initial_params_cost() {
            Ok(cost) => println!("Initial cost for sub-problem: {}", cost),
            Err(e) => println!("Error computing initial cost: {}", e),
        }
    }
}
