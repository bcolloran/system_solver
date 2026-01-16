pub mod gauss_newton;
pub mod lbfgs;
pub mod simulated_annealing;
pub mod solver_run_log_data;

use ad_trait::forward_ad::adfn::adfn;
use argmin::core::{Operator, State};

use crate::prelude::*;

impl<G64, U64, Gadfn, Uadfn, R, A, const N: usize> SubProblem<G64, U64, Gadfn, Uadfn, R, A, N>
where
    G64: GivenParamsFor<f64, N>,
    U64: UnknownParamsFor<f64, N>,
    Gadfn: GivenParamsFor<adfn<1>, N>,
    Uadfn: UnknownParamsFor<adfn<1>, N>,
    R: ResidTransHOF,
    A: ResidAggHOF,
{
    fn print_pre_optimization_summary(&self) {
        println!(
            "\n------- pre optimization (block {})-------",
            self.block.block_idx
        );
        println!("Initial params (model space): {:#?}", self.initial_unknowns);
        println!(
            "Initial unknowns (opt space): {:?}",
            self.subprob_initial_params_optspace()
        );
        println!("initial cost: {:.4e}", self.initial_params_cost().unwrap());

        self.apply(&self.subprob_initial_params_optspace())
            .map_or_else(
                |err| println!("initial gradient computation error: {:?}", err),
                |grad| println!("initial gradient: {:?}", grad.as_slice()),
            );
    }

    fn print_post_optimization_summary<S, Gr, J>(&self, opt_res: &OptRes<S, G64, U64, Gadfn, Uadfn, R, A, N, Gr, J>) {
        println!(
            "------- post optimization (block {})-------",
            self.block.block_idx
        );
        println!("  solver: {}", tynm::type_name::<S>());
        println!(
            "    stop status: {:?} at iteration {}",
            opt_res.state.get_termination_status(),
            opt_res.state.get_iter()
        );
        println!(
            "    stop reason: {:?}",
            opt_res.state.get_termination_status()
        );
        println!(
            "Best cost: {:.6e} (prev: {:.6e})",
            opt_res.state.best_cost, opt_res.state.prev_best_cost
        );

        let best_params_optspace_subprob = opt_res
            .state
            .best_param
            .as_ref()
            .expect("must have best param");
        println!(
            "Best params (opt space): {:?}",
            best_params_optspace_subprob
        );
        self.apply(&best_params_optspace_subprob).map_or_else(
            |err| println!("gradient computation error: {:?}", err),
            |grad| {
                println!(
                    "gradient at best_params_opt_space: {:?};       norm: {:.6e}",
                    grad.as_slice(),
                    grad.iter().map(|x| x * x).sum::<f64>().sqrt()
                )
            },
        );

        let best_params_optspace_fullprob = self.optspace_fullprob_input_from_subprob_input(
            &best_params_optspace_subprob.as_slice().to_vec(),
        );

        let best_params_modspace_fullprob =
            self.optspace_to_modspace(&best_params_optspace_fullprob);

        println!(
            "Best unknowns (POST): {:#?}",
            self.modspace_to_params(&best_params_modspace_fullprob)
        );
    }
}
