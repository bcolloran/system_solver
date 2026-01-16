use crate::prelude::{opt_tools::MyObserver, *};
use ad_trait::forward_ad::adfn::adfn;
use argmin::{core::Executor, solver::simulatedannealing::SimulatedAnnealing};

/// Configuration for the annealing proposal (in *optimization space*, e.g. log-space).
#[derive(Clone, Debug)]
pub struct SimulatedAnnealingConfig {
    /// Should match the `temp` you pass to `SimulatedAnnealing::new(temp)`.
    pub init_temp: f64,

    /// Small-move half-width at `init_temp` (uniform in [-small_step, small_step]).
    pub small_step_init: f64,
    /// Small-move half-width as temp -> 0.
    pub small_step_min: f64,

    /// Big-jump scale at `init_temp` (Cauchy / heavy-tailed).
    ///
    /// In log-space, `ln(10)` is about a 10x multiplicative jump in exp-linked model space.
    pub big_step_init: f64,
    /// Big-jump scale as temp -> 0.
    pub big_step_min: f64,

    /// Probability of using a big jump at `init_temp`.
    pub p_big_init: f64,
    /// Probability of using a big jump as temp -> 0.
    pub p_big_min: f64,

    /// Safety clamp on absolute per-coordinate change (limits extreme Cauchy draws).
    pub max_abs_step: f64,

    /// Optional: max gradient drift scale to use for gradient-informed proposals. If `None`, gradient drift is disabled.
    pub grad_drift_max: Option<f64>,
}

impl Default for SimulatedAnnealingConfig {
    fn default() -> Self {
        Self {
            init_temp: 100.0,
            small_step_init: 0.25,
            small_step_min: 0.01,
            // Default big step is about a 10x multiplicative jump in model space
            big_step_init: std::f64::consts::LN_10, // ~2.302585
            big_step_min: 0.10,
            p_big_init: 0.30,
            p_big_min: 0.02,
            // Default max absolute step size targets about a 100x multiplicative jump in model space
            max_abs_step: 100f64.ln(),
            grad_drift_max: Some(1.0), // set > 0.0 to enable (and compile with feature "sa_grad")
        }
    }
}

impl<G64, U64, Gadfn, Uadfn, R, A, const N: usize> SubProblem<G64, U64, Gadfn, Uadfn, R, A, N>
where
    G64: GivenParamsFor<f64, N>,
    U64: UnknownParamsFor<f64, N>,
    Gadfn: GivenParamsFor<adfn<1>, N>,
    Uadfn: UnknownParamsFor<adfn<1>, N>,
    R: ResidTransHOF,
    A: ResidAggFnToScalarGen,
{
    pub fn solve_simulated_annealing(&self) -> Result<U64, EqSysError> {
        self.print_pre_optimization_summary();

        let optspace_params = self.subprob_initial_params_optspace().clone();

        let temp = self
            .sa_cfg
            .as_ref()
            .expect("Simulated annealing config (sa_cfg) not set on annealing SubProblem")
            .init_temp;

        // Set up simulated annealing solver
        // An alternative random number generator (RNG) can be provided to `new_with_rng`:
        // SimulatedAnnealing::new_with_rng(temp, Xoshiro256PlusPlus::try_from_os_rng()?)?
        let solver = SimulatedAnnealing::new(temp)?
            // Optional: Define temperature function (defaults to `SATempFunc::TemperatureFast`)
            // .with_temp_func(SATempFunc::Boltzmann)
            /////////////////////////
            // Stopping criteria   //
            /////////////////////////
            // Optional: stop if there was no new best solution after 1000 iterations
            .with_stall_best(1000)
            // Optional: stop if there was no accepted solution after 1000 iterations
            .with_stall_accepted(1000);
        /////////////////////////
        // Reannealing         //
        /////////////////////////
        // // Optional: Reanneal after 1000 iterations (resets temperature to initial temperature)
        // .with_reannealing_fixed(1000)
        // // Optional: Reanneal after no accepted solution has been found for `iter` iterations
        // .with_reannealing_accepted(500)
        // // Optional: Start reannealing after no new best solution has been found for 800 iterations
        // .with_reannealing_best(800);

        println!(
            "Sub-problem {} initial params (opt space): {:?}",
            self.block.block_idx, optspace_params
        );

        let observer = MyObserver::new();

        let opt_result = Executor::new(self.clone(), solver)
            .configure(|state| {
                state
                    .param(optspace_params)
                    // Optional: Set maximum number of iterations (defaults to `std::u64::MAX`)
                    .max_iters(10_000)
                    // Optional: Set target cost function value (defaults to `std::f64::NEG_INFINITY`)
                    .target_cost(0.0)
            })
            // Optional: Attach a observer
            .add_observer(
                observer.clone(),
                argmin::core::observers::ObserverMode::NewBest,
            )
            .run()?;

        self.print_post_optimization_summary(&opt_result);
        // println!("Cost history: {:?}", observer.cost_history());

        let best_params_optspace_subprob = opt_result
            .state
            .best_param
            .as_ref()
            .expect("must have best param");

        let best_params_vec: Vec<f64> = best_params_optspace_subprob.as_slice().to_vec();

        Ok(self.modspace_to_params(&self.optspace_to_modspace(
            &self.optspace_fullprob_input_from_subprob_input(&best_params_vec),
        )))
    }
}
