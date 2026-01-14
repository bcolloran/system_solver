use crate::prelude::{opt_tools::MyObserver, *};
use argmin::{
    core::Executor,
    solver::{
        linesearch::{BacktrackingLineSearch, condition::ArmijoCondition},
        quasinewton::LBFGS,
    },
};

impl<R, A> SubProblem<R, A>
where
    R: ResidTransHOF,
    A: ResidAggFnToScalarGen,
{
    pub fn solve_lbfgs(&self) -> Result<DynamicsDerivedParams<f64>, EqSysError> {
        self.print_pre_optimization_summary();

        let linesearch: BacktrackingLineSearch<
            nalgebra::DVector<f64>,
            nalgebra::DVector<f64>,
            _,
            _,
        > = BacktrackingLineSearch::new(ArmijoCondition::new(1e-4f64)?).rho(0.5f64)?;
        let solver = LBFGS::new(linesearch, 10);
        let max_iters = 10000;

        let optspace_params = self.subprob_initial_params_optspace().clone();

        println!(
            "Sub-problem {} initial params (opt space): {:?}",
            self.block.block_idx, optspace_params
        );

        let observer = MyObserver::new();
        let opt_result = Executor::new(self.clone(), solver)
            .configure(|state| state.param(optspace_params).max_iters(max_iters))
            .add_observer(
                observer.clone(),
                argmin::core::observers::ObserverMode::Always,
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
