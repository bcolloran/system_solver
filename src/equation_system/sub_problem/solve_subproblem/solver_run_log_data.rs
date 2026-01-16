use argmin::core::{TerminationReason, TerminationStatus};
use nalgebra::DVector;

pub struct SolverRunPostOptLogData<U> {
    pub termination_status: TerminationStatus,
    pub termination_reason: TerminationReason,
    pub best_cost: f64,
    pub best_params: U,
    pub opt_space_grad_at_best_params: DVector<f64>,
    pub cost_history: Vec<f64>,
}

pub struct SolverRunLogData<U> {
    pub solver_name: String,
    pub best_cost_pre: f64,
    pub input_params: U,
    pub opt_space_grad_at_input: DVector<f64>,
    pub post_run_data: Option<SolverRunPostOptLogData<U>>,
}
