use argmin::core::{TerminationReason, TerminationStatus};
use nalgebra::DVector;
use player_dynamics::DynamicsDerivedParams;

pub struct SolverRunPostOptLogData {
    pub termination_status: TerminationStatus,
    pub termination_reason: TerminationReason,
    pub best_cost: f64,
    pub best_params: DynamicsDerivedParams<f64>,
    pub opt_space_grad_at_best_params: DVector<f64>,
    pub cost_history: Vec<f64>,
}

pub struct SolverRunLogData {
    pub solver_name: String,
    pub best_cost_pre: f64,
    pub input_params: DynamicsDerivedParams<f64>,
    pub opt_space_grad_at_input: DVector<f64>,
    pub post_run_data: Option<SolverRunPostOptLogData>,
}
