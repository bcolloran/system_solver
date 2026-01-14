// mod system_decomposition;
pub mod equation_system;
pub mod error;

pub mod prelude {
    pub use crate::{
        equation_system::{
            EqSysSolutionPlan, EqSysStateInit, EquationSystemBuilder,
            objective::*,
            opt_tools::{self, *},
            param_scaling::*,
            residuals::*,
            residuals::{aggregation_hof::*, transformation_hof::*},
            solution_plan::*,
            sub_problem::*,
        },
        error::*,
        residual_fns,
    };

    pub use player_dynamics::prelude::*;

    pub use struct_to_array::StructToArray;
}
