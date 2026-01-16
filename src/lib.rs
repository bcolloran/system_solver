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
            param_traits::*,
            residuals::*,
            residuals::{aggregation_hof::*, transformation_hof::*},
            solution_plan::*,
            sub_problem::*,
        },
        error::*,
        residual_fns, residual_fns_for_generic_params,
    };

    pub use ad_trait;
    pub use field_names_and_counts;
    pub use nalgebra;
    pub use struct_to_array;
}

pub use field_names_and_counts::FieldNames;
pub use struct_to_array::StructToArray;
