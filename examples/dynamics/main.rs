use ad_trait::forward_ad::adfn::adfn;
use dynamics_example::prelude::*;
use dynamics_example::{
    constraints::{
        aerial::{
            air_no_accel_at_max_air_speed_in_zero_g_residual,
            air_time_to_95pct_max_air_speed_in_zero_g_residual,
        },
        jump::{
            jump_height_residual, jump_return_to_ground_in_time_down, jump_vel_at_peak_residual,
        },
        run::{run_accel_at_max_speed_residual, run_time_to_95pct_max_speed_residual},
    },
    dynamics::wall_and_slope::wall_slide_accel_at_wall_terminal_vel_residual,
};

use system_solver::{prelude::*, residual_fns_for_generic_params};

// Static field names for the unknowns - required for 'static lifetime
static UNKNOWN_FIELD_NAMES: &[&str] = &[
    "air_drag_coeff",
    "air_thrust_max",
    "g",
    "jump_vy_0",
    "jump_boost_force",
    "run_force_max",
    "run_drag_coeff",
    "sticky_glove_force",
];

fn main() {
    let givens_f64 = DynamicsGivenParams {
        mass: 55.5,

        jump_height: 3.3,
        jump_time_up: 0.5,
        jump_time_down: 0.4,

        max_vel_run: 12.2,
        time_to_95pct_max_vel_run: 0.2,
        x_stop_speed_threshold: 0.1,

        max_air_speed_x: 15.8,
        time_to_95pct_max_air_speed_x: 0.3,

        wall_slide_terminal_vel: -4.4,
        sticky_glove_angle_deg: 25.0,
    };

    // Convert givens to adfn<1> version for automatic differentiation
    let givens_adfn: DynamicsGivenParams<adfn<1>> = givens_f64.to_ad();

    let unknowns = DynamicsDerivedParams {
        // analytic solution that we want to converge to: air_drag_coeff=38.509
        air_drag_coeff: 0.2,

        // analytic solution that we want to converge to: air_thrust_max=2982.14
        air_thrust_max: 2252.1212,

        g: -9.81252,
        jump_vy_0: 5.235235,
        jump_boost_force: 50.235235,

        run_force_max: 30.235235,
        run_drag_coeff: 0.498797,

        sticky_glove_force: 200.986967,
    };

    // Use the macro to create ResidualFns with correctly monomorphized function pointers
    let residual_fns = residual_fns_for_generic_params!(
        DynamicsGivenParams, DynamicsDerivedParams;
        air_no_accel_at_max_air_speed_in_zero_g_residual,
        air_time_to_95pct_max_air_speed_in_zero_g_residual,
        jump_height_residual,
        jump_vel_at_peak_residual,
        jump_return_to_ground_in_time_down,
        run_accel_at_max_speed_residual,
        run_time_to_95pct_max_speed_residual,
        wall_slide_accel_at_wall_terminal_vel_residual
    );

    let eq_sys =
        EquationSystemBuilder::new(givens_f64, givens_adfn, residual_fns, UNKNOWN_FIELD_NAMES)
            .unwrap();
    let eq_sys = eq_sys.with_triangularization(&unknowns).unwrap();
    eq_sys.print_lower_tri_mat();
    eq_sys.print_solution_plan();
    eq_sys.solve_system(&unknowns).unwrap();
}
