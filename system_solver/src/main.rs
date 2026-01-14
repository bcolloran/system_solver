use player_dynamics::prelude::*;
use player_dynamics::{
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
use std::rc::Rc;

use param_solver::prelude::*;

fn main() {
    let givens = DynamicsGivenParams {
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

    let residual_fns = residual_fns![
        air_no_accel_at_max_air_speed_in_zero_g_residual,
        air_time_to_95pct_max_air_speed_in_zero_g_residual,
        jump_height_residual,
        jump_vel_at_peak_residual,
        jump_return_to_ground_in_time_down,
        run_accel_at_max_speed_residual,
        run_time_to_95pct_max_speed_residual,
        wall_slide_accel_at_wall_terminal_vel_residual,
    ];

    let eq_sys = EquationSystemBuilder::new(givens, residual_fns).unwrap();
    let eq_sys = eq_sys.with_triangularization(&unknowns).unwrap();
    eq_sys.print_lower_tri_mat();
    eq_sys.print_solution_plan();
    eq_sys.solve_system(&unknowns).unwrap();
}
