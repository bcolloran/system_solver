use ad_trait::AD;

use crate::{
    FrictionContact2D,
    constraints::{
        input_max_x_positive,
        integrate::{IntegrationState, step_state_to_t_on_flat_ground_with_acc_fn},
    },
    dynamics::state::DynamicsState,
    prelude::*,
    total_accel_2d,
};

pub fn run_accel_at_max_speed_residual<T: AD>(
    givens: &DynamicsGivenParams<T>,
    unknowns: &DynamicsDerivedParams<T>,
) -> T {
    let mut s0 = DynamicsState::new_zeroed();
    s0.vel.x = givens.max_vel_run;
    s0.input = input_max_x_positive();
    // Ground contact on flat ground
    // at this equilibrium, vel along tangent should  be max_ground_speed
    s0.contact = Some(FrictionContact2D::new_equilibrium_contact_from_angle(
        T::constant(0.0),
        s0.vel,
        unknowns.g,
        givens.mass,
    ));

    let a = total_accel_2d(&s0, givens, unknowns);

    // We want the x acceleration at max speed to be zero.
    a.x
}

pub fn run_time_to_95pct_max_speed_residual<T: AD>(
    givens: &DynamicsGivenParams<T>,
    unknowns: &DynamicsDerivedParams<T>,
) -> T {
    let mut s0 = IntegrationState::new_zeroed();
    s0.state.input = input_max_x_positive();

    let s_t = step_state_to_t_on_flat_ground_with_acc_fn(
        total_accel_2d,
        s0,
        givens,
        unknowns,
        T::constant(0.01),
        givens.time_to_95pct_max_vel_run,
    );

    let v_end = s_t.unwrap().state.vel.x;
    // We want the velocity at t95 to be 95% of max ground speed.
    v_end - givens.max_vel_run * T::constant(0.95)
}
