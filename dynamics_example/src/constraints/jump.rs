use ad_trait::AD;

use crate::{
    air_accel_2d,
    constraints::integrate::{IntegrationState, step_state_to_t_with_acc_fn},
    prelude::*,
};

pub fn jump_height_residual<T: AD>(
    givens: &DynamicsGivenParams<T>,
    unknowns: &DynamicsDerivedParams<T>,
) -> T {
    let mut s0 = IntegrationState::new_zeroed();
    s0.state.vel.y = unknowns.jump_vy_0;
    s0.state.jump_boost_active = true;

    let s_end = step_state_to_t_with_acc_fn(
        air_accel_2d,
        s0,
        givens,
        unknowns,
        T::constant(0.01),
        givens.jump_time_up,
    );

    let y_at_t_up = s_end.unwrap().pos.y;
    // We want the height at t_up to be equal to the desired jump height.
    y_at_t_up - givens.jump_height
}

pub fn jump_vel_at_peak_residual<T: AD>(
    givens: &DynamicsGivenParams<T>,
    unknowns: &DynamicsDerivedParams<T>,
) -> T {
    let mut s0 = IntegrationState::new_zeroed();
    s0.state.vel.y = unknowns.jump_vy_0;
    s0.state.jump_boost_active = true;

    let s_end = step_state_to_t_with_acc_fn(
        air_accel_2d,
        s0,
        givens,
        unknowns,
        T::constant(0.01),
        givens.jump_time_up,
    );

    // We want the vertical velocity at the peak to be zero.
    s_end.unwrap().state.vel.y
}

pub fn jump_return_to_ground_in_time_down<T: AD>(
    givens: &DynamicsGivenParams<T>,
    unknowns: &DynamicsDerivedParams<T>,
) -> T {
    let mut s0 = IntegrationState::new_zeroed();
    s0.pos.y = givens.jump_height;

    let s_end = step_state_to_t_with_acc_fn(
        air_accel_2d,
        s0,
        givens,
        unknowns,
        T::constant(0.01),
        givens.jump_time_down,
    );

    // We want the vertical position at time_down to be zero (ground level).
    s_end.unwrap().pos.y
}
