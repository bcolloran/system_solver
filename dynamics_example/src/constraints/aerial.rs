use ad_trait::AD;

use crate::{
    air_accel_2d,
    constraints::{
        input_max_x_positive,
        integrate::{IntegrationState, step_state_to_t_with_acc_fn},
    },
    prelude::*,
};

/// Residual: in zero-g and under full input, acceleration should be zero at max air speed
///
/// Note to keep it simple, we model this constraint with horizontal air speed under the assumption that there is no gravity affecting motion, i.e. when the character is moving in a zero-gravity environment, or has anti-grav enabled. This is the setting in which we want to tune air drag coeff to achieve a desired max air speed.
pub fn air_no_accel_at_max_air_speed_in_zero_g_residual<T: AD>(
    givens: &DynamicsGivenParams<T>,
    unknowns: &DynamicsDerivedParams<T>,
) -> T {
    let mut s0 = DynamicsState::new_zeroed();
    s0.vel.x = givens.max_air_speed_x;
    s0.input = input_max_x_positive();
    let mut unknowns = unknowns.clone();

    // // zero out gravity to isolate horizontal acceleration
    unknowns.g = T::zero();

    let a = air_accel_2d(&s0, givens, &unknowns);
    // We want the acceleration at max speed to be zero.
    a.x
}

/// Residual: in zero-g and under full input, time to reach 95% of max air speed x should match desired time
///
/// Note to keep it simple, we model this constraint with horizontal air speed under the assumption that there is no gravity affecting motion, i.e. when the character is moving in a zero-gravity environment, or has anti-grav enabled. This is the setting in which we want to tune air drag coeff to achieve a desired max air speed.
pub fn air_time_to_95pct_max_air_speed_in_zero_g_residual<T: AD>(
    givens: &DynamicsGivenParams<T>,
    unknowns: &DynamicsDerivedParams<T>,
) -> T {
    let mut s0 = IntegrationState::new_zeroed();
    s0.state.input = input_max_x_positive();
    let mut unknowns = unknowns.clone();

    // // zero out gravity to isolate horizontal acceleration
    unknowns.g = T::zero();

    let s_end = step_state_to_t_with_acc_fn(
        air_accel_2d,
        s0,
        givens,
        &unknowns,
        T::constant(1.0 / 300.0),
        givens.time_to_95pct_max_air_speed_x,
    );

    let v_end = s_end.unwrap().state.vel.x;
    // We want the velocity at t95 to be 95% of max air speed x.
    v_end - givens.max_air_speed_x * T::constant(0.95)
}
