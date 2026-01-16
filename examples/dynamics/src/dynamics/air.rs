use crate::prelude::*;
use system_solver::prelude::{ad_trait::AD, nalgebra::Vector2};

/// Note in real code, input is guaranteed to be constrained to unit disk, so not worried about large diagonal inputs.
pub fn air_thrust_2d<T: AD>(input: Vector2<T>, max_air_thrust: T) -> Vector2<T> {
    if input.norm_squared() < T::constant(1e-6) {
        return Vector2::zeros();
    };
    input * max_air_thrust
}

pub fn air_thrust_horizontal<T: AD>(input: Vector2<T>, max_air_thrust: T) -> Vector2<T> {
    if input.x.abs() < T::constant(1e-6) {
        return Vector2::zeros();
    };
    let mut thrust = input.clone() * max_air_thrust;
    thrust.y = T::zero();
    thrust
}

pub fn air_drag_quadratic_2d<T: AD>(vel: Vector2<T>, drag_coefficient: T) -> Vector2<T> {
    -vel * drag_coefficient * vel.norm()
}

pub fn air_drag_linear_2d<T: AD>(vel: Vector2<T>, drag_coefficient: T) -> Vector2<T> {
    -vel * drag_coefficient
}

/// Note in real code, input is guaranteed to be constrained to unit disk, so not worried about large diagonal inputs.
pub fn air_net_force_2d<T: AD>(
    s: &DynamicsState<T>,
    _givens: &DynamicsGivenParams<T>,
    unknowns: &DynamicsDerivedParams<T>,
) -> Vector2<T> {
    // alwways apply drag
    let mut f = air_drag_quadratic_2d(s.vel, unknowns.air_drag_coeff);

    // apply air thrust only when there is no contact
    if s.contact.is_none() {
        f += air_thrust_horizontal(s.input, unknowns.air_thrust_max);
    }

    // apply jump boost force if active
    if s.jump_boost_active {
        let jump_boost_force = Vector2::new(T::zero(), unknowns.jump_boost_force);
        f += jump_boost_force;
    }
    f
}

/// Note in real code, input is guaranteed to be constrained to unit disk, so not worried about large diagonal inputs.
pub fn air_accel_2d<T: AD>(
    s: &DynamicsState<T>,
    givens: &DynamicsGivenParams<T>,
    unknowns: &DynamicsDerivedParams<T>,
) -> Vector2<T> {
    let net_force = air_net_force_2d(s, givens, unknowns);
    net_force / givens.mass + Vector2::new(T::constant(0.0), unknowns.g)
}
