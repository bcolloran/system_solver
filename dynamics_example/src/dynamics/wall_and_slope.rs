use ad_trait::AD;

use crate::{dynamics::state::DynamicsState, prelude::*, total_accel_2d};

pub fn wall_slide_accel_at_wall_terminal_vel_residual<T: AD>(
    givens: &DynamicsGivenParams<T>,
    unknowns: &DynamicsDerivedParams<T>,
) -> T {
    let mut s0 = DynamicsState::new_zeroed();
    s0.vel.y = givens.wall_slide_terminal_vel;
    // contact on vertical wall
    s0.contact = Some(FrictionContact2D::new_equilibrium_contact_from_angle(
        T::constant(std::f64::consts::FRAC_PI_2),
        s0.vel,
        unknowns.g,
        givens.mass,
    ));

    let a = total_accel_2d(&s0, givens, unknowns);

    // We want the y acceleration at terminal wall slide speed to be zero.
    a.y
}
