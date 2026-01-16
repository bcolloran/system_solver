use crate::{
    dynamics::{
        air::air_net_force_2d,
        ground::ground_net_force_2d,
        // test_params::{DynamicsDerivedParams, DynamicsGivenParams},
    },
    prelude::*,
};
use system_solver::prelude::{ad_trait::AD, nalgebra::Vector2};

pub mod air;
pub mod ground;
pub mod wall_and_slope;

pub mod state;
pub mod test_params;

/// Combined net force while grounded:
/// air terms (thrust + drag) + ground terms (drive + rolling).
pub fn total_force_2d<T: AD>(
    s: &DynamicsState<T>,
    givens: &DynamicsGivenParams<T>,
    unknowns: &DynamicsDerivedParams<T>,
) -> Vector2<T> {
    let mut f = air_net_force_2d(s, givens, unknowns);

    if let Some(contact) = s.contact {
        let glove_force = if contact.abs_tangent_angle_degrees() > givens.sticky_glove_angle_deg {
            unknowns.sticky_glove_force
        } else {
            T::zero()
        };

        f += ground_net_force_2d(
            s.input,
            contact,
            unknowns.run_drag_coeff,
            unknowns.run_force_max,
            glove_force,
        );
    }

    f
}

/// Combined acceleration while grounded: (F_air + F_ground)/m + gravity.
#[inline]
pub fn total_accel_2d<T: AD>(
    s: &DynamicsState<T>,
    givens: &DynamicsGivenParams<T>,
    unknowns: &DynamicsDerivedParams<T>,
) -> Vector2<T> {
    let mut f = total_force_2d(s, givens, unknowns) / givens.mass;
    // add gravitational acceleration
    f.y += unknowns.g;
    f
}
