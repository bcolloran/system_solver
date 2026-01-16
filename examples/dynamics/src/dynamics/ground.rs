use crate::prelude::*;
use system_solver::prelude::{
    ad_trait::AD,
    nalgebra::{UnitVector2, Vector2},
};

/// Estimates the normal force from gravity for a surface contact.
///
/// This is a simplified model that works well for floors and slopes where the
/// object is resting on the surface due to gravity. It does NOT work for:
/// - Walls or ceilings (where normal force comes from other sources)
/// - Adhesion or magnetic forces
/// - Active forces pushing the object into the surface
///
/// # Arguments
/// * `mass` - Object mass in kg
/// * `gravity_acc_y` - Gravity acceleration in m/s² (typically negative)
/// * `surface_normal` - Outward unit normal from the surface
///
/// # Returns
/// Estimated normal force in Newtons (always >= 0)
///
/// # Physics
/// For a surface with normal `n` and gravity force `F_g = (0, m*g)`:
/// ```text
/// N = -F_g · n = -m*g*(n.y)
/// ```
/// When the normal points "up" (n.y > 0) and gravity is negative,
/// this gives a positive normal force.
pub fn estimate_normal_force_from_gravity<T: AD>(
    mass: T,
    gravity_acc_y: T,
    surface_normal: UnitVector2<T>,
) -> T {
    // gravity force vector in Newtons
    let fg = Vector2::new(T::constant(0.0), mass * gravity_acc_y);

    // If normal points "up-ish" and gravity is negative Y, -fg·n will be positive.
    let n = surface_normal.into_inner();
    // Clamp to >= 0 to avoid negative normal forces on ceilings/walls.
    (-fg.dot(&n)).max(T::constant(0.0))
}

#[cfg(test)]
mod ground_contact_helper_tests {
    use crate::assert_approx_eq;

    use super::*;
    use std::f32::consts::FRAC_1_SQRT_2;

    #[test]
    fn test_estimate_normal_force_flat_ground() {
        let mass = 2.0;
        let gravity_acc_y = -10.0;
        let normal = UnitVector2::new_normalize(Vector2::new(0.0, 1.0));

        let n = estimate_normal_force_from_gravity(mass, gravity_acc_y, normal) as f64;
        assert!((n - 20.0).abs() < 1.0e-6);
    }

    #[test]
    fn test_estimate_normal_force_45_degree_slope() {
        // Normal for a slope at +45° (tangent ~ (cos, sin)) can be (-sin, cos)
        let mass = 2.0;
        let gravity_acc_y = -10.0;
        let normal = UnitVector2::new_normalize(Vector2::new(-FRAC_1_SQRT_2, FRAC_1_SQRT_2));

        let n = estimate_normal_force_from_gravity(mass, gravity_acc_y, normal);
        // mg*cos(45) = 20 * 0.7071...
        assert_approx_eq!(n, 20.0 * FRAC_1_SQRT_2);
    }

    #[test]
    fn test_ground_contact_new_projects_velocity() {
        let normal = UnitVector2::new_normalize(Vector2::new(0.0, 1.0));
        // Give velocity with both tangent and normal components
        let vel = Vector2::new(3.0, 2.0);
        let contact = FrictionContact2D::new(normal, vel, 100.0, 1.0);

        // Only tangent component should remain (3.0, 0.0)
        assert_approx_eq!(contact.tangent_relative_velocity().x as f64, 3.0);
        assert_approx_eq!(contact.tangent_relative_velocity().y as f64, 0.0);
    }

    #[test]
    fn test_ground_contact_tangent_perpendicular_to_normal() {
        let normal = UnitVector2::new_normalize(Vector2::new(0.6, 0.8));
        let contact = FrictionContact2D::new(normal, Vector2::zeros(), 100.0, 1.0);

        let tangent = contact.tangent();
        let dot = normal.dot(&tangent);
        assert_approx_eq!(dot as f64, 0.0);
    }
}

/// Computes the "throttle" from player input projected onto the ground tangent.
///
/// # Arguments
/// * `input` - 2D player input vector (typically from joystick or keys)
/// * `ground_tangent` - Unit tangent vector of the ground surface
///
/// # Returns
/// Scalar throttle value in [-1, 1] representing desired motion along the surface
///
/// # Examples
/// ```
/// use nalgebra::{UnitVector2, Vector2};
/// use dynamics_example::dynamics::ground::ground_throttle_from_input;
///
/// // Horizontal ground, input points right
/// let tangent = UnitVector2::new_normalize(Vector2::new(1.0, 0.0));
/// let input = Vector2::new(0.5, 0.0);
/// assert_eq!(ground_throttle_from_input(input, tangent), 0.5);
///
/// // Input perpendicular to ground gives zero throttle
/// let input = Vector2::new(0.0, 1.0);
/// assert_eq!(ground_throttle_from_input(input, tangent), 0.0);
/// ```
pub fn ground_throttle_from_input<T: AD>(input: Vector2<T>, ground_tangent: UnitVector2<T>) -> T {
    input
        .dot(&ground_tangent.into_inner())
        .clamp(T::constant(-1.0), T::constant(1.0))
}

/// Computes the motive force along the ground tangent, limited by available traction.
///
/// This function models wheel/track drive force with Coulomb friction limits:
/// - Desired force = throttle × ground_force_max
/// - Maximum available force = μ_trac × N (traction coefficient × normal force)
/// - Actual force = min(desired, available)
///
/// # Arguments
/// * `input` - 2D player input vector
/// * `contact` - Ground contact information (normal, velocity, normal force)
/// * `ground_force_max` - Maximum force the drive system can produce (N)
/// * `traction_coeff` - Traction coefficient (dimensionless, typically 0.5-1.5)
///
/// # Returns
/// 2D force vector in Newtons, directed along the surface tangent
///
/// # Physics
/// The traction limit models the friction circle concept:
/// ```text
/// F_drive_max = μ_trac × N
/// ```
/// where N is the normal force. This means:
/// - More normal force = more traction
/// - Zero normal force = no traction (airborne)
/// - The coefficient μ_trac depends on surface properties
///
pub fn ground_drive_force_2d<T: AD>(
    input: Vector2<T>,
    contact: FrictionContact2D<T>,
    ground_force_max: T,
    sticky_glove_force: T,
) -> Vector2<T> {
    debug_assert!(ground_force_max >= T::constant(0.0));

    let throttle = ground_throttle_from_input(input, contact.tangent());
    if throttle.abs() < T::constant(1.0e-6) {
        return Vector2::zeros();
    }

    // No normal force -> no traction.
    let n = contact.normal_force_mag().max(T::constant(0.0));

    let traction_coeff = contact.traction_coeff();

    contact.tangent().into_inner()
        * ground_force_max
        * throttle
        * traction_coeff
        * (n + sticky_glove_force)
}

#[cfg(test)]
mod ground_drive_force_tests {
    use super::*;
    use test_case::test_case;

    #[test]
    fn test_ground_throttle_from_input_clamps() {
        assert_eq!(
            ground_throttle_from_input(
                Vector2::new(2.0, 0.0),
                UnitVector2::new_normalize(Vector2::new(1.0, 0.0))
            ),
            1.0
        );
        assert_eq!(
            ground_throttle_from_input(
                Vector2::new(-2.0, 0.0),
                UnitVector2::new_normalize(Vector2::new(1.0, 0.0))
            ),
            -1.0
        );
        assert_eq!(
            ground_throttle_from_input(
                Vector2::new(0.25, 0.0),
                UnitVector2::new_normalize(Vector2::new(1.0, 0.0))
            ),
            0.25
        );
    }

    // #[test]
    // fn test_ground_drive_force_saturates_by_traction() {
    //     // Flat ground frame
    //     let normal = UnitVector2::new_normalize(Vector2::new(0.0, 1.0));
    //     // velocity doesn't matter for this test
    //     let vel = Vector2::new(0.0, 0.0);

    //     let contact = GroundContact2D::new(normal, vel, 100.0, 1.0); // N=100
    //     let traction_coeff = 0.5; // traction max = 50N
    //     let engine_max = 200.0; // desired 200N -> clamp to 50N

    //     let f = ground_drive_force_2d(Vector2::new(1.0, 0.0), contact, engine_max, traction_coeff);
    //     assert!((f.x - 50.0).abs() < 1e-6);
    //     assert!((f.y - 0.0).abs() < 1e-6);

    //     let f_rev =
    //         ground_drive_force_2d(Vector2::new(-1.0, 0.0), contact, engine_max, traction_coeff);
    //     assert!((f_rev.x + 50.0).abs() < 1e-6);
    //     assert!((f_rev.y - 0.0).abs() < 1e-6);
    // }
    /// Condition should hold regardless of input or velocity.
    /// All tests with v_y==0
    #[test_case((0.0, 1.0, 0.0); "zero velocity, pos y input")]
    #[test_case((0.0, -1.0, 0.0); "zero velocity, neg y input")]
    #[test_case((1.0, 0.0, 5.0); "pos x velocity, pos x input")]
    #[test_case((-1.0, 0.0, 5.0); "pos x velocity, neg x input")]
    #[test_case((1.0, 0.0, -5.0); "neg x velocity, pos x input")]
    #[test_case((-1.0, 0.0, -5.0); "neg x velocity, neg x input")]
    #[test_case((0.5, 0.5, 3.0); "pos x velocity, pos xy input")]
    fn test_ground_drive_force_zero_when_no_normal_force((u_x, u_y, velocity_x): (f32, f32, f32)) {
        // Flat ground frame
        let normal = UnitVector2::new_normalize(Vector2::new(0.0, 1.0));
        let vel = Vector2::new(velocity_x, 0.0);
        let contact = FrictionContact2D::new(normal, vel, 0.0, 1.0);
        let input = Vector2::new(u_x, u_y);

        let f = ground_drive_force_2d(input, contact, 100.0, 1.0);
        assert_eq!(f, Vector2::zeros());
    }

    #[test]
    fn test_ground_drive_force_respects_input_direction() {
        let normal = UnitVector2::new_normalize(Vector2::new(0.0, 1.0));
        let contact = FrictionContact2D::new(normal, Vector2::zeros(), 100.0, 1.0);

        // Forward input
        let f_fwd = ground_drive_force_2d(Vector2::new(0.5, 0.0), contact, 100.0, 1.0);
        assert!(f_fwd.x > 0.0);

        // Backward input
        let f_back = ground_drive_force_2d(Vector2::new(-0.5, 0.0), contact, 100.0, 1.0);
        assert!(f_back.x < 0.0);

        let fwd = f_fwd.x as f64;
        let back = f_back.x as f64;

        // Magnitudes should be equal
        assert_approx_eq!(fwd.abs(), back.abs());
    }

    // #[test]
    // fn test_ground_drive_force_scales_with_throttle() {
    //     let normal = UnitVector2::new_normalize(Vector2::new(0.0, 1.0));
    //     let contact = GroundContact2D::new(normal, Vector2::zeros(), 100.0, 1.0);
    //     let traction_coeff = 1.0; // traction max = 100N
    //     let engine_max = 50.0;

    //     let f_half =
    //         ground_drive_force_2d(Vector2::new(0.5, 0.0), contact, engine_max, traction_coeff);
    //     let f_full =
    //         ground_drive_force_2d(Vector2::new(1.0, 0.0), contact, engine_max, traction_coeff);

    //     // Half throttle should give half force
    //     assert_approx_eq!(f_half.x, 25.0);
    //     assert_approx_eq!(f_full.x, 50.0);
    // }

    #[test]
    fn test_ground_drive_force_on_slope() {
        // 45° slope: normal points up-left
        let normal = UnitVector2::new_normalize(Vector2::new(
            -std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
        ));
        let contact = FrictionContact2D::new(normal, Vector2::zeros(), 100.0, 1.0);

        // Input to the right should produce force along slope tangent
        let force = ground_drive_force_2d(Vector2::new(1.0, 0.0), contact, 50.0, 1.0);

        // Force should be along tangent (perpendicular to normal)
        let dot = force.dot(&normal.into_inner());
        assert_approx_eq!(dot, 0.0);

        // Should have both X and Y components
        assert!(force.x.abs() > 1e-6);
        assert!(force.y.abs() > 1e-6);
    }
}

/// Computes ground "drag force" opposing motion while in contact with the ground. This is a drag-like force proportional to normal force and tangent relative velocity, as opposed to a simple Coulomb friction model without velocity dependence.
///
/// Intent is twofold:
/// -want the force opposing motion to go to zero as the relative velocity goes to zero, which prevents overshooting and oscillations around zero velocity.
/// - we're modeling a running biped, not a sliding block or a rolling wheel, so we can deviate from simple Coulomb friction models.
pub fn ground_drag_force_2d<T: AD>(
    contact: FrictionContact2D<T>,
    run_drag_coeff: T,
    sticky_glove_force: T,
) -> Vector2<T> {
    debug_assert!(run_drag_coeff >= T::constant(0.0));

    let n = contact.normal_force_mag();
    if n <= T::constant(0.0) {
        return Vector2::zeros();
    }

    -contact.tangent_relative_velocity()
        * contact.traction_coeff()
        * run_drag_coeff
        * (n + sticky_glove_force)
}

#[cfg(test)]
mod ground_drag_tests {
    use super::*;
    use test_case::test_case;

    #[test]
    fn test_ground_rolling_resistance_opposes_motion() {
        let tangent_vel = Vector2::new(5.0, 0.0);
        let normal = UnitVector2::new_normalize(Vector2::new(0.0, 1.0));
        let contact = FrictionContact2D::new(normal, tangent_vel, 100.0, 1.0);

        let run_drag_coeff = 0.1;
        // no glove force on flat ground
        let sticky_glove_force = 0.0;

        let f1 = ground_drag_force_2d(contact, run_drag_coeff, sticky_glove_force);
        assert!(f1.x < 0.0);
        assert_approx_eq!(f1.y as f64, 0.0);

        let tangent_vel = Vector2::new(-5.0, 0.0);
        let normal = UnitVector2::new_normalize(Vector2::new(0.0, 1.0));
        let contact = FrictionContact2D::new(normal, tangent_vel, 100.0, 1.0);
        // moving -x => rolling force +x
        let f2 = ground_drag_force_2d(contact, run_drag_coeff, sticky_glove_force);
        assert!(f2.x > 0.0);
        assert_approx_eq!(f2.y as f64, 0.0);
    }

    #[test]
    fn test_ground_drag_proportional_to_normal_force() {
        let normal = UnitVector2::new_normalize(Vector2::new(0.0, 1.0));
        let vel = Vector2::new(5.0, 0.0);
        let run_drag_coeff = 0.1;
        let sticky_glove_force = 0.0;
        let contact_100n = FrictionContact2D::new(normal, vel, 100.0, 1.0);
        let contact_200n = FrictionContact2D::new(normal, vel, 200.0, 1.0);

        let f1 = ground_drag_force_2d(contact_100n, run_drag_coeff, sticky_glove_force);
        let f2 = ground_drag_force_2d(contact_200n, run_drag_coeff, sticky_glove_force);

        // Doubling normal force should double drag
        assert_approx_eq!(f2.x as f64, 2.0 * f1.x as f64);
    }

    #[test_case((0.0,0.05); "light rolling")]
    #[test_case((0.001,0.1); "medium rolling")]
    #[test_case((0.2,1.0); "heavy rolling")]
    #[test_case((0.5,100.0); "absurd rolling")]
    fn test_ground_drag_scales_with_coefficient((a, b): (f32, f32)) {
        let normal = UnitVector2::new_normalize(Vector2::new(0.0, 1.0));
        let vel = Vector2::new(5.0, 0.0);
        let contact = FrictionContact2D::new(normal, vel, 100.0, 1.0);

        // no glove force on flat ground
        let sticky_glove_force = 0.0;
        let f_a = ground_drag_force_2d(contact, a, sticky_glove_force);
        let f_b = ground_drag_force_2d(contact, b, sticky_glove_force);
        assert!(f_b.x.abs() > f_a.x.abs());
    }
}

/// Ground-only net force (Newtons), excluding aerodynamic drag and excluding gravity.
/// Typically includes:
/// - traction-limited drive along tangent
/// - rolling resistance along tangent
pub fn ground_net_force_2d<T: AD>(
    input: Vector2<T>,
    contact: FrictionContact2D<T>,
    run_drag_coeff: T,
    ground_force_max: T,
    sticky_glove_force: T,
) -> Vector2<T> {
    let drive = ground_drive_force_2d(input, contact, ground_force_max, sticky_glove_force);
    let drag = ground_drag_force_2d(contact, run_drag_coeff, sticky_glove_force);
    drive + drag
}

// #[cfg(test)]
// mod ground_net_force_tests {
//     use super::*;

//     #[test]
//     fn test_ground_net_force_at_rest_with_no_input() {
//         let normal = UnitVector2::new_normalize(Vector2::new(0.0, 1.0));
//         let contact = GroundContact2D::new(normal, Vector2::zeros(), 100.0, 1.0);

//         let f = ground_net_force_2d(Vector2::zeros(), contact, 0.1, 1.0, 50.0);
//         assert_eq!(f, Vector2::zeros());
//     }
// }

/// Ground-only acceleration (m/s^2) = ground forces/m + gravity.
pub fn ground_accel_2d<T: AD>(
    input: Vector2<T>,
    contact: FrictionContact2D<T>,
    givens: &DynamicsGivenParams<T>,
    unknowns: &DynamicsDerivedParams<T>,
) -> Vector2<T> {
    let f_ground = ground_net_force_2d(
        input,
        contact,
        unknowns.run_drag_coeff,
        unknowns.run_force_max,
        unknowns.sticky_glove_force,
    );

    // Add gravity as acceleration
    f_ground / givens.mass + Vector2::new(T::constant(0.0), unknowns.g)
}
