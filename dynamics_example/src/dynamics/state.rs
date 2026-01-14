use crate::prelude::*;

use ad_trait::AD;
use nalgebra::{ComplexField, UnitVector2, Vector2};

fn normal_from_tan_angle<T: AD>(tangent_angle: T) -> UnitVector2<T> {
    UnitVector2::new_normalize(Vector2::new(
        -ComplexField::sin(tangent_angle),
        ComplexField::cos(tangent_angle),
    ))
}

/// A local coordinate frame at a ground contact.
///
/// This struct represents the physical state of a body in contact with a surface,
/// providing the geometric information needed to compute ground forces.
///
/// # Coordinate System
/// - The `normal` points outward from the surface (away from the ground)
/// - The tangent is perpendicular to the normal, computed as `(normal.y, -normal.x)`
///
/// # Physical Meaning
/// - `normal_force_mag`: The magnitude of the contact force perpendicular to the surface (N)
///   This represents how "hard" the object is pressing against the ground.
/// - `tangent_relative_velocity`: The component of velocity parallel to the surface, used for rolling resistance and slip calculations. This relative velocity is is stated as if the object is moving relative to the ground surface; i.e., within the reference frame of the ground. The ground fram may itself be moving so this velocity is not necessarily the same as world-space velocity, but only the relative velocity between the body and the ground is needed for friction calculations.
#[derive(Debug, Clone, Copy)]
pub struct FrictionContact2D<T>
where
    T: AD + Sized + Clone,
{
    /// normal: outward normal from the body to the other surface
    normal: UnitVector2<T>,
    /// magnitude of normal force (Newtons), always >= 0
    normal_force_mag: T,
    /// *relative* velocity along the ground tangent. At initialization, this is projected to be exactly perpendicular to the normal.
    tangent_relative_velocity: Vector2<T>,
    /// the proportion of the normal force that can be used for ground traction
    traction_coeff: T,
}

impl FrictionContact2D<f64> {
    pub fn to_ad<T: AD>(self) -> FrictionContact2D<T> {
        FrictionContact2D {
            normal: UnitVector2::new_normalize(Vector2::new(
                T::constant(self.normal.x),
                T::constant(self.normal.y),
            )),
            normal_force_mag: T::constant(self.normal_force_mag),
            tangent_relative_velocity: Vector2::new(
                T::constant(self.tangent_relative_velocity.x),
                T::constant(self.tangent_relative_velocity.y),
            ),
            traction_coeff: T::constant(self.traction_coeff),
        }
    }
}

impl<T> FrictionContact2D<T>
where
    T: AD + Sized + Clone,
{
    /// Returns the relative velocity component along the ground tangent as a vector. This is given from the perspective of the ground surface's reference frame.
    ///
    pub fn tangent_relative_velocity(&self) -> Vector2<T> {
        self.tangent_relative_velocity
    }

    pub fn normal_force_mag(&self) -> T {
        self.normal_force_mag
    }

    pub fn traction_coeff(&self) -> T {
        self.traction_coeff
    }

    /// Returns the unit tangent vector perpendicular to the surface normal.
    ///
    /// The tangent is computed as `(normal.y, -normal.x)`, which gives a
    /// right-handed coordinate system where the tangent points "forward"
    /// along the surface when the normal points "up".
    pub fn tangent(&self) -> UnitVector2<T> {
        let n = self.normal;
        UnitVector2::new_normalize(Vector2::new(n.y, -n.x))
    }

    /// Absolute tangent angle above horizontal (degrees). 0 = flat ground, +90 = vertical wall.
    pub fn abs_tangent_angle_degrees(&self) -> T {
        let t = self.tangent();
        const DEG_PER_RAD: f64 = 180.0 / std::f64::consts::PI;
        (nalgebra::RealField::atan2(t.y, t.x) * T::constant(DEG_PER_RAD)).abs()
    }

    /// Creates a new ground contact from geometric and physical parameters.
    ///
    /// # Arguments
    /// * `normal` - Outward unit normal from the surface
    /// * `approx_tangent_vel` - Approximate tangent velocity (will be projected onto tangent)
    /// * `normal_force_mag` - Magnitude of normal force in Newtons (will be clamped >= 0)
    ///
    /// # Notes
    /// The `approx_tangent_vel` is projected onto the tangent direction to ensure
    /// it's exactly perpendicular to the normal, removing any numerical errors.
    pub(crate) fn new(
        normal: UnitVector2<T>,
        approx_tangent_vel: Vector2<T>,
        normal_force_mag: T,
        traction_coeff: T,
    ) -> Self {
        #[cfg(debug_assertions)]
        {
            if normal_force_mag < T::constant(0.0) {
                println!(
                    "Warning: GroundContact2D created with negative normal_force_mag: {}",
                    normal_force_mag
                );
            }
        }
        let normal_force_mag = normal_force_mag.max(T::constant(0.0));

        // Ensure tangent is orthogonal to normal by projecting onto tangent vector.
        let t = Vector2::new(normal.y, -normal.x);
        let v_t_mag = approx_tangent_vel.dot(&t);
        let tangent_velocity = t * v_t_mag;

        Self {
            normal,
            normal_force_mag,
            tangent_relative_velocity: tangent_velocity,
            traction_coeff: traction_coeff.max(T::constant(0.0)).min(T::constant(1.0)),
        }
    }

    /// Create a GroundContact2d given a normal vector and approx_tangent_vel and calculating equilibrium normal force required to move along the ground tangent from the mass and gravity.
    ///
    /// Using equilibrium contacts helps prevent friction spikes that occur when calculating normal force from impulses calculated by the physics engine at the initial moment of a contact. In this instant of contact, the
    /// normal force calculated from impulses can be very high due to collision resolution, leading to unrealistic friction forces (that can cause e.g. the player's forward speed to stall for a moment when landing from a jump). By estimating the normal force from gravity, we can prevent this kind of issue.
    /// FIXME: there may be a better way to do this:
    /// - some kind of smoothing/filtering of normal force over time?
    /// - just use equilibrium normal force for the first N ticks after contact?
    pub fn new_equilibrium_contact(
        normal: UnitVector2<T>,
        approx_tangent_vel: Vector2<T>,
        traction_coeff: T,
        gravity_acc_y: T,
        mass: T,
    ) -> Self {
        debug_assert!(
            gravity_acc_y < T::constant(0.0),
            "gravity_acc_y should be negative"
        );
        let f = estimate_normal_force_from_gravity(mass, gravity_acc_y, normal);
        Self::new(normal, approx_tangent_vel, f, traction_coeff)
    }

    /// Create a GroundContact2d based on an angle (radians) from the +X axis represinting the slope of the ground, and calculating equilibrium normal force required to move along the ground tangent from the mass and gravity.
    ///
    /// This is used for optimization purposes only, wherein we can assume that the ground frame is stationary and normal force arises solely from gravity.
    /// In this helper, we give the velocity of the body in world space; since the ground is assumed stationary, this is identical to the relative velocity along the tangent.
    pub fn new_equilibrium_contact_from_angle(
        tangent_angle: T,
        world_vel: Vector2<T>,
        gravity_acc_y: T,
        mass: T,
    ) -> Self {
        debug_assert!(
            gravity_acc_y < T::constant(0.0),
            "gravity_acc_y should be negative"
        );
        // in this scenario, the world velocity must be orthogonal to the normal
        let n = normal_from_tan_angle(tangent_angle);
        debug_assert!(
            world_vel.dot(&n.into_inner()).abs() < T::constant(1.0e-4),
            "world_vel must be orthogonal to normal for ground contact; dot = {}; normal={:#?}; world_vel={}",
            world_vel.dot(&n.into_inner()),
            n,
            world_vel
        );
        let f = estimate_normal_force_from_gravity(mass, gravity_acc_y, n);
        Self::new(n, world_vel, f, T::constant(1.0))
    }

    pub fn to_f64(&self) -> FrictionContact2D<f64> {
        FrictionContact2D {
            normal: UnitVector2::new_normalize(Vector2::new(
                self.normal.x.into(),
                self.normal.y.into(),
            )),
            normal_force_mag: self.normal_force_mag.into(),
            tangent_relative_velocity: Vector2::new(
                self.tangent_relative_velocity.x.into(),
                self.tangent_relative_velocity.y.into(),
            ),
            traction_coeff: self.traction_coeff.into(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DynamicsState<T>
where
    T: AD + Sized + Clone,
{
    pub vel: Vector2<T>,
    pub input: Vector2<T>,
    pub contact: Option<FrictionContact2D<T>>,
    pub jump_boost_active: bool,
}

impl<T> DynamicsState<T>
where
    T: AD + Sized + Clone,
{
    pub fn new_zeroed() -> Self {
        Self {
            vel: Vector2::new(T::constant(0.0), T::constant(0.0)),
            input: Vector2::new(T::constant(0.0), T::constant(0.0)),
            contact: None,
            jump_boost_active: false,
        }
    }
}

impl DynamicsState<f64> {
    pub fn to_ad<T: AD>(self) -> DynamicsState<T> {
        DynamicsState {
            vel: Vector2::new(T::constant(self.vel.x), T::constant(self.vel.y)),
            input: Vector2::new(T::constant(self.input.x), T::constant(self.input.y)),
            contact: self.contact.map(|c| c.to_ad::<T>()),
            jump_boost_active: self.jump_boost_active,
        }
    }
}
