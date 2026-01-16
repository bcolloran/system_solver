use crate::prelude::*;
use system_solver::prelude::{ad_trait::AD, nalgebra::Vector2};

/// Struct wraps the DynamicsState, as well as holding a couple other variables that the integrator should track but which do not need to be seen by the dynamics functions.
#[derive(Copy, Clone, Debug)]
pub struct IntegrationState<T>
where
    T: AD + Sized + Clone,
{
    pub t: T,
    pub pos: Vector2<T>,
    pub state: DynamicsState<T>,
}

impl<T> IntegrationState<T>
where
    T: AD + Sized + Clone,
{
    pub fn new_zeroed() -> Self {
        Self {
            t: T::constant(0.0),
            pos: Vector2::new(T::constant(0.0), T::constant(0.0)),
            state: DynamicsState::new_zeroed(),
        }
    }
}

// /// One semi-implicit Euler step of the 2D dynamics.
// ///
// /// (Not sure if this is the best integration method, but trying to keep things simple. I think this should be ok since the dynamics are fairly simple, and since I can use much smaller time steps for the optimization steps than in the real game sim.)
pub fn step_state<T: AD>(
    acc_fn: &dyn Fn(
        &DynamicsState<T>,
        &DynamicsGivenParams<T>,
        &DynamicsDerivedParams<T>,
    ) -> Vector2<T>,
    integration_state: &IntegrationState<T>,
    givens: &DynamicsGivenParams<T>,
    unk: &DynamicsDerivedParams<T>,
    dt: T,
) -> Option<IntegrationState<T>> {
    let s = &integration_state.state;
    let a = acc_fn(&s, givens, unk);

    let mut next_integration_state = integration_state.clone();

    if !a.x.is_finite() || !a.y.is_finite() {
        // println!("Non-finite acceleration: {:?}", a);
        // return None;
        next_integration_state.t += dt;
        return Some(next_integration_state);
    }

    let v_next = s.vel + a * dt;
    next_integration_state.pos = integration_state.pos + v_next * dt; // semi-implicit / symplectic Euler
    next_integration_state.t = integration_state.t + dt;

    // Numeric guard: abort this trajectory if it blows up
    if !next_integration_state.t.is_finite()
        || !next_integration_state.pos.x.is_finite()
        || !next_integration_state.pos.y.is_finite()
        || !v_next.x.is_finite()
        || !v_next.y.is_finite()
    {
        println!(
            "Non-finite next state found;  t_next={}, p_next={:?}, v_next={:?}",
            next_integration_state.t, next_integration_state.pos, v_next
        );
        // return None;
        return Some(next_integration_state);
    }

    next_integration_state.state.vel = v_next;

    Some(next_integration_state)
}

pub fn step_state_to_t_with_acc_fn<T: AD>(
    acc_fn: fn(&DynamicsState<T>, &DynamicsGivenParams<T>, &DynamicsDerivedParams<T>) -> Vector2<T>,
    integration_state: IntegrationState<T>,
    givens: &DynamicsGivenParams<T>,
    unk: &DynamicsDerivedParams<T>,
    dt: T,
    t_target: T,
) -> Option<IntegrationState<T>> {
    let mut s_curr = integration_state;
    while s_curr.t < t_target {
        s_curr = step_state(&acc_fn, &s_curr, givens, unk, dt)?;
    }
    Some(s_curr)
}

// note that while we set the ground contact to use the calculated unknonwns.g value, we need to set `g` to zero in the unknowns passed to the acc_fn because in the actually engine the normal force is applied to the body by the engine's collision handling, but within the simplified dynamics we only use the ground contact to compute friction and drive forces, not to apply gravity compensation.
pub fn step_state_to_t_on_flat_ground_with_acc_fn<T: AD>(
    acc_fn: fn(&DynamicsState<T>, &DynamicsGivenParams<T>, &DynamicsDerivedParams<T>) -> Vector2<T>,
    integration_state: IntegrationState<T>,
    givens: &DynamicsGivenParams<T>,
    unk: &DynamicsDerivedParams<T>,
    dt: T,
    t_target: T,
) -> Option<IntegrationState<T>> {
    let mut s_curr = integration_state;
    let contact_g = unk.g;
    let mut unk = unk.clone();
    unk.g = T::zero();
    while s_curr.t < t_target {
        let contact = FrictionContact2D::new_equilibrium_contact_from_angle(
            T::constant(0.0),
            s_curr.state.vel,
            contact_g,
            givens.mass,
        );
        s_curr.state.contact = Some(contact);
        s_curr = step_state(&acc_fn, &s_curr, givens, &unk, dt)?;
    }
    Some(s_curr)
}
