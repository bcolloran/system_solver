use system_solver::equation_system::param_traits::{GivenParams, UnknownParams};
use system_solver::prelude::*;

use field_names_and_counts::FieldNames;
use struct_to_array::StructToArray;

// use as_gd_res::AsGdRes; // Only needed for Godot integration
// use crate::DynamicsDerivedParams;
// use nalgebra::{Complex, ComplexField as CF};

/// These are the parameters that are chosen manually by game designers.
/// These parameters are meant to be simple to understand and tune, and should
/// use physical units that are intuitive-- how long a jump lasts, how high
/// it goes, how fast something travels.
#[derive(Debug, Clone, Copy, PartialEq, StructToArray)]
pub struct DynamicsGivenParams<T> {
    pub mass: T,

    pub jump_time_up: T,
    pub jump_time_down: T,
    pub jump_height: T,

    /// asymptotic max air speed under full input when no gravity is present
    pub max_air_speed_x: T,
    /// time to reach 95% of max air speed under full input when no gravity is present
    pub time_to_95pct_max_air_speed_x: T,

    pub max_vel_run: T,
    pub time_to_95pct_max_vel_run: T,
    // speed below which we zero out horizontal velocity to prevent jitter.
    pub x_stop_speed_threshold: T,

    pub wall_slide_terminal_vel: T,

    /// angle (degrees) of ground tangent at which sticky glove kicks in
    pub sticky_glove_angle_deg: T,
}

/// These paramaters are the "unknowns" that will never be touched directly by
/// game designers, but will instead be solved for by the parameter solver.
/// They are more low-level, and may not have intuitive physical meanings, or
/// might not be critical in and of themselves to tune directly, but need to
/// be set correctly to get the desired dynamics to work out.

#[derive(Copy, Clone, Debug, StructToArray, FieldNames)]
pub struct DynamicsDerivedParams<T> {
    pub air_drag_coeff: T,
    pub air_thrust_max: T,

    pub g: T,

    pub jump_vy_0: T,
    pub jump_boost_force: T,

    /// Max ground force magnitude (N) before traction limiting.
    pub run_force_max: T,
    /// ground "drag" coefficient (dimensionless) for canonical ref case.
    pub run_drag_coeff: T,

    /// additional normal force applied when sticky glove is active
    pub sticky_glove_force: T,
}

pub const N_UNKNOWNS: usize =
    core::mem::size_of::<DynamicsDerivedParams<f32>>() / core::mem::size_of::<f32>();

pub const N_GIVENS: usize =
    core::mem::size_of::<DynamicsGivenParams<f32>>() / core::mem::size_of::<f32>();

// Implement system_solver traits
impl<T> GivenParams for DynamicsGivenParams<T> where T: Clone + Copy + std::fmt::Debug {}
impl<T> UnknownParams for DynamicsDerivedParams<T> where T: Clone + Copy + std::fmt::Debug {}

// Note: to_ad() conversion methods are defined in dynamics/test_params.rs
