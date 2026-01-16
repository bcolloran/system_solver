use crate::{DynamicsDerivedParams, DynamicsGivenParams};
use system_solver::prelude::ad_trait::AD;

impl DynamicsGivenParams<f64> {
    pub fn to_ad<T: AD>(self) -> DynamicsGivenParams<T> {
        DynamicsGivenParams {
            mass: T::constant(self.mass),
            jump_time_up: T::constant(self.jump_time_up),
            jump_time_down: T::constant(self.jump_time_down),
            jump_height: T::constant(self.jump_height),
            max_vel_run: T::constant(self.max_vel_run),
            time_to_95pct_max_vel_run: T::constant(self.time_to_95pct_max_vel_run),
            x_stop_speed_threshold: T::constant(self.x_stop_speed_threshold),
            wall_slide_terminal_vel: T::constant(self.wall_slide_terminal_vel),
            sticky_glove_angle_deg: T::constant(self.sticky_glove_angle_deg),
            max_air_speed_x: T::constant(self.max_air_speed_x),
            time_to_95pct_max_air_speed_x: T::constant(self.time_to_95pct_max_air_speed_x),
        }
    }
}

impl<T: AD> DynamicsGivenParams<T> {
    pub fn to_f64(&self) -> DynamicsGivenParams<f64> {
        DynamicsGivenParams {
            mass: self.mass.into(),
            jump_time_up: self.jump_time_up.into(),
            jump_time_down: self.jump_time_down.into(),
            jump_height: self.jump_height.into(),
            max_vel_run: self.max_vel_run.into(),
            time_to_95pct_max_vel_run: self.time_to_95pct_max_vel_run.into(),
            x_stop_speed_threshold: self.x_stop_speed_threshold.into(),
            wall_slide_terminal_vel: self.wall_slide_terminal_vel.into(),
            sticky_glove_angle_deg: self.sticky_glove_angle_deg.into(),
            max_air_speed_x: self.max_air_speed_x.into(),
            time_to_95pct_max_air_speed_x: self.time_to_95pct_max_air_speed_x.into(),
        }
    }
}

// test round-trip conversion between f64 and AD types
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_dynamics_given_params_test_conversion() {
        let params_f64 = DynamicsGivenParams {
            mass: 70.0,
            jump_time_up: 0.5,
            jump_time_down: 0.5,
            jump_height: 2.0,
            max_vel_run: 5.0,
            time_to_95pct_max_vel_run: 1.0,
            x_stop_speed_threshold: 0.1,
            wall_slide_terminal_vel: -3.0,
            sticky_glove_angle_deg: 30.0,
            max_air_speed_x: 4.0,
            time_to_95pct_max_air_speed_x: 1.0,
        };
        let params_ad = params_f64.to_ad::<f32>();
        let params_f64_converted = params_ad.to_f64();
        assert_eq!(params_f64, params_f64_converted);
    }
}

impl DynamicsDerivedParams<f64> {
    pub fn new_nans() -> Self {
        Self {
            air_drag_coeff: f64::NAN,
            air_thrust_max: f64::NAN,
            g: f64::NAN,
            jump_vy_0: f64::NAN,
            jump_boost_force: f64::NAN,
            run_force_max: f64::NAN,
            run_drag_coeff: f64::NAN,
            sticky_glove_force: f64::NAN,
        }
    }
    pub fn to_ad<T: AD>(&self) -> DynamicsDerivedParams<T> {
        DynamicsDerivedParams {
            air_drag_coeff: T::constant(self.air_drag_coeff),
            air_thrust_max: T::constant(self.air_thrust_max),
            g: T::constant(self.g),
            jump_vy_0: T::constant(self.jump_vy_0),
            jump_boost_force: T::constant(self.jump_boost_force),
            run_force_max: T::constant(self.run_force_max),
            run_drag_coeff: T::constant(self.run_drag_coeff),
            sticky_glove_force: T::constant(self.sticky_glove_force),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ParamBounds {
    pub lb: f64,
    pub prior: f64,
    pub ub: f64,
}
impl ParamBounds {
    pub fn new(lb: f64, prior: f64, ub: f64) -> Self {
        Self { lb, prior, ub }
    }
}

pub type DynamicsDerivedParamsBounds = DynamicsDerivedParams<ParamBounds>;
