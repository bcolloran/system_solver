pub mod aerial;
pub mod jump;
pub mod run;

pub(super) mod integrate;

use ad_trait::AD;
use nalgebra::Vector2;

pub fn input_max_x_positive<T: AD>() -> Vector2<T> {
    Vector2::new(T::one(), T::zero())
}
