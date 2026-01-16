pub mod constraints;
pub mod dynamics;
pub mod params;

pub mod prelude {
    pub use crate::{
        assert_approx_eq,
        dynamics::{
            air::air_accel_2d,
            ground::estimate_normal_force_from_gravity,
            state::{DynamicsState, FrictionContact2D},
            total_accel_2d, total_force_2d,
        },
        params::{DynamicsDerivedParams, DynamicsGivenParams, N_UNKNOWNS},
    };
}

pub use crate::prelude::*;

#[macro_export]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {{
        let eps = 1.0e-6;
        let (a, b) = (&$a, &$b);
        assert!(
            (*a - *b).abs() < eps,
            "assertion failed: `(left !== right)` \
             (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
            *a,
            *b,
            eps,
            (*a - *b).abs()
        );
    }};
    ($a:expr, $b:expr, $eps:expr) => {{
        let (a, b) = (&$a, &$b);
        let eps = $eps;
        assert!(
            (*a - *b).abs() < eps,
            "assertion failed: `(left !== right)` \
             (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
            *a,
            *b,
            eps,
            (*a - *b).abs()
        );
    }};
}
