use std::rc::Rc;

use ad_trait::AD;

/// Trait for specifying a higher-order-function that can generate *generic* vectors of residual transformation functions for residuals of any type `T:AD`.
///
/// These functions are applied element-wise to the residuals vector, and is where weighting, scaling, loss transforms (L1, L2, etc) can be applied.
///
/// We do this as a trait rather than a normal struct or function so that we can more easily specify the HOF in one place, and then pladd that around to the locations where `ad_trait` needs concrete `f64`` and `adfn<1>` versions.es, it's kind of a pain to need this much abstraction, but it's seems better than passing around multiple copies of functions specified for different types all over the place.
pub trait ResidTransHOF: Clone {
    fn make_loss_fns<T: AD>(&self) -> Vec<Rc<dyn Fn(T) -> T>>;
}

#[derive(Clone)]
pub struct ResidTransIdentity {
    pub n: usize,
}
impl ResidTransIdentity {
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl ResidTransHOF for ResidTransIdentity {
    fn make_loss_fns<T: AD>(&self) -> Vec<Rc<dyn Fn(T) -> T>> {
        let f: Rc<dyn Fn(T) -> T> = Rc::new(|r: T| r);
        (0..self.n).map(|_| f.clone()).collect()
    }
}

/// Unscaled L2 loss functions (r^2) for each residual.
///
#[derive(Clone)]
pub struct ResidTransUnscaledL2 {
    pub n: usize,
}
impl ResidTransHOF for ResidTransUnscaledL2 {
    fn make_loss_fns<T: AD>(&self) -> Vec<Rc<dyn Fn(T) -> T>> {
        let f: Rc<dyn Fn(T) -> T> = Rc::new(|r: T| r * r);
        (0..self.n).map(|_| f.clone()).collect()
    }
}

#[derive(Clone)]
pub struct ResidTransScaledL2 {
    scales: Vec<f64>,
}
impl ResidTransHOF for ResidTransScaledL2 {
    fn make_loss_fns<T: AD>(&self) -> Vec<Rc<dyn Fn(T) -> T>> {
        self.scales
            .iter()
            .map(|&s| {
                let f: Rc<dyn Fn(T) -> T> = Rc::new(move |r: T| r * r / T::constant(s));
                f
            })
            .collect()
    }
}
