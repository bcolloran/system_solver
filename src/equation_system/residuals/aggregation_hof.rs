use std::rc::Rc;

use ad_trait::AD;

use crate::prelude::*;

/// Trait for specifying a higher-order-function that can generate *generic* residual aggregation functions for vecs of residuals of any type `T:AD`.
///
/// Typically this should probably be a summation or norm, but for optimizers like Gauss-Newton that expect a *vector* of residuals, it could be no aggregation at all, just passing residuals through.
///
/// We do this as a trait rather than a normal struct or function so that we can more easily specify the HOF in one place, and then pladd that around to the locations where `ad_trait` needs concrete `f64`` and `adfn<1>` versions.es, it's kind of a pain to need this much abstraction, but it's seems better than passing around multiple copies of functions specified for different types all over the place.
pub trait ResidAggHOF: Clone {
    fn make_residual_operator_fn<T: AD>(&self) -> Rc<dyn Fn(Vec<T>) -> Vec<T>>;
    fn scalar_cost_f64(&self, residuals: Vec<f64>) -> f64;
    fn num_outputs(&self) -> usize;
}

pub trait ResidAggFnToScalarGen: Clone {
    fn make_residuals_to_scalar_fn<T: AD>(&self) -> Rc<dyn Fn(Vec<T>) -> T>;
}

impl<R: ResidAggFnToScalarGen> ResidAggHOF for R {
    fn make_residual_operator_fn<T: AD>(&self) -> Rc<dyn Fn(Vec<T>) -> Vec<T>> {
        let to_scalar_fn = self.make_residuals_to_scalar_fn::<T>();
        Rc::new(move |residuals: Vec<T>| vec![to_scalar_fn(residuals)])
    }
    fn scalar_cost_f64(&self, residuals: Vec<f64>) -> f64 {
        let to_scalar_fn = self.make_residuals_to_scalar_fn::<f64>();
        to_scalar_fn(residuals)
    }
    fn num_outputs(&self) -> usize {
        1
    }
}

#[derive(Clone)]
pub struct ResidAggSum;
impl ResidAggFnToScalarGen for ResidAggSum {
    fn make_residuals_to_scalar_fn<T: AD>(&self) -> Rc<dyn Fn(Vec<T>) -> T> {
        Rc::new(|residuals: Vec<T>| {
            residuals
                .iter()
                .cloned()
                .fold(T::constant(0.0), |acc, x| acc + x)
        })
    }
}

#[derive(Clone)]
pub struct ResidNoOpGaussNewton {
    n: usize,
}
impl ResidNoOpGaussNewton {
    pub fn new_fullprob(n: usize) -> Self {
        Self { n }
    }
    pub fn new_subprob(block: &SolutionBlock) -> Self {
        Self {
            n: block.equation_idxs.len(),
        }
    }
}
impl ResidAggHOF for ResidNoOpGaussNewton {
    fn make_residual_operator_fn<T: AD>(&self) -> Rc<dyn Fn(Vec<T>) -> Vec<T>> {
        Rc::new(|residuals: Vec<T>| residuals)
    }
    fn num_outputs(&self) -> usize {
        self.n
    }
    fn scalar_cost_f64(&self, residuals: Vec<f64>) -> f64 {
        residuals.iter().fold(0.0, |acc, &x| acc + x)
    }
}
