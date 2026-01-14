use std::rc::Rc;

use ad_trait::{AD, forward_ad::adfn::adfn};

use crate::prelude::*;

#[derive(Clone)]
pub struct ResidualsFn<T: AD> {
    pub residual_scale: T,
    pub res_fn: Rc<fn(&DynamicsGivenParams<T>, &DynamicsDerivedParams<T>) -> T>,
}

pub struct ResidualFns2 {
    pub f64: Vec<ResidualsFn<f64>>,
    pub adfn_1: Vec<ResidualsFn<adfn<1>>>,
    pub fn_names: Vec<&'static str>,
}

/// Create ResidualFns struct with given residual function names.
#[macro_export]
macro_rules! residual_fns_2 {
    ($($fn_name:expr),* $(,)?) => {
        ResidualFns {
            f64: vec![
                $(Rc::new($fn_name)),*
            ],
            adfn_1: vec![
                $(Rc::new($fn_name)),*
            ],
            fn_names: vec![
                $(stringify!($fn_name)),*
            ],
        }
    };
}

/// Create ResidualFns struct with given residual function names.
#[macro_export]
macro_rules! residual_fns {
    ($($fn_name:expr),* $(,)?) => {
        ResidualFns {
            f64: vec![
                $(Rc::new($fn_name)),*
            ],
            adfn_1: vec![
                $(Rc::new($fn_name)),*
            ],
            fn_names: vec![
                $(stringify!($fn_name)),*
            ],
        }
    };
}

/// Container for residual functions in both f64 and adfn<1> forms.
#[derive(Clone)]
pub struct ResidualFns {
    pub f64: Vec<Rc<fn(&DynamicsGivenParams<f64>, &DynamicsDerivedParams<f64>) -> f64>>,
    pub adfn_1:
        Vec<Rc<fn(&DynamicsGivenParams<adfn<1>>, &DynamicsDerivedParams<adfn<1>>) -> adfn<1>>>,
    pub fn_names: Vec<&'static str>,
}

fn filter_res_fns_to_block<T>(
    fns: Vec<Rc<fn(&DynamicsGivenParams<T>, &DynamicsDerivedParams<T>) -> T>>,
    solution_block: &SolutionBlock,
) -> Vec<Rc<fn(&DynamicsGivenParams<T>, &DynamicsDerivedParams<T>) -> T>> {
    fns.iter()
        .enumerate()
        .filter_map(|(i, f)| {
            if solution_block.equation_idxs.contains(&i) {
                Some(f.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
}

impl ResidualFns {
    /// Filters the residual functions to only those in the given solution block.
    pub fn filter_res_fns_to_block(&self, solution_block: &SolutionBlock) -> ResidualFns {
        let res_fns_64 = filter_res_fns_to_block(self.f64.clone(), solution_block);
        let res_fns_adfn1 = filter_res_fns_to_block(self.adfn_1.clone(), solution_block);

        let fn_names = solution_block
            .equation_idxs
            .iter()
            .map(|&i| self.fn_names[i])
            .collect::<Vec<_>>();

        ResidualFns {
            f64: res_fns_64,
            adfn_1: res_fns_adfn1,
            fn_names,
        }
    }
}
