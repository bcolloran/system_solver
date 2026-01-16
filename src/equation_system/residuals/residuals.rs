use std::rc::Rc;

use ad_trait::{AD, forward_ad::adfn::adfn};

use crate::prelude::*;

#[derive(Clone)]
pub struct ResidualsFn<T: AD, G, U> {
    pub residual_scale: T,
    pub res_fn: Rc<fn(&G, &U) -> T>,
}

pub struct ResidualFns2<G, U> {
    pub f64: Vec<ResidualsFn<f64, G, U>>,
    pub adfn_1: Vec<ResidualsFn<adfn<1>, G, U>>,
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
/// Separate type parameters allow the givens/unknowns types to be parameterized by the AD type.
#[derive(Clone)]
pub struct ResidualFns<G64, U64, Gadfn, Uadfn> {
    pub f64: Vec<Rc<fn(&G64, &U64) -> f64>>,
    pub adfn_1: Vec<Rc<fn(&Gadfn, &Uadfn) -> adfn<1>>>,
    pub fn_names: Vec<&'static str>,
}

/// Create ResidualFns for types that are generic over T: AD.
/// Usage: `residual_fns_for_generic_params!(GivenType, UnknownType; fn1, fn2, ...)`
/// where GivenType<T> and UnknownType<T> are the parameter types.
#[macro_export]
macro_rules! residual_fns_for_generic_params {
    ($g:ident, $u:ident; $($fn_name:ident),* $(,)?) => {
        $crate::equation_system::residuals::residuals::ResidualFns::<
            $g<f64>, $u<f64>,
            $g<ad_trait::forward_ad::adfn::adfn<1>>, $u<ad_trait::forward_ad::adfn::adfn<1>>
        > {
            f64: vec![$(std::rc::Rc::new($fn_name::<f64>)),*],
            adfn_1: vec![$(std::rc::Rc::new($fn_name::<ad_trait::forward_ad::adfn::adfn<1>>)),*],
            fn_names: vec![$(stringify!($fn_name)),*],
        }
    };
}

fn filter_res_fns_to_block<T, G, U>(
    fns: Vec<Rc<fn(&G, &U) -> T>>,
    solution_block: &SolutionBlock,
) -> Vec<Rc<fn(&G, &U) -> T>> {
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

impl<G64, U64, Gadfn, Uadfn> ResidualFns<G64, U64, Gadfn, Uadfn> {
    /// Filters the residual functions to only those in the given solution block.
    pub fn filter_res_fns_to_block(
        &self,
        solution_block: &SolutionBlock,
    ) -> ResidualFns<G64, U64, Gadfn, Uadfn> {
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
