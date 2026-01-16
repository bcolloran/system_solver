use std::rc::Rc;

use ad_trait::{AD, differentiable_function::DifferentiableFunctionTrait};

use crate::prelude::*;

pub fn l2_loss_fns<T: AD>(n: usize) -> Vec<Rc<dyn Fn(T) -> T>> {
    let f: Rc<dyn Fn(T) -> T> = Rc::new(|r: T| r * r);
    (0..n).map(|_| f.clone()).collect()
}

pub fn sum_loss<T: AD>(residuals: Vec<T>) -> T {
    residuals
        .iter()
        .cloned()
        .fold(T::constant(0.0), |acc, x| acc + x)
}

/// Forward and inverse parameter scaling functions between (constrained)model space and optimization (unconstrained) parameter space.
#[derive(Clone)]
pub struct ParamScaler<T: AD, const N: usize> {
    model_to_opt: Rc<dyn Fn([T; N]) -> [T; N]>,
    opt_to_model: Rc<dyn Fn([T; N]) -> [T; N]>,
}

impl<T: AD, const N: usize> ParamScaler<T, N> {
    /// Creates a ParamScaler from priors. The priors provide the f64 values which are
    /// converted to the target AD type T for use in the link functions.
    pub fn new_link_fns_from_priors<U>(priors: &U) -> Self
    where
        U: UnknownParamsFor<f64, N>,
    {
        let priors_f64 = priors.to_arr();
        let (opt_to_model, model_to_opt) = default_link_fns_builder::<T, N>(
            priors_f64
                .iter()
                .map(|&x| T::constant(x))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        );
        Self {
            model_to_opt: Rc::new(model_to_opt),
            opt_to_model: Rc::new(opt_to_model),
        }
    }
    pub fn model_to_opt(&self, model_params: [T; N]) -> [T; N] {
        (self.model_to_opt)(model_params)
    }
    pub fn opt_to_model(&self, opt_params: [T; N]) -> [T; N] {
        (self.opt_to_model)(opt_params)
    }
}

/// Container for objective function computing residuals from given residual functions. May or may not include residual transforms and residuals-to-loss functions.
#[derive(Clone)]
pub struct ObjectiveFunction<T: AD, G, U, R: ResidTransHOF, A: ResidAggHOF, const N: usize> {
    givens: G,
    fns: Vec<Rc<fn(&G, &U) -> T>>,

    /// Optional vector of functions to transform each residual before computing loss. This is applied element-wise to the residuals vector, and is where weighting, scaling, loss transforms (L1, L2, etc) can be applied.
    residual_transforms_gen: R,

    /// Optional function to convert residuals vector to a single loss value. Typically this should probably be a summation or norm?
    residual_agg_gen: A,
    param_scaling: Option<ParamScaler<T, N>>,
}

impl<T, G, U, R, A, const N: usize> ObjectiveFunction<T, G, U, R, A, N>
where
    T: AD,
    G: GivenParamsFor<T, N>,
    U: UnknownParamsFor<T, N>,
    R: ResidTransHOF,
    A: ResidAggHOF,
{
    pub fn new(
        givens: &G,
        fns: &Vec<Rc<fn(&G, &U) -> T>>,
        residual_transforms_gen: R,
        residual_agg_gen: A,
        param_scaling: Option<ParamScaler<T, N>>,
    ) -> Self
    where
        G: Clone,
    {
        Self {
            givens: givens.clone(),
            fns: fns.clone(),
            residual_transforms_gen,
            residual_agg_gen,
            param_scaling,
        }
    }
}

impl<T, G, U, R, A, const N: usize> DifferentiableFunctionTrait<T>
    for ObjectiveFunction<T, G, U, R, A, N>
where
    T: AD,
    G: GivenParamsFor<T, N>,
    U: UnknownParamsFor<T, N>,
    R: ResidTransHOF,
    A: ResidAggHOF,
{
    const NAME: &'static str = "ResidualsFunctions";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        let inputs: [T; N] = inputs.try_into().unwrap_or_else(|_| {
            panic!(
                "`inputs` length mismatch in `ResidualsFunctions::call`: expected {}, got {}",
                N,
                inputs.len()
            )
        });

        // Convert opt space inputs back to model space if scaling is used
        let p_model = self
            .param_scaling
            .as_ref()
            .map_or(inputs, |scaling| scaling.opt_to_model(inputs));

        // generate unknowns from model-space vector
        let unknowns = U::from_arr(p_model);

        let residuals = self.fns.iter().map(|f| f(&self.givens, &unknowns));

        let residual_transforms = self.residual_transforms_gen.make_loss_fns::<T>();

        let scaled_residuals: Vec<T> = residual_transforms
            .iter()
            .zip(residuals)
            .map(|(transform_fn, r)| transform_fn(r).clone())
            .collect();

        self.residual_agg_gen.make_residual_operator_fn::<T>()(scaled_residuals)

        // if let Some(loss_total_fn) = self.residual_agg_gen {
        //     vec![loss_total_fn(scaled_residuals)]
        // } else {
        //     scaled_residuals.to_vec()
        // }
    }

    fn num_inputs(&self) -> usize {
        N
    }

    fn num_outputs(&self) -> usize {
        self.residual_agg_gen.num_outputs()
    }
}
