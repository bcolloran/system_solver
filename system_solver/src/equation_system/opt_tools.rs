use std::{cell::RefCell, rc::Rc};

use argmin::core::{Error, IterState, KV, OptimizationResult, State, observers::Observe};

use crate::prelude::SubProblem;

pub type OptRes<S, R, A, G, J = ()> =
    OptimizationResult<SubProblem<R, A>, S, IterState<nalgebra::DVector<f64>, G, J, (), (), f64>>;

#[derive(Clone)]
pub struct MyObserver {
    cost_history: Rc<RefCell<Vec<f64>>>,
}

impl MyObserver {
    pub fn new() -> Self {
        Self {
            cost_history: Rc::new(RefCell::new(Vec::new())),
        }
    }

    pub fn cost_history(&self) -> Vec<f64> {
        self.cost_history.borrow().clone()
    }

    pub fn observe_cost(&self, cost: f64) {
        self.cost_history.borrow_mut().push(cost);
    }
}

impl<I> Observe<I> for MyObserver
where
    // Optional constraint on `I`. The `State` trait, which every state used in argmin needs to
    // implement, offers a range of methods which can be useful.
    I: State,
    I: State<Float = f64>,
{
    fn observe_init(&mut self, _name: &str, _state: &I, _kv: &KV) -> Result<(), Error> {
        Ok(())
    }

    fn observe_iter(&mut self, state: &I, _kv: &KV) -> Result<(), Error> {
        self.observe_cost(state.get_cost());

        Ok(())
    }
}
