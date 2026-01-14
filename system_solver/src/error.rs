use thiserror::Error;

#[derive(Error, Debug)]
pub enum EqSysError {
    #[error("Number of equations!=unknowns; {n_eqs} equations, {n_unks} unknowns")]
    NumEquationsNumUnknownsMismatch { n_eqs: usize, n_unks: usize },

    #[error("Argmin error: {0}")]
    ArgminError(#[from] argmin::core::Error),

    #[error("No best individual found in optimization result")]
    NoBestPsoIndividual,
}

#[derive(Error, Debug)]
pub enum SolverError {
    #[error("Equation system error: {0}")]
    EqSysError(#[from] EqSysError),
}
