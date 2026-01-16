use crate::{
    equation_system::{
        solution_plan::{SolutionBlock, SolutionPlan},
        sub_problem::SubProblem,
    },
    prelude::{solve_subproblem::simulated_annealing::SimulatedAnnealingConfig, *},
};
use ad_trait::{
    differentiable_function::ForwardAD, forward_ad::adfn::adfn, function_engine::FunctionEngine,
};

use nalgebra::{Dyn, Matrix, PermutationSequence, VecStorage};
use nalgebra_block_triangularization::{
    LowerBtfStructure, lower_block_triangular_structure, lower_triangular_permutations,
};
use struct_to_array::StructToVec;

pub mod objective;
pub mod opt_tools;
pub mod param_scaling;
pub mod param_traits;
pub mod residuals;
pub mod solution_plan;
pub mod sub_problem;

#[cfg(test)]
mod tests;

/// EquationSystemBuilder for solving systems of equations.
///
/// Type parameters:
/// - `G64`: Given params type for f64 (e.g., `DynamicsGivenParams<f64>`)
/// - `U64`: Unknown params type for f64 (e.g., `DynamicsDerivedParams<f64>`)
/// - `Gadfn`: Given params type for adfn<1> (e.g., `DynamicsGivenParams<adfn<1>>`)
/// - `Uadfn`: Unknown params type for adfn<1> (e.g., `DynamicsDerivedParams<adfn<1>>`)
/// - `S`: State type (e.g., `EqSysStateInit` or `EqSysSolutionPlan`)
/// - `N`: Number of unknown parameters
pub struct EquationSystemBuilder<G64, U64, Gadfn, Uadfn, S, const N: usize>
where
    G64: GivenParamsFor<f64, N>,
    U64: UnknownParamsFor<f64, N>,
    Gadfn: GivenParamsFor<adfn<1>, N>,
    Uadfn: UnknownParamsFor<adfn<1>, N>,
{
    givens_f64: G64,
    givens_adfn: Gadfn,
    /// The raw residual functions before:
    /// - filtering to sub-problems
    /// - parameter scaling
    /// - residual transforms
    raw_res_fns: ResidualFns<G64, U64, Gadfn, Uadfn>,
    /// The function engine to compute residuals and derivatives of the
    /// raw residual functions.
    raw_res_fn_engine: FunctionEngine<
        ObjectiveFunction<f64, G64, U64, ResidTransIdentity, ResidNoOpGaussNewton, N>,
        ObjectiveFunction<adfn<1>, Gadfn, Uadfn, ResidTransIdentity, ResidNoOpGaussNewton, N>,
        ForwardAD,
    >,
    /// Field names for the unknown parameters (for debugging/logging)
    unknown_field_names: &'static [&'static str],
    state: S,
}

pub struct EqSysStateInit;

impl<G64, U64, Gadfn, Uadfn, const N: usize> EquationSystemBuilder<G64, U64, Gadfn, Uadfn, (), N>
where
    G64: GivenParamsFor<f64, N> + Clone,
    U64: UnknownParamsFor<f64, N>,
    Gadfn: GivenParamsFor<adfn<1>, N> + Clone,
    Uadfn: UnknownParamsFor<adfn<1>, N>,
{
    pub fn new(
        givens_f64: G64,
        givens_adfn: Gadfn,
        raw_residual_fns: ResidualFns<G64, U64, Gadfn, Uadfn>,
        unknown_field_names: &'static [&'static str],
    ) -> Result<EquationSystemBuilder<G64, U64, Gadfn, Uadfn, EqSysStateInit, N>, EqSysError> {
        let num_eqs = raw_residual_fns.f64().len();
        let identity_loss_gen = ResidTransIdentity { n: num_eqs };
        let resid_pass_through = ResidNoOpGaussNewton::new_fullprob(num_eqs);

        let residuals_f64 = ObjectiveFunction::new(
            &givens_f64,
            &raw_residual_fns.f64(),
            identity_loss_gen.clone(),
            resid_pass_through.clone(),
            None,
        );
        let residuals_adfn = ObjectiveFunction::new(
            &givens_adfn,
            &raw_residual_fns.adfn_1(),
            identity_loss_gen,
            resid_pass_through.clone(),
            None,
        );

        let res_fn_engine = FunctionEngine::new(
            residuals_f64.clone(),
            residuals_adfn.clone(),
            ForwardAD::new(),
        );

        Ok(EquationSystemBuilder {
            givens_f64,
            givens_adfn,
            raw_res_fns: raw_residual_fns,
            raw_res_fn_engine: res_fn_engine,
            unknown_field_names,
            state: EqSysStateInit {},
        })
    }
}

impl<G64, U64, Gadfn, Uadfn, const N: usize>
    EquationSystemBuilder<G64, U64, Gadfn, Uadfn, EqSysStateInit, N>
where
    G64: GivenParamsFor<f64, N>,
    U64: UnknownParamsFor<f64, N>,
    Gadfn: GivenParamsFor<adfn<1>, N>,
    Uadfn: UnknownParamsFor<adfn<1>, N>,
{
    pub fn with_triangularization(
        self,
        inital_unknowns: &U64,
    ) -> Result<EquationSystemBuilder<G64, U64, Gadfn, Uadfn, EqSysSolutionPlan, N>, EqSysError>
    {
        let unknowns_vec = inital_unknowns.to_arr();
        let (_val_all, grad_all) = self.raw_res_fn_engine.derivative(&unknowns_vec);

        if grad_all.nrows() != grad_all.ncols() {
            return Err(EqSysError::NumEquationsNumUnknownsMismatch {
                n_eqs: grad_all.nrows(),
                n_unks: grad_all.ncols(),
            });
        }

        let binary_matrix = to_binary_matrix(grad_all);
        let structure = lower_block_triangular_structure(&binary_matrix);

        let (pr, pc) = lower_triangular_permutations(&binary_matrix);

        let mut u = binary_matrix.clone();
        pr.permute_rows(&mut u);
        pc.permute_columns(&mut u);

        let soln_blocks: Vec<SolutionBlock> = structure
            .block_indices()
            .iter()
            .enumerate()
            .map(|(block_num, (row_idxs, col_idxs))| SolutionBlock {
                block_idx: block_num,
                equation_idxs: row_idxs.clone(),
                unknown_idxs: col_idxs.clone(),
            })
            .collect();

        let solution_plan = SolutionPlan::new(soln_blocks);

        Ok(EquationSystemBuilder {
            givens_f64: self.givens_f64,
            givens_adfn: self.givens_adfn,
            raw_res_fns: self.raw_res_fns,
            raw_res_fn_engine: self.raw_res_fn_engine,
            unknown_field_names: self.unknown_field_names,
            state: EqSysSolutionPlan {
                binary_matrix,
                lower_tri_mat: u,
                block_structure: structure,
                row_permutation: pr,
                col_permutation: pc,
                solution_plan,
            },
        })
    }
}

fn to_binary_matrix(
    mat: Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>,
) -> Matrix<f32, Dyn, Dyn, VecStorage<f32, Dyn, Dyn>> {
    let (nrows, ncols) = mat.shape();
    let mut bin_mat = Matrix::<f32, Dyn, Dyn, VecStorage<f32, Dyn, Dyn>>::zeros(nrows, ncols);
    for r in 0..nrows {
        for c in 0..ncols {
            let x = mat[(r, c)];
            let out = if !x.is_finite() {
                f32::NAN
            } else if x != 0.0 {
                1.0
            } else {
                0.0
            };
            bin_mat[(r, c)] = out;
        }
    }
    bin_mat
}

pub struct EqSysSolutionPlan {
    binary_matrix: Matrix<f32, Dyn, Dyn, VecStorage<f32, Dyn, Dyn>>,
    lower_tri_mat: Matrix<f32, Dyn, Dyn, VecStorage<f32, Dyn, Dyn>>,
    block_structure: LowerBtfStructure,
    row_permutation: PermutationSequence<Dyn>,
    col_permutation: PermutationSequence<Dyn>,
    solution_plan: SolutionPlan,
}

impl<G64, U64, Gadfn, Uadfn, const N: usize>
    EquationSystemBuilder<G64, U64, Gadfn, Uadfn, EqSysSolutionPlan, N>
where
    G64: GivenParamsFor<f64, N>,
    U64: UnknownParamsFor<f64, N>,
    Gadfn: GivenParamsFor<adfn<1>, N>,
    Uadfn: UnknownParamsFor<adfn<1>, N>,
{
    pub fn block_structure(&self) -> &LowerBtfStructure {
        &self.state.block_structure
    }
    pub fn print_lower_tri_mat(&self) {
        println!(
            "Lower block triangular matrix:\n{}",
            self.state.lower_tri_mat
        );
    }
    pub fn print_block_structure(&self) {
        println!("Lower block triangular structure:");
        println!(
            "   matching_size = {}",
            self.state.block_structure.matching_size
        );
        println!(
            "   block_sizes   = {:?}",
            self.state.block_structure.block_sizes
        );
        println!("   row_order = {:?}", self.state.block_structure.row_order);
        println!("   col_order = {:?}", self.state.block_structure.col_order);
    }
    pub fn print_permuted_function_names(&self) {
        println!("Permuted residual function names:");
        for &r in &self.state.block_structure.row_order {
            let fn_name = self.raw_res_fns.fn_names()[r];
            println!("   {}", fn_name);
        }
    }
    pub fn print_permuted_unknowns_names(&self) {
        println!("Permuted unknowns names:");

        for &c in &self.state.block_structure.col_order {
            let unk_name = self.unknown_field_names[c];
            println!("   {}", unk_name);
        }
    }

    pub fn print_solution_plan(&self) {
        self.state
            .solution_plan
            .print_solution_plan(&self.raw_res_fns, self.unknown_field_names);
    }

    pub fn print_per_fn_residuals_at_params(&self, params: &U64) {
        let residuals = self.raw_res_fn_engine.call(&params.to_vec());

        println!("Per-function residuals at given params (plan order):");

        for block in self.state.solution_plan.blocks.iter() {
            println!(" Block {}:", block.block_idx);
            for &eq_idx in &block.equation_idxs {
                let fn_name = self.raw_res_fns.fn_names()[eq_idx];
                let res_val = residuals[eq_idx];
                println!("   {}: {:.6}", fn_name, res_val);
            }
        }
    }

    /// Solves a single sub-problem using L-BFGS optimization.
    pub fn solve_sub_problem_lbfgs(
        &self,
        block: &SolutionBlock,
        initial_unknowns: &U64,
    ) -> Result<U64, EqSysError> {
        let l2_loss_gen = ResidTransUnscaledL2 {
            n: self.raw_res_fns.f64().len(),
        };

        let subprob = SubProblem::new(
            &self.raw_res_fns,
            &block,
            &self.givens_f64,
            &self.givens_adfn,
            &initial_unknowns,
            l2_loss_gen,
            ResidAggSum {},
            true,
        );

        Ok(subprob.solve_lbfgs()?)
    }

    pub fn solve_sub_problem_simulated_annealing(
        &self,
        block: &SolutionBlock,
        initial_unknowns: &U64,
    ) -> Result<U64, EqSysError> {
        let l2_loss_gen = ResidTransUnscaledL2 {
            n: self.raw_res_fns.f64().len(),
        };

        let subprob = SubProblem::new(
            &self.raw_res_fns,
            &block,
            &self.givens_f64,
            &self.givens_adfn,
            &initial_unknowns,
            l2_loss_gen,
            ResidAggSum {},
            true,
        )
        .with_simulated_annealing_config(SimulatedAnnealingConfig::default());

        let best_params = subprob.solve_simulated_annealing()?;

        // self.print_per_fn_residuals_at_params(&best_params);

        Ok(best_params)
    }

    pub fn solve_sub_problem_gauss_newton(
        &self,
        block: &SolutionBlock,
        initial_unknowns: &U64,
    ) -> Result<U64, EqSysError> {
        let l2_loss_gen = ResidTransUnscaledL2 {
            n: self.raw_res_fns.f64().len(),
        };

        let subprob = SubProblem::new(
            &self.raw_res_fns,
            &block,
            &self.givens_f64,
            &self.givens_adfn,
            &initial_unknowns,
            l2_loss_gen,
            ResidNoOpGaussNewton::new_subprob(&block),
            true,
        );

        let best_params = subprob.solve_gauss_newton()?;

        Ok(best_params)
    }

    pub fn solve_system(&self, initial_unknowns: &U64) -> Result<U64, EqSysError> {
        let mut current_unknowns = initial_unknowns.clone();

        for (i, block) in self.state.solution_plan.blocks.iter().enumerate() {
            println!(
                "\n\n################## Solving sub-problem {} ##################",
                i
            );

            self.state.solution_plan.print_solution_block(
                block,
                &self.raw_res_fns,
                self.unknown_field_names,
            );

            let gn_soln = self.solve_sub_problem_gauss_newton(block, &current_unknowns);

            if let Ok(best_params) = gn_soln {
                current_unknowns = best_params;
                continue;
            } else if let Err(e) = &gn_soln {
                println!(
                    ">>>>> Gauss-Newton failed for sub-problem {}: {:?}. Trying Simulated Annealing",
                    i, e
                );
            }

            let sa_soln = self.solve_sub_problem_simulated_annealing(block, &current_unknowns);

            let sa_soln = match sa_soln {
                Ok(best_params) => best_params,
                Err(e) => {
                    println!(
                        "    >>>>> Simulated Annealing also failed for sub-problem {}: {:?}",
                        i, e
                    );
                    return Err(e);
                }
            };

            // If we got an SA solution, refine it with Gauss-Newton
            let refined_gn_soln = self.solve_sub_problem_gauss_newton(block, &sa_soln);

            current_unknowns = match refined_gn_soln {
                Ok(best_params) => best_params,
                Err(e) => {
                    panic!(
                        "\n    >>>>> Gauss-Newton refinement after SA also failed for sub-problem {}: {:?}.",
                        i, e
                    );
                    // sa_soln
                }
            };

            self.print_per_fn_residuals_at_params(&current_unknowns);
        }

        // Do a final fine-tuning pass over the full problem
        println!("\n\n################## full-problem refinement ##################");

        let full_prob_block = SolutionBlock::new_fullprob(self.raw_res_fns.f64().len());

        current_unknowns = self.solve_sub_problem_lbfgs(&full_prob_block, &current_unknowns)?;

        self.print_per_fn_residuals_at_params(&current_unknowns);

        Ok(current_unknowns)
    }
}
