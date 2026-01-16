use crate::prelude::*;

#[derive(Debug)]
/// A solution plan for an equation system.
pub struct SolutionPlan {
    pub blocks: Vec<SolutionBlock>,
}

impl SolutionPlan {
    /// Creates a new SolutionPlan with the given blocks.
    pub fn new(blocks: Vec<SolutionBlock>) -> Self {
        Self { blocks }
    }

    pub fn print_solution_plan<G64, U64, Gadfn, Uadfn>(&self, res_fns: &ResidualFns<G64, U64, Gadfn, Uadfn>, field_names: &[&str]) {
        for block in self.blocks.iter() {
            println!("Solution Block {}:", block.block_idx);
            self.print_solution_block(block, res_fns, field_names);
        }
    }

    pub fn print_solution_block<G64, U64, Gadfn, Uadfn>(
        &self,
        block: &SolutionBlock,
        res_fns: &ResidualFns<G64, U64, Gadfn, Uadfn>,
        field_names: &[&str],
    ) {
        println!("  equations:");
        // print the name of each equation
        for e in &block.equation_idxs {
            let fn_name = res_fns.fn_names[*e];
            println!("    {e}: {}", fn_name);
        }
        println!("  unknowns:");

        for u in &block.unknown_idxs {
            let unk_name = field_names[*u];
            println!("    {u}: {}", unk_name);
        }
    }
}

/// A block in the solution plan, representing a subset of equations and unknowns. The indices refer to the positions in the original, unpermuted system.
#[derive(Debug, Clone)]
pub struct SolutionBlock {
    pub block_idx: usize,
    pub equation_idxs: Vec<usize>,
    pub unknown_idxs: Vec<usize>,
}

impl SolutionBlock {
    /// Creates a new SolutionBlock.
    pub fn new_fullprob(size: usize) -> Self {
        Self {
            block_idx: 0,
            equation_idxs: (0..size).collect(),
            unknown_idxs: (0..size).collect(),
        }
    }
}
