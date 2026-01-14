// pub fn pso_solve(
//     &self,
//     subprob_idx: usize,
//     initial_unknowns: &DynamicsDerivedParams<f64>,
// ) -> Result<DynamicsDerivedParams<f64>, EqSysError> {
//     let pso_subprob = SubProblem::new(
//         &self.residual_fns,
//         &self.state.solution_plan.blocks[subprob_idx],
//         &self.givens,
//         initial_unknowns,
//         false, // No parameter scaling for PSO
//     );

//     println!("\n------- pre PSO optimization -------");
//     println!("Initial unknowns: {:?}", initial_unknowns);
//     println!("initial cost: {:?}", pso_subprob.initial_params_cost());

//     // Bound constraints for particle swarm optimizer.
//     // For positive parameters, we'll use [prior * 1e-2, prior * 1e2];
//     // for negative parameters, we'll use [prior * 1e2, prior * 1e-2];
//     let initial_arr = initial_unknowns.to_arr();
//     let bounds = pso_subprob
//         .unknown_idxs
//         .iter()
//         .map(|&idx| {
//             let prior = initial_arr[idx];
//             if prior > 0.0 {
//                 (prior * 1e-2, prior * 1e2)
//             } else if prior < 0.0 {
//                 (prior * 1e2, prior * 1e-2)
//             } else {
//                 unimplemented!(
//                     "Cannot set PSO bounds for parameter idx {} with zero prior value",
//                     idx
//                 );
//             }
//         })
//         .collect::<Vec<(f64, f64)>>();
//     let l_bounds: Vec<f64> = bounds.iter().map(|(l, _u)| *l).collect();
//     let u_bounds: Vec<f64> = bounds.iter().map(|(_l, u)| *u).collect();

//     let pso_solver: ParticleSwarm<_, f64, _> = ParticleSwarm::new((l_bounds, u_bounds), 1000);

//     let max_iters = 10;
//     let opt_result = Executor::new(pso_subprob.clone(), pso_solver)
//         .configure(|state| state.max_iters(max_iters))
//         .run()?;

//     println!("\n------- post PSO optimization -------");

//     let best_particle = opt_result
//         .state
//         .best_individual
//         .ok_or(EqSysError::NoBestPsoIndividual)?;
//     println!("Best particle: {:?}", best_particle);
//     let best_active_params = best_particle.position;

//     // Reconstruct full parameter vector from active parameters
//     let mut best_full_params = initial_unknowns.to_arr();
//     for (i, &idx) in pso_subprob.unknown_idxs.iter().enumerate() {
//         best_full_params[idx] = best_active_params[i];
//     }

//     let per_fn_residual_values = self.residual_fn_engine.call(&best_full_params.to_vec());

//     println!("Per-function residuals at solution:");
//     for (i, res_val) in per_fn_residual_values.iter().enumerate() {
//         let fn_name = self.residual_fns.fn_names[i];
//         println!("   {}: {}", fn_name, res_val);
//     }
//     Ok(DynamicsDerivedParams::from_arr(best_full_params))
// }
