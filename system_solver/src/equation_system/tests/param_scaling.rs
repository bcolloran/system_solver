// use player_dynamics::assert_approx_eq;
// use proptest::prelude::*;
// use test_case::test_case;

// use crate::equation_system::param_scaling::*;

// #[test_case((0.0, 1.0); "x=0, prior=1")]
// #[test_case((1.0, 1.0); "x=1, prior=1")]
// #[test_case((10.0, 1.0); "x=10, prior=1")]
// #[test_case((1.0, 1000.0); "x=1, prior=1000")]
// #[test_case((10.0, 1000.0); "x=10, prior=1000")]
// #[test_case((100.0, 1000.0); "x=100, prior=1000")]
// #[test_case((1000.0, 1000.0); "x=1000, prior=1000")]
// #[test_case((-1.0, 1.0); "x=neg1, prior=1")]
// #[test_case((-1.0, 10.0); "x=neg1, prior=10")]
// #[test_case((1.0, -1000.0); "x=1, prior=neg1000")]
// fn test_softplus_default_round_trip((x, prior): (f64, f64)) {
//     let y = softplus_default(x, prior);
//     let x_reconstructed = softplus_default_inv(y, prior);
//     assert_approx_eq!(x, x_reconstructed);
// }

// #[test_case((0.0, 1.0, 5.0, 0.01); "x=0, prior=1, scale=5, lb=0.01")]
// #[test_case((1.0, 1.0, 10.0, 0.1); "x=1, prior=1, scale=10, lb=0.1")]
// #[test_case((10.0, 5.0, 2.0, 0.05); "x=10, prior=5, scale=2, lb=0.05")]
// #[test_case((-1.0, 2.0, 8.0, 0.02); "x=neg1, prior=2, scale=8, lb=0.02")]
// fn test_softplus_scaled_round_trip((x, prior, scale, lb): (f64, f64, f64, f64)) {
//     let y = softplus_scaled(x, prior, scale, lb);
//     let x_reconstructed = softplus_scaled_inv(y, prior, scale, lb);
//     assert_approx_eq!(x, x_reconstructed);
// }

// #[test]
// fn test_softplus_basic_properties() {
//     // softplus(0) should be ln(2) ≈ 0.693
//     let result = softplus(0.0_f64);
//     assert_approx_eq!(result, 0.693147180559945_f64);

//     // For large positive x, softplus(x) ≈ x
//     let x = 100.0_f64;
//     let result = softplus(x);
//     assert!((result - x).abs() < 1e-10);

//     // For large negative x, softplus(x) ≈ 0
//     let x = -100.0_f64;
//     let result = softplus(x);
//     assert!(result < 1e-40);

//     // softplus is always positive
//     assert!(softplus(-10.0_f64) > 0.0);
//     assert!(softplus(0.0_f64) > 0.0);
//     assert!(softplus(10.0_f64) > 0.0);
// }

// #[test]
// fn test_softplus_scaled_respects_lower_bound() {
//     let x = -100.0_f64; // Very negative input
//     let prior = 1.0_f64;
//     let scale = 10.0_f64;
//     let lb = 0.5_f64;

//     let y = softplus_scaled(x, prior, scale, lb);
//     // Result should be approximately equal to lb (since softplus of very negative is ~0)
//     assert!(y >= lb);
//     assert!((y - lb).abs() < 1e-10);
// }

// #[test]
// fn test_softplus_default_lower_bound() {
//     // For a prior of 100.0, lb should be 1.0
//     let prior = 100.0_f64;
//     let x = -1000.0_f64; // Very negative to approach lower bound
//     let y = softplus_default(x, prior);

//     // y should be approximately prior.abs() * 0.01 = 1.0
//     let expected_lb = prior.abs() * 0.01;
//     assert!(y >= expected_lb);
//     assert!((y - expected_lb).abs() < 1e-10);
// }

// #[test]
// fn test_softplus_output_always_positive() {
//     // Note: For very negative values, floating-point underflow may produce exactly 0.0,
//     // but in practice we don't use softplus with such extreme inputs
//     let test_inputs = vec![-10.0_f64, -1.0_f64, 0.0_f64, 1.0_f64, 10.0_f64, 100.0_f64];
//     for x in test_inputs {
//         let y = softplus(x);
//         assert!(y > 0.0, "softplus({}) = {} should be positive", x, y);
//     }
// }

// // Property-based tests using proptest

// proptest! {
//     /// Property: softplus round-trip should preserve the input value
//     /// For any x, softplus_inv(softplus(x)) should equal x (within numerical tolerance)
//     /// We avoid very negative x values where floating-point precision becomes an issue
//     #[test]
//     fn prop_softplus_roundtrip(x in -10.0_f64..100.0_f64) {
//         let y = softplus(x);
//         let x_reconstructed = softplus_inv(y);

//         // Use relative error for comparison since values can be large
//         let rel_error = ((x - x_reconstructed) / (x.abs() + 1.0)).abs();
//         prop_assert!(rel_error < 1e-9, "x={}, y={}, x_reconstructed={}, rel_error={}",
//                      x, y, x_reconstructed, rel_error);
//     }

//     /// Property: softplus is monotonically increasing
//     /// For any x1 < x2, softplus(x1) < softplus(x2)
//     #[test]
//     fn prop_softplus_monotonic(x1 in -20.0_f64..100.0_f64, delta in 0.001_f64..100.0_f64) {
//         let x2 = x1 + delta;
//         let y1 = softplus(x1);
//         let y2 = softplus(x2);
//         prop_assert!(y2 > y1, "softplus should be monotonic: softplus({}) = {} should be < softplus({}) = {}",
//                      x1, y1, x2, y2);
//     }

//     /// Property: softplus_scaled round-trip with random parameters
//     /// We use non-zero lower bounds to avoid numerical issues near zero
//     #[test]
//     fn prop_softplus_scaled_roundtrip(
//         x in -5.0_f64..100.0_f64,
//         prior in 0.1_f64..1000.0_f64,
//         scale in 2.0_f64..20.0_f64,
//         lb in 0.01_f64..10.0_f64
//     ) {
//         let y = softplus_scaled(x, prior, scale, lb);
//         let x_reconstructed = softplus_scaled_inv(y, prior, scale, lb);

//         let rel_error = ((x - x_reconstructed) / (x.abs() + 1.0)).abs();
//         prop_assert!(rel_error < 1e-8,
//                      "x={}, prior={}, scale={}, lb={}, y={}, x_reconstructed={}, rel_error={}",
//                      x, prior, scale, lb, y, x_reconstructed, rel_error);
//     }

//     /// Property: softplus_default round-trip with random inputs and priors
//     /// We keep x and prior with same sign to avoid extreme edge cases
//     #[test]
//     fn prop_softplus_default_roundtrip(
//         x in -5.0_f64..100.0_f64,
//         prior in 1.0_f64..1000.0_f64
//     ) {
//         let y = softplus_default(x, prior);
//         let x_reconstructed = softplus_default_inv(y, prior);

//         let rel_error = ((x - x_reconstructed) / (x.abs() + 1.0)).abs();
//         prop_assert!(rel_error < 1e-8,
//                      "x={}, prior={}, y={}, x_reconstructed={}, rel_error={}",
//                      x, prior, y, x_reconstructed, rel_error);
//     }

//     /// Property: softplus_scaled output always respects lower bound
//     #[test]
//     fn prop_softplus_scaled_lower_bound(
//         x in -100.0_f64..100.0_f64,
//         prior in 0.1_f64..1000.0_f64,
//         scale in 1.0_f64..20.0_f64,
//         lb in 0.0_f64..10.0_f64
//     ) {
//         let y = softplus_scaled(x, prior, scale, lb);
//         prop_assert!(y >= lb, "y={} should be >= lb={}", y, lb);
//     }

//     /// Property: For large positive x, softplus_scaled(x) ≈ scale*x/prior + lb
//     #[test]
//     fn prop_softplus_scaled_linear_region(
//         x in 10.0_f64..100.0_f64,
//         prior in 0.5_f64..10.0_f64,  // Smaller priors to ensure z is large
//         scale in 5.0_f64..10.0_f64,  // Higher scales
//         lb in 0.01_f64..1.0_f64
//     ) {
//         let z = scale * x / prior;
//         // Only test when z is truly in the linear region (z > 10)
//         prop_assume!(z > 10.0);
//         prop_assume!(z < 200.0);  // Avoid exp overflow

//         let y = softplus_scaled(x, prior, scale, lb);
//         let expected = z + lb;

//         // In the linear region, softplus(z) ≈ z for large z
//         // The relative error should be small
//         let rel_error = ((y - expected) / expected).abs();
//         prop_assert!(rel_error < 0.01,
//                      "In linear region: x={}, prior={}, scale={}, lb={}, z={}, y={}, expected≈{}, rel_error={}",
//                      x, prior, scale, lb, z, y, expected, rel_error);
//     }

//     /// Property: For very negative x, softplus_scaled(x) ≈ lb
//     /// Note: The difference depends on scale - larger scales make the transition sharper
//     #[test]
//     fn prop_softplus_scaled_lower_bound_region(
//         x in -100.0_f64..-10.0_f64,
//         prior in 0.5_f64..10.0_f64,  // Smaller priors to ensure |z| is large
//         scale in 10.0_f64..20.0_f64, // Higher scales for sharper transition
//         lb in 0.1_f64..1.0_f64
//     ) {
//         let z = scale * x / prior;  // This should be very negative
//         // Only test when z is truly very negative
//         prop_assume!(z < -10.0);

//         let y = softplus_scaled(x, prior, scale, lb);

//         // For very negative z, softplus(z) ≈ 0, so y ≈ lb
//         // Check that y is close to lb
//         prop_assert!(y >= lb && y < lb * 1.5,
//                      "For very negative x: x={}, prior={}, scale={}, lb={}, z={}, y={}",
//                      x, prior, scale, lb, z, y);
//     }
// }
