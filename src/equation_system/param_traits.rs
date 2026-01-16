use ad_trait::AD;
use struct_to_array::StructToArray;

/// Trait for "Given" parameters - the fixed parameters that define a problem instance.
/// These are the design parameters that are chosen manually.
///
/// Implementors must be generic over a numeric type `T: AD` and provide conversions
/// between the f64 and generic AD versions.
///
/// # Example
/// ```ignore
/// #[derive(Clone, Copy, Debug, StructToArray)]
/// struct MyGivens<T> {
///     mass: T,
///     max_speed: T,
/// }
///
/// impl MyGivens<f64> {
///     pub fn to_ad<T: AD>(self) -> MyGivens<T> {
///         MyGivens {
///             mass: T::constant(self.mass),
///             max_speed: T::constant(self.max_speed),
///         }
///     }
/// }
/// ```
pub trait GivenParams: Clone + Copy + std::fmt::Debug {}

/// Trait for "Unknown" parameters - the parameters that will be solved for.
/// These are typically low-level parameters derived from the given parameters.
///
/// Implementors must be generic over a numeric type `T: AD` and provide:
/// - The number of parameters (N)
/// - Field names for debugging
/// - Conversions between f64 and generic AD versions
///
/// # Example
/// ```ignore
/// #[derive(Clone, Copy, Debug, StructToArray, FieldNames)]
/// struct MyUnknowns<T> {
///     drag_coeff: T,
///     thrust_max: T,
/// }
///
/// impl MyUnknowns<f64> {
///     pub fn to_ad<T: AD>(&self) -> MyUnknowns<T> {
///         MyUnknowns {
///             drag_coeff: T::constant(self.drag_coeff),
///             thrust_max: T::constant(self.thrust_max),
///         }
///     }
/// }
///
/// impl<T> MyUnknowns<T> {
///     pub const N: usize = 2;
///     pub fn field_names() -> &'static [&'static str] {
///         &["drag_coeff", "thrust_max"]
///     }
/// }
/// ```
pub trait UnknownParams: Clone + Copy + std::fmt::Debug {}

/// Marker trait to ensure that types implementing GivenParams are properly
/// bounded. Note that GivenParams do NOT need StructToArray - they are just
/// passed to residual functions and never converted to arrays.
pub trait GivenParamsFor<T, const N: usize>: GivenParams
where
    T: AD,
{
}

/// Automatically implement for any type that satisfies the bounds
impl<T, G, const N: usize> GivenParamsFor<T, N> for G
where
    T: AD,
    G: GivenParams,
{
}

/// Marker trait to ensure that types implementing UnknownParams are properly
/// bounded by AD and StructToArray
pub trait UnknownParamsFor<T, const N: usize>: UnknownParams + StructToArray<T, N>
where
    T: AD,
{
}

/// Automatically implement for any type that satisfies the bounds
impl<T, U, const N: usize> UnknownParamsFor<T, N> for U
where
    T: AD,
    U: UnknownParams + StructToArray<T, N>,
{
}
