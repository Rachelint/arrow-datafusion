// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::cmp::Ordering;
use std::fmt::{Debug, Formatter};
use std::mem::{size_of, size_of_val};
use std::sync::Arc;

use arrow::array::{
    downcast_integer, ArrowNumericType, BooleanArray, GenericListBuilder,
    GenericListViewArray, ListArray, ListBuilder, PrimitiveArray, PrimitiveBuilder,
};
use arrow::buffer::{OffsetBuffer, ScalarBuffer};
use arrow::{
    array::{ArrayRef, AsArray},
    datatypes::{
        DataType, Decimal128Type, Decimal256Type, Field, Float16Type, Float32Type,
        Float64Type,
    },
};

use arrow::array::Array;
use arrow::array::ArrowNativeTypeOp;
use arrow::datatypes::{ArrowNativeType, ArrowPrimitiveType};

use datafusion_common::{DataFusionError, HashSet, Result, ScalarValue};
use datafusion_expr::function::StateFieldsArgs;
use datafusion_expr::{
    function::AccumulatorArgs, utils::format_state_name, Accumulator, AggregateUDFImpl,
    Documentation, Signature, Volatility,
};
use datafusion_expr::{EmitTo, GroupsAccumulator};
use datafusion_functions_aggregate_common::utils::Hashable;
use datafusion_macros::user_doc;
use datafusion_physical_expr::NullState;

make_udaf_expr_and_func!(
    Median,
    median,
    expression,
    "Computes the median of a set of numbers",
    median_udaf
);

#[user_doc(
    doc_section(label = "General Functions"),
    description = "Returns the median value in the specified column.",
    syntax_example = "median(expression)",
    sql_example = r#"```sql
> SELECT median(column_name) FROM table_name;
+----------------------+
| median(column_name)   |
+----------------------+
| 45.5                 |
+----------------------+
```"#,
    standard_argument(name = "expression", prefix = "The")
)]
/// MEDIAN aggregate expression. If using the non-distinct variation, then this uses a
/// lot of memory because all values need to be stored in memory before a result can be
/// computed. If an approximation is sufficient then APPROX_MEDIAN provides a much more
/// efficient solution.
///
/// If using the distinct variation, the memory usage will be similarly high if the
/// cardinality is high as it stores all distinct values in memory before computing the
/// result, but if cardinality is low then memory usage will also be lower.
pub struct Median {
    signature: Signature,
}

impl Debug for Median {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.debug_struct("Median")
            .field("name", &self.name())
            .field("signature", &self.signature)
            .finish()
    }
}

impl Default for Median {
    fn default() -> Self {
        Self::new()
    }
}

impl Median {
    pub fn new() -> Self {
        Self {
            signature: Signature::numeric(1, Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for Median {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "median"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(arg_types[0].clone())
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<Field>> {
        //Intermediate state is a list of the elements we have collected so far
        let field = Field::new_list_field(args.input_types[0].clone(), true);
        let state_name = if args.is_distinct {
            "distinct_median"
        } else {
            "median"
        };

        Ok(vec![Field::new(
            format_state_name(args.name, state_name),
            DataType::List(Arc::new(field)),
            true,
        )])
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        macro_rules! helper {
            ($t:ty, $dt:expr) => {
                if acc_args.is_distinct {
                    Ok(Box::new(DistinctMedianAccumulator::<$t> {
                        data_type: $dt.clone(),
                        distinct_values: HashSet::new(),
                    }))
                } else {
                    Ok(Box::new(MedianAccumulator::<$t> {
                        data_type: $dt.clone(),
                        all_values: vec![],
                    }))
                }
            };
        }

        let dt = acc_args.exprs[0].data_type(acc_args.schema)?;
        downcast_integer! {
            dt => (helper, dt),
            DataType::Float16 => helper!(Float16Type, dt),
            DataType::Float32 => helper!(Float32Type, dt),
            DataType::Float64 => helper!(Float64Type, dt),
            DataType::Decimal128(_, _) => helper!(Decimal128Type, dt),
            DataType::Decimal256(_, _) => helper!(Decimal256Type, dt),
            _ => Err(DataFusionError::NotImplemented(format!(
                "MedianAccumulator not supported for {} with {}",
                acc_args.name,
                dt,
            ))),
        }
    }

    fn aliases(&self) -> &[String] {
        &[]
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

/// The median accumulator accumulates the raw input values
/// as `ScalarValue`s
///
/// The intermediate state is represented as a List of scalar values updated by
/// `merge_batch` and a `Vec` of `ArrayRef` that are converted to scalar values
/// in the final evaluation step so that we avoid expensive conversions and
/// allocations during `update_batch`.
struct MedianAccumulator<T: ArrowNumericType> {
    data_type: DataType,
    all_values: Vec<T::Native>,
}

impl<T: ArrowNumericType> Debug for MedianAccumulator<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "MedianAccumulator({})", self.data_type)
    }
}

impl<T: ArrowNumericType> Accumulator for MedianAccumulator<T> {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let all_values = self
            .all_values
            .iter()
            .map(|x| ScalarValue::new_primitive::<T>(Some(*x), &self.data_type))
            .collect::<Result<Vec<_>>>()?;

        let arr = ScalarValue::new_list_nullable(&all_values, &self.data_type);
        Ok(vec![ScalarValue::List(arr)])
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let values = values[0].as_primitive::<T>();
        self.all_values.reserve(values.len() - values.null_count());
        self.all_values.extend(values.iter().flatten());
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let array = states[0].as_list::<i32>();
        for v in array.iter().flatten() {
            self.update_batch(&[v])?
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let d = std::mem::take(&mut self.all_values);
        let median = calculate_median::<T>(d);
        ScalarValue::new_primitive::<T>(median, &self.data_type)
    }

    fn size(&self) -> usize {
        size_of_val(self) + self.all_values.capacity() * size_of::<T::Native>()
    }
}

/// The median groups accumulator accumulates the raw input values
///
/// For calculating the accurate medians of groups, we need to store all values
/// of groups before final evaluation.
/// And values in each group will be stored in a `Vec<T>`, so the total group values
/// will be actually organized as a `Vec<Vec<T>>`.
///
/// In partial aggregation stage, the `values`
///
#[derive(Debug)]
struct MedianGroupsAccumulator<T: ArrowNumericType + Send> {
    data_type: DataType,
    group_values: Vec<Vec<T::Native>>,
    null_state: NullState,
}

impl<T: ArrowNumericType + Send> MedianGroupsAccumulator<T> {
    pub fn new(data_type: DataType) -> Self {
        Self {
            data_type,
            group_values: Vec::new(),
            null_state: NullState::new(),
        }
    }
}

impl<T: ArrowNumericType + Send> GroupsAccumulator for MedianGroupsAccumulator<T> {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        assert_eq!(values.len(), 1, "single argument to update_batch");
        let values = values[0].as_primitive::<T>();

        // increment counts, update sums
        self.group_values.resize(total_num_groups, Vec::new());
        self.null_state.accumulate(
            group_indices,
            values,
            opt_filter,
            total_num_groups,
            |group_index, new_value| {
                self.group_values[group_index].push(new_value);
            },
        );

        Ok(())
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        // Since aggregate filter should be applied in partial stage, in final stage there should be no filter
        _opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        assert_eq!(values.len(), 1, "one argument to merge_batch");

        // The merged values should be organized like as a `non-nullable ListArray` like:
        //
        // ```text
        //   group 0: [1, 2, 3]
        //   group 1: [4, 5]
        //   group 2: [6, 7, 8]
        //   ...
        //   group n: [...]
        // ```
        //
        let input_group_values = values[0].as_list::<i32>();
        assert!(input_group_values.null_count() == 0);

        // Ensure group values big enough
        self.group_values.resize(total_num_groups, Vec::new());

        // Extend values to related groups
        group_indices
            .iter()
            .zip(input_group_values.iter())
            .for_each(|(&group_index, values_opt)| {
                let values = values_opt.unwrap();
                let values = values.as_primitive::<T>();
                self.group_values[group_index].extend(values.values().iter());
            });

        Ok(())
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let emit_group_values = emit_to.take_needed(&mut self.group_values);

        // Build offsets
        let mut offsets = Vec::with_capacity(self.group_values.len() + 1);
        offsets.push(0);
        let mut cur_len = 0;
        for group_value in &emit_group_values {
            cur_len += group_value.len() as i32;
            offsets.push(cur_len);
        }
        let offsets = OffsetBuffer::new(ScalarBuffer::from(offsets));

        // Build inner array
        let flatten_group_values =
            emit_group_values.into_iter().flatten().collect::<Vec<_>>();
        let group_values_array =
            PrimitiveArray::<T>::new(ScalarBuffer::from(flatten_group_values), None);

        // Build the result list array
        let result_list_array = ListArray::new(
            Arc::new(Field::new_list_field(self.data_type.clone(), false)),
            offsets,
            Arc::new(group_values_array),
            None,
        );

        Ok(vec![Arc::new(result_list_array)])
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        todo!()
    }

    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        todo!()
    }

    fn supports_convert_to_state(&self) -> bool {
        todo!()
    }

    fn size(&self) -> usize {
        todo!()
    }
}

/// The distinct median accumulator accumulates the raw input values
/// as `ScalarValue`s
///
/// The intermediate state is represented as a List of scalar values updated by
/// `merge_batch` and a `Vec` of `ArrayRef` that are converted to scalar values
/// in the final evaluation step so that we avoid expensive conversions and
/// allocations during `update_batch`.
struct DistinctMedianAccumulator<T: ArrowNumericType> {
    data_type: DataType,
    distinct_values: HashSet<Hashable<T::Native>>,
}

impl<T: ArrowNumericType> Debug for DistinctMedianAccumulator<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "DistinctMedianAccumulator({})", self.data_type)
    }
}

impl<T: ArrowNumericType> Accumulator for DistinctMedianAccumulator<T> {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let all_values = self
            .distinct_values
            .iter()
            .map(|x| ScalarValue::new_primitive::<T>(Some(x.0), &self.data_type))
            .collect::<Result<Vec<_>>>()?;

        let arr = ScalarValue::new_list_nullable(&all_values, &self.data_type);
        Ok(vec![ScalarValue::List(arr)])
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }

        let array = values[0].as_primitive::<T>();
        match array.nulls().filter(|x| x.null_count() > 0) {
            Some(n) => {
                for idx in n.valid_indices() {
                    self.distinct_values.insert(Hashable(array.value(idx)));
                }
            }
            None => array.values().iter().for_each(|x| {
                self.distinct_values.insert(Hashable(*x));
            }),
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let array = states[0].as_list::<i32>();
        for v in array.iter().flatten() {
            self.update_batch(&[v])?
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let d = std::mem::take(&mut self.distinct_values)
            .into_iter()
            .map(|v| v.0)
            .collect::<Vec<_>>();
        let median = calculate_median::<T>(d);
        ScalarValue::new_primitive::<T>(median, &self.data_type)
    }

    fn size(&self) -> usize {
        size_of_val(self) + self.distinct_values.capacity() * size_of::<T::Native>()
    }
}

/// Get maximum entry in the slice,
fn slice_max<T>(array: &[T::Native]) -> T::Native
where
    T: ArrowPrimitiveType,
    T::Native: PartialOrd, // Ensure the type supports PartialOrd for comparison
{
    // Make sure that, array is not empty.
    debug_assert!(!array.is_empty());
    // `.unwrap()` is safe here as the array is supposed to be non-empty
    *array
        .iter()
        .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Less))
        .unwrap()
}

fn calculate_median<T: ArrowNumericType>(
    mut values: Vec<T::Native>,
) -> Option<T::Native> {
    let cmp = |x: &T::Native, y: &T::Native| x.compare(*y);

    let len = values.len();
    if len == 0 {
        None
    } else if len % 2 == 0 {
        let (low, high, _) = values.select_nth_unstable_by(len / 2, cmp);
        // Get the maximum of the low (left side after bi-partitioning)
        let left_max = slice_max::<T>(low);
        let median = left_max
            .add_wrapping(*high)
            .div_wrapping(T::Native::usize_as(2));
        Some(median)
    } else {
        let (_, median, _) = values.select_nth_unstable_by(len / 2, cmp);
        Some(*median)
    }
}
