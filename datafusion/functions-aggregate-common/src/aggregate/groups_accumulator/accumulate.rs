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

//! [`GroupsAccumulator`] helpers: [`NullState`] and [`accumulate_indices`]
//!
//! [`GroupsAccumulator`]: datafusion_expr_common::groups_accumulator::GroupsAccumulator

use std::fmt::Debug;
use std::marker::PhantomData;

use arrow::array::{Array, BooleanArray, BooleanBufferBuilder, PrimitiveArray};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::datatypes::ArrowPrimitiveType;

use datafusion_expr_common::groups_accumulator::EmitTo;

use crate::aggregate::groups_accumulator::blocks::{Block, Blocks};
use crate::aggregate::groups_accumulator::group_index_operations::{
    BlockedGroupIndexOperations, FlatGroupIndexOperations, GroupIndexOperations,
};

/// Track the accumulator null state per row: if any values for that
/// group were null and if any values have been seen at all for that group.
///
/// This is part of the inner loop for many [`GroupsAccumulator`]s,
/// and thus the performance is critical and so there are multiple
/// specialized implementations, invoked depending on the specific
/// combinations of the input.
///
/// Typically there are 4 potential combinations of inputs must be
/// special cased for performance:
///
/// * With / Without filter
/// * With / Without nulls in the input
///
/// If the input has nulls, then the accumulator must potentially
/// handle each input null value specially (e.g. for `SUM` to mark the
/// corresponding sum as null)
///
/// If there are filters present, `NullState` tracks if it has seen
/// *any* value for that group (as some values may be filtered
/// out). Without a filter, the accumulator is only passed groups that
/// had at least one value to accumulate so they do not need to track
/// if they have seen values for a particular group.
///
/// [`GroupsAccumulator`]: datafusion_expr_common::groups_accumulator::GroupsAccumulator
#[derive(Debug)]
pub struct NullState<O: GroupIndexOperations> {
    /// Have we seen any non-filtered input values for `group_index`?
    ///
    /// If `seen_values[i]` is true, have seen at least one non null
    /// value for group `i`
    ///
    /// If `seen_values[i]` is false, have not seen any values that
    /// pass the filter yet for group `i`
    seen_values: Blocks<BooleanBufferBuilder>,

    /// phantom data for required type `<O>`
    _phantom: PhantomData<O>,
}

impl<O: GroupIndexOperations> NullState<O> {
    /// return the size of all buffers allocated by this null state, not including self
    pub fn size(&self) -> usize {
        // capacity is in bits, so convert to bytes
        self.seen_values.size() / 8
    }

    /// Invokes `value_fn(group_index, value)` for each non null, non
    /// filtered value of `value`, while tracking which groups have
    /// seen null inputs and which groups have seen any inputs if necessary
    //
    /// # Arguments:
    ///
    /// * `values`: the input arguments to the accumulator
    /// * `group_indices`:  To which groups do the rows in `values` belong, (aka group_index)
    /// * `opt_filter`: if present, only rows for which is Some(true) are included
    /// * `value_fn`: function invoked for  (group_index, value) where value is non null
    ///
    /// See [`accumulate`], for more details on how value_fn is called
    ///
    /// When value_fn is called it also sets
    ///
    /// 1. `self.seen_values[group_index]` to true for all rows that had a non null value
    pub fn accumulate<T, F>(
        &mut self,
        group_indices: &[usize],
        values: &PrimitiveArray<T>,
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
        mut value_fn: F,
    ) where
        T: ArrowPrimitiveType + Send,
        F: FnMut(u32, u64, T::Native) + Send,
    {
        // ensure the seen_values is big enough (start everything at
        // "not seen" valid)
        let new_block = |block_size: Option<usize>| {
            BooleanBufferBuilder::new(block_size.unwrap_or(0))
        };
        self.seen_values.resize(total_num_groups, new_block, false);

        let seen_values = &mut self.seen_values;
        accumulate(group_indices, values, opt_filter, |packed_index, value| {
            let packed_index = packed_index as u64;
            let block_id = O::get_block_id(packed_index);
            let block_offset = O::get_block_offset(packed_index);
            seen_values.set_bit(block_id, block_offset, true);
            value_fn(block_id, block_offset, value);
        });
    }

    /// Invokes `value_fn(group_index, value)` for each non null, non
    /// filtered value in `values`, while tracking which groups have
    /// seen null inputs and which groups have seen any inputs, for
    /// [`BooleanArray`]s.
    ///
    /// Since `BooleanArray` is not a [`PrimitiveArray`] it must be
    /// handled specially.
    ///
    /// See [`Self::accumulate`], which handles `PrimitiveArray`s, for
    /// more details on other arguments.
    pub fn accumulate_boolean<F>(
        &mut self,
        group_indices: &[usize],
        values: &BooleanArray,
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
        mut value_fn: F,
    ) where
        F: FnMut(u32, u64, bool) + Send,
    {
        let data = values.values();
        assert_eq!(data.len(), group_indices.len());

        // ensure the seen_values is big enough (start everything at
        // "not seen" valid)
        let new_block = |block_size: Option<usize>| {
            BooleanBufferBuilder::new(block_size.unwrap_or(0))
        };
        self.seen_values.resize(total_num_groups, new_block, false);

        // These could be made more performant by iterating in chunks of 64 bits at a time
        let seen_values = &mut self.seen_values;
        match (values.null_count() > 0, opt_filter) {
            // no nulls, no filter,
            (false, None) => {
                // if we have previously seen nulls, ensure the null
                // buffer is big enough (start everything at valid)
                group_indices.iter().zip(data.iter()).for_each(
                    |(&packed_index, new_value)| {
                        let packed_index = packed_index as u64;
                        let block_id = O::get_block_id(packed_index);
                        let block_offset = O::get_block_offset(packed_index);
                        seen_values.set_bit(block_id, block_offset, true);
                        value_fn(block_id, block_offset, new_value)
                    },
                )
            }
            // nulls, no filter
            (true, None) => {
                let nulls = values.nulls().unwrap();
                group_indices
                    .iter()
                    .zip(data.iter())
                    .zip(nulls.iter())
                    .for_each(|((&packed_index, new_value), is_valid)| {
                        if is_valid {
                            let packed_index = packed_index as u64;
                            let block_id = O::get_block_id(packed_index);
                            let block_offset = O::get_block_offset(packed_index);
                            seen_values.set_bit(block_id, block_offset, true);
                            value_fn(block_id, block_offset, new_value);
                        }
                    })
            }
            // no nulls, but a filter
            (false, Some(filter)) => {
                assert_eq!(filter.len(), group_indices.len());

                group_indices
                    .iter()
                    .zip(data.iter())
                    .zip(filter.iter())
                    .for_each(|((&packed_index, new_value), filter_value)| {
                        if let Some(true) = filter_value {
                            let packed_index = packed_index as u64;
                            let block_id = O::get_block_id(packed_index);
                            let block_offset = O::get_block_offset(packed_index);
                            seen_values.set_bit(block_id, block_offset, true);
                            value_fn(block_id, block_offset, new_value);
                        }
                    })
            }
            // both null values and filters
            (true, Some(filter)) => {
                assert_eq!(filter.len(), group_indices.len());
                filter
                    .iter()
                    .zip(group_indices.iter())
                    .zip(values.iter())
                    .for_each(|((filter_value, &packed_index), new_value)| {
                        if let Some(true) = filter_value {
                            if let Some(new_value) = new_value {
                                let packed_index = packed_index as u64;
                                let block_id = O::get_block_id(packed_index);
                                let block_offset = O::get_block_offset(packed_index);
                                seen_values.set_bit(block_id, block_offset, true);
                                value_fn(block_id, block_offset, new_value);
                            }
                        }
                    })
            }
        }
    }

    /// Creates the a [`NullBuffer`] representing which group_indices
    /// should have null values (because they never saw any values)
    /// for the `emit_to` rows.
    ///
    /// resets the internal state appropriately
    pub fn build(&mut self, emit_to: EmitTo) -> NullBuffer {
        self.seen_values.emit(emit_to)
    }
}

/// Adapter for supporting dynamic dispatching of [`FlatNullState`] and [`BlockedNullState`].
/// For performance, the cost of batch-level dynamic dispatching is acceptable.
#[derive(Debug)]
pub enum NullStateAdapter {
    Flat(FlatNullState),
    Blocked(BlockedNullState),
}

impl NullStateAdapter {
    pub fn new(block_size: Option<usize>) -> Self {
        if let Some(blk_size) = block_size {
            Self::Blocked(BlockedNullState::new(blk_size))
        } else {
            Self::Flat(FlatNullState::new())
        }
    }

    pub fn accumulate<T, F>(
        &mut self,
        group_indices: &[usize],
        values: &PrimitiveArray<T>,
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
        value_fn: F,
    ) where
        T: ArrowPrimitiveType + Send,
        F: FnMut(u32, u64, T::Native) + Send,
    {
        match self {
            NullStateAdapter::Flat(null_state) => null_state.accumulate(
                group_indices,
                values,
                opt_filter,
                total_num_groups,
                value_fn,
            ),
            NullStateAdapter::Blocked(null_state) => null_state.accumulate(
                group_indices,
                values,
                opt_filter,
                total_num_groups,
                value_fn,
            ),
        }
    }

    pub fn accumulate_boolean<F>(
        &mut self,
        group_indices: &[usize],
        values: &BooleanArray,
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
        value_fn: F,
    ) where
        F: FnMut(u32, u64, bool) + Send,
    {
        match self {
            NullStateAdapter::Flat(null_state) => null_state.accumulate_boolean(
                group_indices,
                values,
                opt_filter,
                total_num_groups,
                value_fn,
            ),
            NullStateAdapter::Blocked(null_state) => null_state.accumulate_boolean(
                group_indices,
                values,
                opt_filter,
                total_num_groups,
                value_fn,
            ),
        }
    }

    pub fn build(&mut self, emit_to: EmitTo) -> NullBuffer {
        match self {
            NullStateAdapter::Flat(null_state) => null_state.build(emit_to),
            NullStateAdapter::Blocked(null_state) => null_state.build(emit_to),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            NullStateAdapter::Flat(null_state) => null_state.size(),
            NullStateAdapter::Blocked(null_state) => null_state.size(),
        }
    }

    /// Clone and build a single [`BooleanBuffer`] from `seen_values`,
    /// only used for testing.
    #[cfg(test)]
    fn build_cloned_seen_values(&self) -> BooleanBuffer {
        match self {
            NullStateAdapter::Flat(null_state) => {
                null_state.seen_values[0].finish_cloned()
            }
            NullStateAdapter::Blocked(null_state) => {
                let mut return_builder = BooleanBufferBuilder::new(0);
                let num_blocks = null_state.seen_values.len();
                for blk_idx in 0..num_blocks {
                    let builder = &null_state.seen_values[blk_idx];
                    for idx in 0..builder.len() {
                        return_builder.append(builder.get_bit(idx));
                    }
                }
                return_builder.finish()
            }
        }
    }

    #[cfg(test)]
    fn build_all_in_once(&mut self) -> NullBuffer {
        match self {
            NullStateAdapter::Flat(null_state) => null_state.build(EmitTo::All),
            NullStateAdapter::Blocked(null_state) => {
                let mut return_builder = BooleanBufferBuilder::new(0);
                let num_blocks = null_state.seen_values.len();
                for _ in 0..num_blocks {
                    let blocked_nulls = null_state.build(EmitTo::NextBlock);
                    for bit in blocked_nulls.inner().iter() {
                        return_builder.append(bit);
                    }
                }

                NullBuffer::new(return_builder.finish())
            }
        }
    }
}

/// [`NullState`] for `flat groups input`
///
/// At first, you may need to see something about `block_id` and `block_offset`
/// from [`GroupsAccumulator::supports_blocked_groups`].
///
/// The `flat groups input` are organized like:
///
/// ```text
///     row_0 group_index_0
///     row_1 group_index_1
///     row_2 group_index_2
///     ...
///     row_n group_index_n     
/// ```
///
/// If `row_x group_index_x` is not filtered(`group_index_x` is seen)
/// `seen_values[group_index_x]` will be set to `true`.
///
/// For `set_bit(block_id, block_offset, value)`, `block_id` is unused,
/// `block_offset` will be set to `group_index`.
///
/// [`GroupsAccumulator::supports_blocked_groups`]: datafusion_expr_common::groups_accumulator::GroupsAccumulator::supports_blocked_groups
///
pub type FlatNullState = NullState<FlatGroupIndexOperations>;

impl FlatNullState {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for FlatNullState {
    fn default() -> Self {
        Self {
            seen_values: Blocks::new(None),
            _phantom: PhantomData {},
        }
    }
}

/// [`NullState`] for `blocked groups input`
///
/// At first, you may need to see something about `block_id` and `block_offset`
/// from [`GroupsAccumulator::supports_blocked_groups`].
///
/// The `flat groups input` are organized like:
///
/// ```text
///     row_0 (block_id_0, block_offset_0)
///     row_1 (block_id_1, block_offset_1)
///     row_2 (block_id_1, block_offset_1)
///     ...
///     row_n (block_id_n, block_offset_n)    
/// ```
///
/// If `row_x (block_id_x, block_offset_x)` is not filtered
/// (`block_id_x, block_offset_x` is seen), `seen_values[block_id_x][block_offset_x]`
/// will be set to `true`.
///
/// [`GroupsAccumulator::supports_blocked_groups`]: datafusion_expr_common::groups_accumulator::GroupsAccumulator::supports_blocked_groups
///
pub type BlockedNullState = NullState<BlockedGroupIndexOperations>;

impl BlockedNullState {
    pub fn new(block_size: usize) -> Self {
        Self {
            seen_values: Blocks::new(Some(block_size)),
            _phantom: PhantomData {},
        }
    }
}

impl Block for BooleanBufferBuilder {
    type T = bool;

    fn fill_default_value(&mut self, fill_len: usize, default_value: Self::T) {
        self.append_n(fill_len, default_value);
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn size(&self) -> usize {
        self.capacity()
    }
}

impl Blocks<BooleanBufferBuilder> {
    fn set_bit(&mut self, block_id: u32, block_offset: u64, value: bool) {
        self[block_id as usize].set_bit(block_offset as usize, value);
    }

    fn emit(&mut self, emit_to: EmitTo) -> NullBuffer {
        let nulls = match emit_to {
            EmitTo::All | EmitTo::First(_) => self[0].finish(),
            EmitTo::NextBlock => {
                let mut block = self
                    .pop_block()
                    .expect("should not try to emit empty blocks");
                block.finish()
            }
        };

        let nulls = if let EmitTo::First(n) = emit_to {
            // split off the first N values in seen_values
            //
            // TODO make this more efficient rather than two
            // copies and bitwise manipulation
            let first_n_null: BooleanBuffer = nulls.iter().take(n).collect();
            // reset the existing seen buffer
            for seen in nulls.iter().skip(n) {
                self[0].append(seen);
            }
            first_n_null
        } else {
            nulls
        };

        NullBuffer::new(nulls)
    }
}

/// Invokes `value_fn(group_index, value)` for each non null, non
/// filtered value of `value`,
///
/// # Arguments:
///
/// * `group_indices`:  To which groups do the rows in `values` belong, (aka group_index)
/// * `values`: the input arguments to the accumulator
/// * `opt_filter`: if present, only rows for which is Some(true) are included
/// * `value_fn`: function invoked for  (group_index, value) where value is non null
///
/// # Example
///
/// ```text
///  ┌─────────┐   ┌─────────┐   ┌ ─ ─ ─ ─ ┐
///  │ ┌─────┐ │   │ ┌─────┐ │     ┌─────┐
///  │ │  2  │ │   │ │ 200 │ │   │ │  t  │ │
///  │ ├─────┤ │   │ ├─────┤ │     ├─────┤
///  │ │  2  │ │   │ │ 100 │ │   │ │  f  │ │
///  │ ├─────┤ │   │ ├─────┤ │     ├─────┤
///  │ │  0  │ │   │ │ 200 │ │   │ │  t  │ │
///  │ ├─────┤ │   │ ├─────┤ │     ├─────┤
///  │ │  1  │ │   │ │ 200 │ │   │ │NULL │ │
///  │ ├─────┤ │   │ ├─────┤ │     ├─────┤
///  │ │  0  │ │   │ │ 300 │ │   │ │  t  │ │
///  │ └─────┘ │   │ └─────┘ │     └─────┘
///  └─────────┘   └─────────┘   └ ─ ─ ─ ─ ┘
///
/// group_indices   values        opt_filter
/// ```
///
/// In the example above, `value_fn` is invoked for each (group_index,
/// value) pair where `opt_filter[i]` is true and values is non null
///
/// ```text
/// value_fn(2, 200)
/// value_fn(0, 200)
/// value_fn(0, 300)
/// ```
pub fn accumulate<T, F>(
    group_indices: &[usize],
    values: &PrimitiveArray<T>,
    opt_filter: Option<&BooleanArray>,
    mut value_fn: F,
) where
    T: ArrowPrimitiveType + Send,
    F: FnMut(usize, T::Native) + Send,
{
    let data: &[T::Native] = values.values();
    assert_eq!(data.len(), group_indices.len());

    match (values.null_count() > 0, opt_filter) {
        // no nulls, no filter,
        (false, None) => {
            let iter = group_indices.iter().zip(data.iter());
            for (&group_index, &new_value) in iter {
                value_fn(group_index, new_value);
            }
        }
        // nulls, no filter
        (true, None) => {
            let nulls = values.nulls().unwrap();
            // This is based on (ahem, COPY/PASTE) arrow::compute::aggregate::sum
            // iterate over in chunks of 64 bits for more efficient null checking
            let group_indices_chunks = group_indices.chunks_exact(64);
            let data_chunks = data.chunks_exact(64);
            let bit_chunks = nulls.inner().bit_chunks();

            let group_indices_remainder = group_indices_chunks.remainder();
            let data_remainder = data_chunks.remainder();

            group_indices_chunks
                .zip(data_chunks)
                .zip(bit_chunks.iter())
                .for_each(|((group_index_chunk, data_chunk), mask)| {
                    // index_mask has value 1 << i in the loop
                    let mut index_mask = 1;
                    group_index_chunk.iter().zip(data_chunk.iter()).for_each(
                        |(&group_index, &new_value)| {
                            // valid bit was set, real value
                            let is_valid = (mask & index_mask) != 0;
                            if is_valid {
                                value_fn(group_index, new_value);
                            }
                            index_mask <<= 1;
                        },
                    )
                });

            // handle any remaining bits (after the initial 64)
            let remainder_bits = bit_chunks.remainder_bits();
            group_indices_remainder
                .iter()
                .zip(data_remainder.iter())
                .enumerate()
                .for_each(|(i, (&group_index, &new_value))| {
                    let is_valid = remainder_bits & (1 << i) != 0;
                    if is_valid {
                        value_fn(group_index, new_value);
                    }
                });
        }
        // no nulls, but a filter
        (false, Some(filter)) => {
            assert_eq!(filter.len(), group_indices.len());
            // The performance with a filter could be improved by
            // iterating over the filter in chunks, rather than a single
            // iterator. TODO file a ticket
            group_indices
                .iter()
                .zip(data.iter())
                .zip(filter.iter())
                .for_each(|((&group_index, &new_value), filter_value)| {
                    if let Some(true) = filter_value {
                        value_fn(group_index, new_value);
                    }
                })
        }
        // both null values and filters
        (true, Some(filter)) => {
            assert_eq!(filter.len(), group_indices.len());
            // The performance with a filter could be improved by
            // iterating over the filter in chunks, rather than using
            // iterators. TODO file a ticket
            filter
                .iter()
                .zip(group_indices.iter())
                .zip(values.iter())
                .for_each(|((filter_value, &group_index), new_value)| {
                    if let Some(true) = filter_value {
                        if let Some(new_value) = new_value {
                            value_fn(group_index, new_value)
                        }
                    }
                })
        }
    }
}

/// Accumulates with multiple accumulate(value) columns. (e.g. `corr(c1, c2)`)
///
/// This method assumes that for any input record index, if any of the value column
/// is null, or it's filtered out by `opt_filter`, then the record would be ignored.
/// (won't be accumulated by `value_fn`)
///
/// # Arguments
///
/// * `group_indices` - To which groups do the rows in `value_columns` belong
/// * `value_columns` - The input arrays to accumulate
/// * `opt_filter` - Optional filter array. If present, only rows where filter is `Some(true)` are included
/// * `value_fn` - Callback function for each valid row, with parameters:
///     * `group_idx`: The group index for the current row
///     * `batch_idx`: The index of the current row in the input arrays
///     * `columns`: Reference to all input arrays for accessing values
// TODO: support `blocked group index` for `accumulate_multiple`
// (for supporting `blocked group index` for correlation group accumulator)
pub fn accumulate_multiple<T, F>(
    group_indices: &[usize],
    value_columns: &[&PrimitiveArray<T>],
    opt_filter: Option<&BooleanArray>,
    mut value_fn: F,
) where
    T: ArrowPrimitiveType + Send,
    F: FnMut(usize, usize, &[&PrimitiveArray<T>]) + Send,
{
    // Calculate `valid_indices` to accumulate, non-valid indices are ignored.
    // `valid_indices` is a bit mask corresponding to the `group_indices`. An index
    // is considered valid if:
    // 1. All columns are non-null at this index.
    // 2. Not filtered out by `opt_filter`

    // Take AND from all null buffers of `value_columns`.
    let combined_nulls = value_columns
        .iter()
        .map(|arr| arr.logical_nulls())
        .fold(None, |acc, nulls| {
            NullBuffer::union(acc.as_ref(), nulls.as_ref())
        });

    // Take AND from previous combined nulls and `opt_filter`.
    let valid_indices = match (combined_nulls, opt_filter) {
        (None, None) => None,
        (None, Some(filter)) => Some(filter.clone()),
        (Some(nulls), None) => Some(BooleanArray::new(nulls.inner().clone(), None)),
        (Some(nulls), Some(filter)) => {
            let combined = nulls.inner() & filter.values();
            Some(BooleanArray::new(combined, None))
        }
    };

    for col in value_columns.iter() {
        debug_assert_eq!(col.len(), group_indices.len());
    }

    match valid_indices {
        None => {
            for (batch_idx, &group_idx) in group_indices.iter().enumerate() {
                value_fn(group_idx, batch_idx, value_columns);
            }
        }
        Some(valid_indices) => {
            for (batch_idx, &group_idx) in group_indices.iter().enumerate() {
                if valid_indices.value(batch_idx) {
                    value_fn(group_idx, batch_idx, value_columns);
                }
            }
        }
    }
}

/// This function is called to update the accumulator state per row
/// when the value is not needed (e.g. COUNT)
///
/// `F`: Invoked like `value_fn(group_index) for all non null values
/// passing the filter. Note that no tracking is done for null inputs
/// or which groups have seen any values
///
/// See [`NullState::accumulate`], for more details on other
/// arguments.
// TODO: support `blocked group index` for `accumulate_indices`
// (for supporting `blocked group index` for count group accumulator)
pub fn accumulate_indices<F>(
    group_indices: &[usize],
    nulls: Option<&NullBuffer>,
    opt_filter: Option<&BooleanArray>,
    mut index_fn: F,
) where
    F: FnMut(usize) + Send,
{
    match (nulls, opt_filter) {
        (None, None) => {
            for &group_index in group_indices.iter() {
                index_fn(group_index)
            }
        }
        (None, Some(filter)) => {
            debug_assert_eq!(filter.len(), group_indices.len());
            let group_indices_chunks = group_indices.chunks_exact(64);
            let bit_chunks = filter.values().bit_chunks();

            let group_indices_remainder = group_indices_chunks.remainder();

            group_indices_chunks.zip(bit_chunks.iter()).for_each(
                |(group_index_chunk, mask)| {
                    // index_mask has value 1 << i in the loop
                    let mut index_mask = 1;
                    group_index_chunk.iter().for_each(|&group_index| {
                        // valid bit was set, real vale
                        let is_valid = (mask & index_mask) != 0;
                        if is_valid {
                            index_fn(group_index);
                        }
                        index_mask <<= 1;
                    })
                },
            );

            // handle any remaining bits (after the initial 64)
            let remainder_bits = bit_chunks.remainder_bits();
            group_indices_remainder
                .iter()
                .enumerate()
                .for_each(|(i, &group_index)| {
                    let is_valid = remainder_bits & (1 << i) != 0;
                    if is_valid {
                        index_fn(group_index)
                    }
                });
        }
        (Some(valids), None) => {
            debug_assert_eq!(valids.len(), group_indices.len());
            // This is based on (ahem, COPY/PASTA) arrow::compute::aggregate::sum
            // iterate over in chunks of 64 bits for more efficient null checking
            let group_indices_chunks = group_indices.chunks_exact(64);
            let bit_chunks = valids.inner().bit_chunks();

            let group_indices_remainder = group_indices_chunks.remainder();

            group_indices_chunks.zip(bit_chunks.iter()).for_each(
                |(group_index_chunk, mask)| {
                    // index_mask has value 1 << i in the loop
                    let mut index_mask = 1;
                    group_index_chunk.iter().for_each(|&group_index| {
                        // valid bit was set, real vale
                        let is_valid = (mask & index_mask) != 0;
                        if is_valid {
                            index_fn(group_index);
                        }
                        index_mask <<= 1;
                    })
                },
            );

            // handle any remaining bits (after the initial 64)
            let remainder_bits = bit_chunks.remainder_bits();
            group_indices_remainder
                .iter()
                .enumerate()
                .for_each(|(i, &group_index)| {
                    let is_valid = remainder_bits & (1 << i) != 0;
                    if is_valid {
                        index_fn(group_index)
                    }
                });
        }

        (Some(valids), Some(filter)) => {
            debug_assert_eq!(filter.len(), group_indices.len());
            debug_assert_eq!(valids.len(), group_indices.len());

            let group_indices_chunks = group_indices.chunks_exact(64);
            let valid_bit_chunks = valids.inner().bit_chunks();
            let filter_bit_chunks = filter.values().bit_chunks();

            let group_indices_remainder = group_indices_chunks.remainder();

            group_indices_chunks
                .zip(valid_bit_chunks.iter())
                .zip(filter_bit_chunks.iter())
                .for_each(|((group_index_chunk, valid_mask), filter_mask)| {
                    // index_mask has value 1 << i in the loop
                    let mut index_mask = 1;
                    group_index_chunk.iter().for_each(|&group_index| {
                        // valid bit was set, real vale
                        let is_valid = (valid_mask & filter_mask & index_mask) != 0;
                        if is_valid {
                            index_fn(group_index);
                        }
                        index_mask <<= 1;
                    })
                });

            // handle any remaining bits (after the initial 64)
            let remainder_valid_bits = valid_bit_chunks.remainder_bits();
            let remainder_filter_bits = filter_bit_chunks.remainder_bits();
            group_indices_remainder
                .iter()
                .enumerate()
                .for_each(|(i, &group_index)| {
                    let is_valid =
                        remainder_valid_bits & remainder_filter_bits & (1 << i) != 0;
                    if is_valid {
                        index_fn(group_index)
                    }
                });
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use arrow::array::{Int32Array, UInt32Array};
    use rand::{rngs::ThreadRng, Rng};
    use std::{cmp, collections::HashSet};

    #[test]
    fn accumulate() {
        let group_indices = (0..100).collect();
        let values = (0..100).map(|i| (i + 1) * 10).collect();
        let values_with_nulls = (0..100)
            .map(|i| if i % 3 == 0 { None } else { Some((i + 1) * 10) })
            .collect();

        // default to every fifth value being false, every even
        // being null
        let filter: BooleanArray = (0..100)
            .map(|i| {
                let is_even = i % 2 == 0;
                let is_fifth = i % 5 == 0;
                if is_even {
                    None
                } else if is_fifth {
                    Some(false)
                } else {
                    Some(true)
                }
            })
            .collect();

        // Test flat style
        Fixture {
            group_indices,
            values,
            values_with_nulls,
            filter,
            block_size: 3,
            acc_rounds: 5,
        }
        .run()
    }

    #[test]
    fn accumulate_fuzz() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            Fixture::new_random(&mut rng).run();
        }
    }

    /// Values for testing (there are enough values to exercise the 64 bit chunks
    struct Fixture {
        /// 100..0
        group_indices: Vec<usize>,

        /// 10, 20, ... 1010
        values: Vec<u32>,

        /// same as values, but every third is null:
        /// None, Some(20), Some(30), None ...
        values_with_nulls: Vec<Option<u32>>,

        /// filter (defaults to None)
        filter: BooleanArray,

        /// block size for testing [`BlockedNullState`]
        block_size: usize,

        acc_rounds: usize,
    }

    impl Fixture {
        fn new_random(rng: &mut ThreadRng) -> Self {
            // Number of input values in a batch
            let num_values: usize = rng.gen_range(1..200);
            // number of distinct groups
            let num_groups: usize = rng.gen_range(2..1000);
            let max_group = num_groups - 1;

            let group_indices: Vec<usize> = (0..num_values)
                .map(|_| rng.gen_range(0..max_group))
                .collect();

            let values: Vec<u32> = (0..num_values).map(|_| rng.gen()).collect();

            // random block size
            let block_size = rng.gen_range(1..num_groups);

            // random acc rounds
            let acc_rounds = rng.gen_range(1..=group_indices.len());

            // 10% chance of false
            // 10% change of null
            // 80% chance of true
            let filter: BooleanArray = (0..num_values)
                .map(|_| {
                    let filter_value = rng.gen_range(0.0..1.0);
                    if filter_value < 0.1 {
                        Some(false)
                    } else if filter_value < 0.2 {
                        None
                    } else {
                        Some(true)
                    }
                })
                .collect();

            // random values with random number and location of nulls
            // random null percentage
            let null_pct: f32 = rng.gen_range(0.0..1.0);
            let values_with_nulls: Vec<Option<u32>> = (0..num_values)
                .map(|_| {
                    let is_null = null_pct < rng.gen_range(0.0..1.0);
                    if is_null {
                        None
                    } else {
                        Some(rng.gen())
                    }
                })
                .collect();

            Self {
                group_indices,
                values,
                values_with_nulls,
                filter,
                block_size,
                acc_rounds,
            }
        }

        /// returns `Self::values` an Array
        fn values_array(&self) -> UInt32Array {
            UInt32Array::from(self.values.clone())
        }

        /// returns `Self::values_with_nulls` as an Array
        fn values_with_nulls_array(&self) -> UInt32Array {
            UInt32Array::from(self.values_with_nulls.clone())
        }

        /// Calls `NullState::accumulate` and `accumulate_indices`
        /// with all combinations of nulls and filter values
        fn run(&self) {
            let total_num_groups = *self.group_indices.iter().max().unwrap() + 1;

            let group_indices = &self.group_indices;
            let values_array = self.values_array();
            let values_with_nulls_array = self.values_with_nulls_array();
            let filter = &self.filter;

            // no null, no filters
            Self::accumulate_test(
                group_indices,
                &values_array,
                None,
                total_num_groups,
                self.block_size,
                self.acc_rounds,
            );

            // nulls, no filters
            Self::accumulate_test(
                group_indices,
                &values_with_nulls_array,
                None,
                total_num_groups,
                self.block_size,
                self.acc_rounds,
            );

            // no nulls, filters
            Self::accumulate_test(
                group_indices,
                &values_array,
                Some(filter),
                total_num_groups,
                self.block_size,
                self.acc_rounds,
            );

            // nulls, filters
            Self::accumulate_test(
                group_indices,
                &values_with_nulls_array,
                Some(filter),
                total_num_groups,
                self.block_size,
                self.acc_rounds,
            );
        }

        /// Calls `NullState::accumulate` and `accumulate_indices` to
        /// ensure it generates the correct values.
        ///
        fn accumulate_test(
            group_indices: &[usize],
            values: &UInt32Array,
            opt_filter: Option<&BooleanArray>,
            total_num_groups: usize,
            block_size: usize,
            acc_rounds: usize,
        ) {
            // Test `accumulate` of `FlatNullState` + accumulate in once
            Self::accumulate_values_test(
                group_indices,
                values,
                opt_filter,
                total_num_groups,
                None,
                None,
            );

            // Test `accumulate` of `FlatNullState` + accumulate in multiple times
            Self::accumulate_values_test(
                group_indices,
                values,
                opt_filter,
                total_num_groups,
                None,
                Some(acc_rounds),
            );

            // Test `accumulate` of `BlockedNullState` + accumulate in once
            Self::accumulate_values_test(
                group_indices,
                values,
                opt_filter,
                total_num_groups,
                Some(block_size),
                None,
            );

            // Test `accumulate` of `BlockedNullState` + accumulate in multiple times
            Self::accumulate_values_test(
                group_indices,
                values,
                opt_filter,
                total_num_groups,
                Some(block_size),
                Some(acc_rounds),
            );

            // Convert values into a boolean array (anything above the
            // average is true, otherwise false)
            let avg: usize = values.iter().filter_map(|v| v.map(|v| v as usize)).sum();
            let boolean_values: BooleanArray =
                values.iter().map(|v| v.map(|v| v as usize > avg)).collect();

            // Test `accumulate_boolean` of `FlatNullState` + accumulate in once
            Self::accumulate_boolean_test(
                group_indices,
                &boolean_values,
                opt_filter,
                total_num_groups,
                None,
                None,
            );

            // Test `accumulate_boolean` of `FlatNullState` + accumulate in multiple times
            Self::accumulate_boolean_test(
                group_indices,
                &boolean_values,
                opt_filter,
                total_num_groups,
                None,
                Some(acc_rounds),
            );

            // Test `accumulate_boolean` of `BlockedNullState` + accumulate in once
            Self::accumulate_boolean_test(
                group_indices,
                &boolean_values,
                opt_filter,
                total_num_groups,
                Some(block_size),
                None,
            );

            // Test `accumulate_boolean` of `BlockedNullState` + accumulate in multiple times
            Self::accumulate_boolean_test(
                group_indices,
                &boolean_values,
                opt_filter,
                total_num_groups,
                Some(block_size),
                Some(acc_rounds),
            );

            // Test `accumulate_indices`
            Self::accumulate_indices_test(group_indices, values.nulls(), opt_filter);
        }

        /// This is effectively a different implementation of
        /// accumulate that we compare with the above implementation
        fn accumulate_values_test(
            group_indices: &[usize],
            values: &UInt32Array,
            opt_filter: Option<&BooleanArray>,
            total_num_groups: usize,
            block_size: Option<usize>,
            acc_rounds: Option<usize>,
        ) {
            // Chunking `group_indices`, `values`, `opt_filter`, and we also need to generate
            // `chunked acc_group_indices` basing on `group_indices`
            let (group_indices_chunks, values_chunks, opt_filter_chunks) =
                if let Some(rounds) = acc_rounds {
                    let chunk_size = group_indices.len() / rounds;

                    let group_indices_chunks = group_indices
                        .chunks(chunk_size)
                        .map(|chunk| chunk.to_vec())
                        .collect::<Vec<_>>();

                    let values_chunks = values
                        .iter()
                        .collect::<Vec<_>>()
                        .chunks(chunk_size)
                        .map(|chunk| UInt32Array::from_iter(chunk.iter().copied()))
                        .collect::<Vec<_>>();

                    let opt_filter_chunks = if let Some(filter) = opt_filter {
                        filter
                            .iter()
                            .collect::<Vec<_>>()
                            .chunks(chunk_size)
                            .map(|chunk| Some(BooleanArray::from_iter(chunk.iter())))
                            .collect::<Vec<_>>()
                    } else {
                        vec![None; values_chunks.len()]
                    };

                    (group_indices_chunks, values_chunks, opt_filter_chunks)
                } else {
                    (
                        vec![group_indices.to_vec()],
                        vec![values.clone()],
                        vec![opt_filter.cloned()],
                    )
                };

            let mut total_num_groups_chunks = vec![];
            let mut cur_total_num_groups = usize::MIN;
            for group_indices in &group_indices_chunks {
                let num_groups = *group_indices.iter().max().unwrap() + 1;
                cur_total_num_groups = cmp::max(cur_total_num_groups, num_groups);
                total_num_groups_chunks.push(cur_total_num_groups);
            }

            // Build needed test contexts
            let (mut null_state, block_size, acc_group_indices_chunks) =
                if let Some(blk_size) = block_size {
                    let mut acc_group_indices_chunks = vec![];
                    for group_indices in group_indices_chunks {
                        let acc_group_indices = group_indices
                            .into_iter()
                            .map(|index| {
                                let block_id = (index / blk_size) as u32;
                                let block_offset = (index % blk_size) as u64;
                                BlockedGroupIndexOperations::pack_index(
                                    block_id,
                                    block_offset,
                                ) as usize
                            })
                            .collect::<Vec<_>>();
                        acc_group_indices_chunks.push(acc_group_indices);
                    }

                    (
                        NullStateAdapter::new(Some(blk_size)),
                        blk_size,
                        acc_group_indices_chunks,
                    )
                } else {
                    (NullStateAdapter::new(None), 0, group_indices_chunks)
                };

            // Start the test
            let mut accumulated_values = vec![];
            for (((acc_group_indices, values), total_num_groups), cur_opt_filter) in
                acc_group_indices_chunks
                    .into_iter()
                    .zip(values_chunks)
                    .zip(total_num_groups_chunks)
                    .zip(opt_filter_chunks)
            {
                null_state.accumulate(
                    &acc_group_indices,
                    &values,
                    cur_opt_filter.as_ref(),
                    total_num_groups,
                    |block_id, block_offset, value| {
                        let flatten_index = ((block_id as u64 * block_size as u64)
                            + block_offset)
                            as usize;
                        accumulated_values.push((flatten_index, value));
                    },
                );
            }

            // Figure out the expected values
            let mut expected_values = vec![];
            let mut mock = MockNullState::new();

            match opt_filter {
                None => group_indices.iter().zip(values.iter()).for_each(
                    |(&group_index, value)| {
                        if let Some(value) = value {
                            mock.saw_value(group_index);
                            expected_values.push((group_index, value));
                        }
                    },
                ),
                Some(filter) => {
                    group_indices
                        .iter()
                        .zip(values.iter())
                        .zip(filter.iter())
                        .for_each(|((&group_index, value), is_included)| {
                            // if value passed filter
                            if let Some(true) = is_included {
                                if let Some(value) = value {
                                    mock.saw_value(group_index);
                                    expected_values.push((group_index, value));
                                }
                            }
                        });
                }
            }

            assert_eq!(accumulated_values, expected_values,
                       "\n\naccumulated_values:{accumulated_values:#?}\n\nexpected_values:{expected_values:#?}");
            let seen_values = null_state.build_cloned_seen_values();
            mock.validate_seen_values(&seen_values);

            // Validate the final buffer (one value per group)
            let expected_null_buffer = mock.expected_null_buffer(total_num_groups);

            let null_buffer = null_state.build_all_in_once();

            assert_eq!(null_buffer, expected_null_buffer);
        }

        // Calls `accumulate_indices`
        // and opt_filter and ensures it calls the right values
        fn accumulate_indices_test(
            group_indices: &[usize],
            nulls: Option<&NullBuffer>,
            opt_filter: Option<&BooleanArray>,
        ) {
            let mut accumulated_values = vec![];

            accumulate_indices(group_indices, nulls, opt_filter, |group_index| {
                accumulated_values.push(group_index);
            });

            // Figure out the expected values
            let mut expected_values = vec![];

            match (nulls, opt_filter) {
                (None, None) => group_indices.iter().for_each(|&group_index| {
                    expected_values.push(group_index);
                }),
                (Some(nulls), None) => group_indices.iter().zip(nulls.iter()).for_each(
                    |(&group_index, is_valid)| {
                        if is_valid {
                            expected_values.push(group_index);
                        }
                    },
                ),
                (None, Some(filter)) => group_indices.iter().zip(filter.iter()).for_each(
                    |(&group_index, is_included)| {
                        if let Some(true) = is_included {
                            expected_values.push(group_index);
                        }
                    },
                ),
                (Some(nulls), Some(filter)) => {
                    group_indices
                        .iter()
                        .zip(nulls.iter())
                        .zip(filter.iter())
                        .for_each(|((&group_index, is_valid), is_included)| {
                            // if value passed filter
                            if let (true, Some(true)) = (is_valid, is_included) {
                                expected_values.push(group_index);
                            }
                        });
                }
            }

            assert_eq!(accumulated_values, expected_values,
                       "\n\naccumulated_values:{accumulated_values:#?}\n\nexpected_values:{expected_values:#?}");
        }

        /// This is effectively a different implementation of
        /// accumulate_boolean that we compare with the above implementation
        fn accumulate_boolean_test(
            group_indices: &[usize],
            values: &BooleanArray,
            opt_filter: Option<&BooleanArray>,
            total_num_groups: usize,
            block_size: Option<usize>,
            acc_rounds: Option<usize>,
        ) {
            // Chunking `group_indices`, `values`, `opt_filter`, and we also need to generate
            // `chunked acc_group_indices` basing on `group_indices`
            let (group_indices_chunks, values_chunks, opt_filter_chunks) =
                if let Some(rounds) = acc_rounds {
                    let chunk_size = group_indices.len() / rounds;

                    let group_indices_chunks = group_indices
                        .chunks(chunk_size)
                        .map(|chunk| chunk.to_vec())
                        .collect::<Vec<_>>();

                    let values_chunks = values
                        .iter()
                        .collect::<Vec<_>>()
                        .chunks(chunk_size)
                        .map(|chunk| BooleanArray::from_iter(chunk.iter().copied()))
                        .collect::<Vec<_>>();

                    let opt_filter_chunks = if let Some(filter) = opt_filter {
                        filter
                            .iter()
                            .collect::<Vec<_>>()
                            .chunks(chunk_size)
                            .map(|chunk| Some(BooleanArray::from_iter(chunk.iter())))
                            .collect::<Vec<_>>()
                    } else {
                        vec![None; values_chunks.len()]
                    };

                    (group_indices_chunks, values_chunks, opt_filter_chunks)
                } else {
                    (
                        vec![group_indices.to_vec()],
                        vec![values.clone()],
                        vec![opt_filter.cloned()],
                    )
                };

            let mut total_num_groups_chunks = vec![];
            let mut cur_total_num_groups = usize::MIN;
            for group_indices in &group_indices_chunks {
                let num_groups = *group_indices.iter().max().unwrap() + 1;
                cur_total_num_groups = cmp::max(cur_total_num_groups, num_groups);
                total_num_groups_chunks.push(cur_total_num_groups);
            }

            // Build needed test contexts
            let (mut null_state, block_size, acc_group_indices_chunks) =
                if let Some(blk_size) = block_size {
                    let mut acc_group_indices_chunks = vec![];
                    for group_indices in group_indices_chunks {
                        let acc_group_indices = group_indices
                            .into_iter()
                            .map(|index| {
                                let block_id = (index / blk_size) as u32;
                                let block_offset = (index % blk_size) as u64;
                                BlockedGroupIndexOperations::pack_index(
                                    block_id,
                                    block_offset,
                                ) as usize
                            })
                            .collect::<Vec<_>>();
                        acc_group_indices_chunks.push(acc_group_indices);
                    }

                    (
                        NullStateAdapter::new(Some(blk_size)),
                        blk_size,
                        acc_group_indices_chunks,
                    )
                } else {
                    (NullStateAdapter::new(None), 0, group_indices_chunks)
                };

            // Start the test
            let mut accumulated_values = vec![];
            for (((acc_group_indices, values), total_num_groups), opt_filter) in
                acc_group_indices_chunks
                    .into_iter()
                    .zip(values_chunks)
                    .zip(total_num_groups_chunks)
                    .zip(opt_filter_chunks)
            {
                null_state.accumulate_boolean(
                    &acc_group_indices,
                    &values,
                    opt_filter.as_ref(),
                    total_num_groups,
                    |block_id, block_offset, value| {
                        let flatten_index = ((block_id as u64 * block_size as u64)
                            + block_offset)
                            as usize;
                        accumulated_values.push((flatten_index, value));
                    },
                );
            }

            // Figure out the expected values
            let mut expected_values = vec![];
            let mut mock = MockNullState::new();

            match opt_filter {
                None => group_indices.iter().zip(values.iter()).for_each(
                    |(&group_index, value)| {
                        if let Some(value) = value {
                            mock.saw_value(group_index);
                            expected_values.push((group_index, value));
                        }
                    },
                ),
                Some(filter) => {
                    group_indices
                        .iter()
                        .zip(values.iter())
                        .zip(filter.iter())
                        .for_each(|((&group_index, value), is_included)| {
                            // if value passed filter
                            if let Some(true) = is_included {
                                if let Some(value) = value {
                                    mock.saw_value(group_index);
                                    expected_values.push((group_index, value));
                                }
                            }
                        });
                }
            }

            assert_eq!(accumulated_values, expected_values,
                       "\n\naccumulated_values:{accumulated_values:#?}\n\nexpected_values:{expected_values:#?}");

            let seen_values = null_state.build_cloned_seen_values();
            mock.validate_seen_values(&seen_values);

            // Validate the final buffer (one value per group)
            let expected_null_buffer = mock.expected_null_buffer(total_num_groups);

            let null_buffer = null_state.build_all_in_once();

            assert_eq!(null_buffer, expected_null_buffer);
        }
    }

    /// Parallel implementation of NullState to check expected values
    #[derive(Debug, Default)]
    struct MockNullState {
        /// group indices that had values that passed the filter
        seen_values: HashSet<usize>,
    }

    impl MockNullState {
        fn new() -> Self {
            Default::default()
        }

        fn saw_value(&mut self, group_index: usize) {
            self.seen_values.insert(group_index);
        }

        /// did this group index see any input?
        fn expected_seen(&self, group_index: usize) -> bool {
            self.seen_values.contains(&group_index)
        }

        /// Validate that the seen_values matches self.seen_values
        fn validate_seen_values(&self, seen_values: &BooleanBuffer) {
            for (group_index, is_seen) in seen_values.iter().enumerate() {
                let expected_seen = self.expected_seen(group_index);
                assert_eq!(
                    expected_seen, is_seen,
                    "mismatch at for group {group_index}"
                );
            }
        }

        /// Create the expected null buffer based on if the input had nulls and a filter
        fn expected_null_buffer(&self, total_num_groups: usize) -> NullBuffer {
            (0..total_num_groups)
                .map(|group_index| self.expected_seen(group_index))
                .collect()
        }
    }

    #[test]
    fn test_accumulate_multiple_no_nulls_no_filter() {
        let group_indices = vec![0, 1, 0, 1];
        let values1 = Int32Array::from(vec![1, 2, 3, 4]);
        let values2 = Int32Array::from(vec![10, 20, 30, 40]);
        let value_columns = [values1, values2];

        let mut accumulated = vec![];
        accumulate_multiple(
            &group_indices,
            &value_columns.iter().collect::<Vec<_>>(),
            None,
            |group_idx, batch_idx, columns| {
                let values = columns.iter().map(|col| col.value(batch_idx)).collect();
                accumulated.push((group_idx, values));
            },
        );

        let expected = vec![
            (0, vec![1, 10]),
            (1, vec![2, 20]),
            (0, vec![3, 30]),
            (1, vec![4, 40]),
        ];
        assert_eq!(accumulated, expected);
    }

    #[test]
    fn test_accumulate_multiple_with_nulls() {
        let group_indices = vec![0, 1, 0, 1];
        let values1 = Int32Array::from(vec![Some(1), None, Some(3), Some(4)]);
        let values2 = Int32Array::from(vec![Some(10), Some(20), None, Some(40)]);
        let value_columns = [values1, values2];

        let mut accumulated = vec![];
        accumulate_multiple(
            &group_indices,
            &value_columns.iter().collect::<Vec<_>>(),
            None,
            |group_idx, batch_idx, columns| {
                let values = columns.iter().map(|col| col.value(batch_idx)).collect();
                accumulated.push((group_idx, values));
            },
        );

        // Only rows where both columns are non-null should be accumulated
        let expected = vec![(0, vec![1, 10]), (1, vec![4, 40])];
        assert_eq!(accumulated, expected);
    }

    #[test]
    fn test_accumulate_multiple_with_filter() {
        let group_indices = vec![0, 1, 0, 1];
        let values1 = Int32Array::from(vec![1, 2, 3, 4]);
        let values2 = Int32Array::from(vec![10, 20, 30, 40]);
        let value_columns = [values1, values2];

        let filter = BooleanArray::from(vec![true, false, true, false]);

        let mut accumulated = vec![];
        accumulate_multiple(
            &group_indices,
            &value_columns.iter().collect::<Vec<_>>(),
            Some(&filter),
            |group_idx, batch_idx, columns| {
                let values = columns.iter().map(|col| col.value(batch_idx)).collect();
                accumulated.push((group_idx, values));
            },
        );

        // Only rows where filter is true should be accumulated
        let expected = vec![(0, vec![1, 10]), (0, vec![3, 30])];
        assert_eq!(accumulated, expected);
    }

    #[test]
    fn test_accumulate_multiple_with_nulls_and_filter() {
        let group_indices = vec![0, 1, 0, 1];
        let values1 = Int32Array::from(vec![Some(1), None, Some(3), Some(4)]);
        let values2 = Int32Array::from(vec![Some(10), Some(20), None, Some(40)]);
        let value_columns = [values1, values2];

        let filter = BooleanArray::from(vec![true, true, true, false]);

        let mut accumulated = vec![];
        accumulate_multiple(
            &group_indices,
            &value_columns.iter().collect::<Vec<_>>(),
            Some(&filter),
            |group_idx, batch_idx, columns| {
                let values = columns.iter().map(|col| col.value(batch_idx)).collect();
                accumulated.push((group_idx, values));
            },
        );

        // Only rows where both:
        // 1. Filter is true
        // 2. Both columns are non-null
        // should be accumulated
        let expected = [(0, vec![1, 10])];
        assert_eq!(accumulated, expected);
    }
}
