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

use std::env;

use crate::aggregates::group_values::GroupValues;
use crate::aggregates::AggregateMode;
use ahash::RandomState;
use arrow::compute::cast;
use arrow::record_batch::RecordBatch;
use arrow::row::{RowConverter, Rows, SortField};
use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, SchemaRef};
use datafusion_common::hash_utils::create_hashes;
use datafusion_common::utils::proxy::{HashTableLike, PartitionedHashTable};
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::memory_pool::proxy::{RawTableAllocExt, VecAllocExt};
use datafusion_expr::EmitTo;
use hashbrown::raw::RawTable;

/// A [`GroupValues`] making use of [`Rows`]
pub struct GroupValuesRows {
    /// The output schema
    schema: SchemaRef,

    /// Converter for the group values
    row_converter: RowConverter,

    /// Logically maps group values to a group_index in
    /// [`Self::group_values`] and in each accumulator
    ///
    /// Uses the raw API of hashbrown to avoid actually storing the
    /// keys (group values) in the table
    ///
    /// keys: u64 hashes of the GroupValue
    /// values: (hash, group_index)
    map: HashTableLike<(u64, usize)>,

    /// The size of `map` in bytes
    map_size: usize,

    /// The actual group by values, stored in arrow [`Row`] format.
    /// `group_values[i]` holds the group value for group_index `i`.
    ///
    /// The row format is used to compare group keys quickly and store
    /// them efficiently in memory. Quick comparison is especially
    /// important for multi-column group keys.
    ///
    /// [`Row`]: arrow::row::Row
    group_values: Option<Rows>,

    /// reused buffer to store hashes
    hashes_buffer: Vec<u64>,

    /// index buffer
    indexes_buffer: Vec<Vec<usize>>,

    /// reused buffer to store rows
    rows_buffer: Rows,

    /// Random state for creating hashes
    random_state: RandomState,
}

impl GroupValuesRows {
    pub fn try_new(schema: SchemaRef, agg_mode: AggregateMode) -> Result<Self> {
        let row_converter = RowConverter::new(
            schema
                .fields()
                .iter()
                .map(|f| SortField::new(f.data_type().clone()))
                .collect(),
        )?;

        let map = match agg_mode {
            AggregateMode::Partial => HashTableLike::Normal(RawTable::with_capacity(0)),
            AggregateMode::Final |
            AggregateMode::FinalPartitioned |
            AggregateMode::Single |
            AggregateMode::SinglePartitioned => {
                let num_parts = env::var("TEST_NUM_PARTS").unwrap_or("16".to_string());
                let num_parts = num_parts.parse::<usize>().unwrap();
                HashTableLike::Partitioned(PartitionedHashTable::new(num_parts))
            },
        };

        let starting_rows_capacity = 1000;
        let starting_data_capacity = 64 * starting_rows_capacity;
        let rows_buffer =
            row_converter.empty_rows(starting_rows_capacity, starting_data_capacity);
        Ok(Self {
            schema,
            row_converter,
            map,
            map_size: 0,
            group_values: None,
            hashes_buffer: Default::default(),
            indexes_buffer: Default::default(),
            rows_buffer,
            random_state: Default::default(),
        })
    }
}

impl GroupValues for GroupValuesRows {
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        // Convert the group keys into the row format
        let group_rows = &mut self.rows_buffer;
        group_rows.clear();
        self.row_converter.append(group_rows, cols)?;
        let n_rows = group_rows.num_rows();

        let mut group_values = match self.group_values.take() {
            Some(group_values) => group_values,
            None => self.row_converter.empty_rows(0, 0),
        };

        // tracks to which group each of the input rows belongs
        groups.clear();

        // 1.1 Calculate the group keys for the group values
        let batch_hashes = &mut self.hashes_buffer;
        batch_hashes.clear();
        batch_hashes.resize(n_rows, 0);
        create_hashes(cols, &self.random_state, batch_hashes)?;

        let num_partitions = self.map.num_partitions();
        if num_partitions > 1 {
            if self.indexes_buffer.is_empty() {
                self.indexes_buffer.resize(num_partitions, Vec::new());
            }

            self.indexes_buffer.iter_mut().for_each(|b| b.clear());

            for (row, target_hash) in batch_hashes.iter().enumerate() {
                let partition_idx = *target_hash as usize & (num_partitions - 1);
                self.indexes_buffer[partition_idx].push(row);
            }

            for (part_idx, partition) in self.indexes_buffer.iter().enumerate() {
                for &row in partition.iter() {
                    let target_hash = batch_hashes[row];
                    let entry = self.map.get_mut(part_idx, target_hash, |(exist_hash, group_idx)| {
                        // Somewhat surprisingly, this closure can be called even if the
                        // hash doesn't match, so check the hash first with an integer
                        // comparison first avoid the more expensive comparison with
                        // group value. https://github.com/apache/datafusion/pull/11718
                        target_hash == *exist_hash
                            // verify that the group that we are inserting with hash is
                            // actually the same key value as the group in
                            // existing_idx  (aka group_values @ row)
                            && group_rows.row(row) == group_values.row(*group_idx)
                    });
    
                    let group_idx = match entry {
                        // Existing group_index for this group value
                        Some((_hash, group_idx)) => *group_idx,
                        //  1.2 Need to create new entry for the group
                        None => {
                            // Add new entry to aggr_state and save newly created index
                            let group_idx = group_values.num_rows();
                            group_values.push(group_rows.row(row));
    
                            // for hasher function, use precomputed hash value
                            self.map.insert_accounted(
                                part_idx,
                                (target_hash, group_idx),
                                |(hash, _group_index)| *hash,
                                &mut self.map_size,
                            );
                            group_idx
                        }
                    };
                    groups.push(group_idx);
                }
            }
        } else {
            for (row, &target_hash) in batch_hashes.iter().enumerate() {
                let entry = self.map.get_mut(0, target_hash, |(exist_hash, group_idx)| {
                    // Somewhat surprisingly, this closure can be called even if the
                    // hash doesn't match, so check the hash first with an integer
                    // comparison first avoid the more expensive comparison with
                    // group value. https://github.com/apache/datafusion/pull/11718
                    target_hash == *exist_hash
                        // verify that the group that we are inserting with hash is
                        // actually the same key value as the group in
                        // existing_idx  (aka group_values @ row)
                        && group_rows.row(row) == group_values.row(*group_idx)
                });

                let group_idx = match entry {
                    // Existing group_index for this group value
                    Some((_hash, group_idx)) => *group_idx,
                    //  1.2 Need to create new entry for the group
                    None => {
                        // Add new entry to aggr_state and save newly created index
                        let group_idx = group_values.num_rows();
                        group_values.push(group_rows.row(row));

                        // for hasher function, use precomputed hash value
                        self.map.insert_accounted(
                            0,
                            (target_hash, group_idx),
                            |(hash, _group_index)| *hash,
                            &mut self.map_size,
                        );
                        group_idx
                    }
                };
                groups.push(group_idx);
            }
        }

        self.group_values = Some(group_values);

        Ok(())
    }

    fn size(&self) -> usize {
        let group_values_size = self.group_values.as_ref().map(|v| v.size()).unwrap_or(0);
        self.row_converter.size()
            + group_values_size
            + self.map_size
            + self.rows_buffer.size()
            + self.hashes_buffer.allocated_size()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        self.group_values
            .as_ref()
            .map(|group_values| group_values.num_rows())
            .unwrap_or(0)
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let mut group_values = self
            .group_values
            .take()
            .expect("Can not emit from empty rows");

        let mut output = match emit_to {
            EmitTo::All => {
                let output = self.row_converter.convert_rows(&group_values)?;
                group_values.clear();
                output
            }
            EmitTo::First(n) => {
                let groups_rows = group_values.iter().take(n);
                let output = self.row_converter.convert_rows(groups_rows)?;
                // Clear out first n group keys by copying them to a new Rows.
                // TODO file some ticket in arrow-rs to make this more efficient?
                let mut new_group_values = self.row_converter.empty_rows(0, 0);
                for row in group_values.iter().skip(n) {
                    new_group_values.push(row);
                }
                std::mem::swap(&mut new_group_values, &mut group_values);

                // SAFETY: self.map outlives iterator and is not modified concurrently
                unsafe {
                    for bucket in self.map.iter() {
                        // Decrement group index by n
                        // let (hash, group_idx) = bucket.as_ref();
                        // let hash = *hash;
                        // match group_idx.checked_sub(n) {
                        //     // Group index was >= n, shift value down
                        //     Some(sub) => bucket.as_mut().1 = sub,
                        //     // Group index was < n, so remove from table
                        //     None => self.map.erase(hash, bucket),
                        // }
                    }
                }
                output
            }
        };

        // TODO: Materialize dictionaries in group keys (#7647)
        for (field, array) in self.schema.fields.iter().zip(&mut output) {
            let expected = field.data_type();
            if let DataType::Dictionary(_, v) = expected {
                let actual = array.data_type();
                if v.as_ref() != actual {
                    return Err(DataFusionError::Internal(format!(
                        "Converted group rows expected dictionary of {v} got {actual}"
                    )));
                }
                *array = cast(array.as_ref(), expected)?;
            }
        }

        self.group_values = Some(group_values);
        Ok(output)
    }

    fn clear_shrink(&mut self, batch: &RecordBatch) {
        let count = batch.num_rows();
        self.group_values = self.group_values.take().map(|mut rows| {
            rows.clear();
            rows
        });

        self.map_size = self.map.clear_shrink(count);
        // self.map.clear();
        // self.map.shrink_to(count, |_| 0); // hasher does not matter since the map is cleared
        // self.map_size = self.map.capacity() * std::mem::size_of::<(u64, usize)>();
        self.hashes_buffer.clear();
        self.hashes_buffer.shrink_to(count);
    }
}
