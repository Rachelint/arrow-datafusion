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

//! [`VecAllocExt`] and [`RawTableAllocExt`] to help tracking of memory allocations

use hashbrown::raw::{Bucket, RawIter, RawTable};

/// Extension trait for [`Vec`] to account for allocations.
pub trait VecAllocExt {
    /// Item type.
    type T;

    /// [Push](Vec::push) new element to vector and increase
    /// `accounting` by any newly allocated bytes.
    ///
    /// Note that allocation counts  capacity, not size
    ///
    /// # Example:
    /// ```
    /// # use datafusion_common::utils::proxy::VecAllocExt;
    /// // use allocated to incrementally track how much memory is allocated in the vec
    /// let mut allocated = 0;
    /// let mut vec = Vec::new();
    /// // Push data into the vec and the accounting will be updated to reflect
    /// // memory allocation
    /// vec.push_accounted(1, &mut allocated);
    /// assert_eq!(allocated, 16); // space for 4 u32s
    /// vec.push_accounted(1, &mut allocated);
    /// assert_eq!(allocated, 16); // no new allocation needed
    ///
    /// // push more data into the vec
    /// for _ in 0..10 { vec.push_accounted(1, &mut allocated); }
    /// assert_eq!(allocated, 64); // underlying vec has space for 10 u32s
    /// assert_eq!(vec.allocated_size(), 64);
    /// ```
    /// # Example with other allocations:
    /// ```
    /// # use datafusion_common::utils::proxy::VecAllocExt;
    /// // You can use the same allocated size to track memory allocated by
    /// // another source. For example
    /// let mut allocated = 27;
    /// let mut vec = Vec::new();
    /// vec.push_accounted(1, &mut allocated); // allocates 16 bytes for vec
    /// assert_eq!(allocated, 43); // 16 bytes for vec, 27 bytes for other
    /// ```
    fn push_accounted(&mut self, x: Self::T, accounting: &mut usize);

    /// Return the amount of memory allocated by this Vec to store elements
    /// (`size_of<T> * capacity`).
    ///
    /// Note this calculation is not recursive, and does not include any heap
    /// allocations contained within the Vec's elements. Does not include the
    /// size of `self`
    ///
    /// # Example:
    /// ```
    /// # use datafusion_common::utils::proxy::VecAllocExt;
    /// let mut vec = Vec::new();
    /// // Push data into the vec and the accounting will be updated to reflect
    /// // memory allocation
    /// vec.push(1);
    /// assert_eq!(vec.allocated_size(), 16); // space for 4 u32s
    /// vec.push(1);
    /// assert_eq!(vec.allocated_size(), 16); // no new allocation needed
    ///
    /// // push more data into the vec
    /// for _ in 0..10 { vec.push(1); }
    /// assert_eq!(vec.allocated_size(), 64); // space for 64 now
    /// ```
    fn allocated_size(&self) -> usize;
}

impl<T> VecAllocExt for Vec<T> {
    type T = T;

    fn push_accounted(&mut self, x: Self::T, accounting: &mut usize) {
        let prev_capacty = self.capacity();
        self.push(x);
        let new_capacity = self.capacity();
        if new_capacity > prev_capacty {
            // capacity changed, so we allocated more
            let bump_size = (new_capacity - prev_capacty) * std::mem::size_of::<T>();
            // Note multiplication should never overflow because `push` would
            // have panic'd first, but the checked_add could potentially
            // overflow since accounting could be tracking additional values, and
            // could be greater than what is stored in the Vec
            *accounting = (*accounting).checked_add(bump_size).expect("overflow");
        }
    }
    fn allocated_size(&self) -> usize {
        std::mem::size_of::<T>() * self.capacity()
    }
}

/// Extension trait for hash browns [`RawTable`] to account for allocations.
pub trait RawTableAllocExt {
    /// Item type.
    type T;

    /// [Insert](RawTable::insert) new element into table and increase
    /// `accounting` by any newly allocated bytes.
    ///
    /// Returns the bucket where the element was inserted.
    /// Note that allocation counts capacity, not size.
    ///
    /// # Example:
    /// ```
    /// # use datafusion_common::utils::proxy::RawTableAllocExt;
    /// # use hashbrown::raw::RawTable;
    /// let mut table = RawTable::new();
    /// let mut allocated = 0;
    /// let hash_fn = |x: &u32| (*x as u64) % 1000;
    /// // pretend 0x3117 is the hash value for 1
    /// table.insert_accounted(1, hash_fn, &mut allocated);
    /// assert_eq!(allocated, 64);
    ///
    /// // insert more values
    /// for i in 0..100 { table.insert_accounted(i, hash_fn, &mut allocated); }
    /// assert_eq!(allocated, 400);
    /// ```
    fn insert_accounted(
        &mut self,
        partition_idx: usize,
        x: Self::T,
        hasher: impl Fn(&Self::T) -> u64,
        accounting: &mut usize,
    ) -> Bucket<Self::T>;
}

impl<T> RawTableAllocExt for RawTable<T> {
    type T = T;

    fn insert_accounted(
        &mut self,
        partition_idx: usize,
        x: Self::T,
        hasher: impl Fn(&Self::T) -> u64,
        accounting: &mut usize,
    ) -> Bucket<Self::T> {
        let hash = hasher(&x);

        match self.try_insert_no_grow(hash, x) {
            Ok(bucket) => bucket,
            Err(x) => {
                // need to request more memory

                let bump_elements = self.capacity().max(16);
                let bump_size = bump_elements * std::mem::size_of::<T>();
                *accounting = (*accounting).checked_add(bump_size).expect("overflow");

                self.reserve(bump_elements, hasher);

                // still need to insert the element since first try failed
                // Note: cannot use `.expect` here because `T` may not implement `Debug`
                match self.try_insert_no_grow(hash, x) {
                    Ok(bucket) => bucket,
                    Err(_) => panic!("just grew the container"),
                }
            }
        }
    }
}

impl<T> RawTableAllocExt for HashTableLike<T> {
    type T = T;

    fn insert_accounted(
        &mut self,
        partition_idx: usize,
        x: Self::T,
        hasher: impl Fn(&Self::T) -> u64,
        accounting: &mut usize,
    ) -> Bucket<Self::T> {
        let hash = hasher(&x);
        let map = match self {
            HashTableLike::Normal(n) => n,
            HashTableLike::Partitioned(p) => {
                let part = &mut p.partitions[partition_idx];
                part
            }
        };

        match map.try_insert_no_grow(hash, x) {
            Ok(bucket) => bucket,
            Err(x) => {
                // need to request more memory

                let bump_elements = map.capacity().max(16);
                let bump_size = bump_elements * std::mem::size_of::<T>();
                *accounting = (*accounting).checked_add(bump_size).expect("overflow");

                map.reserve(bump_elements, hasher);

                // still need to insert the element since first try failed
                // Note: cannot use `.expect` here because `T` may not implement `Debug`
                match map.try_insert_no_grow(hash, x) {
                    Ok(bucket) => bucket,
                    Err(_) => panic!("just grew the container"),
                }
            }
        }
    }
}

pub struct PartitionedHashTable<T> {
    partitions: Vec<RawTable<T>>,
}

impl<T> PartitionedHashTable<T> {
    pub fn new(num_parts: usize) -> Self {
        let mut partitions = Vec::with_capacity(num_parts);
        for _ in 0..num_parts {
            partitions.push(RawTable::with_capacity(0));
        }
        
        Self {
            partitions,
        }
    }

    fn get_mut(&mut self, partition_idx: usize, hash: u64, eq: impl FnMut(&T) -> bool) -> Option<&mut T> {
        let part = &mut self.partitions[partition_idx];
        part.get_mut(hash, eq)
    }

    fn iter(&self) -> PartitionedHashTableIter<'_, T> {
        let parts = self.partitions.iter();
        PartitionedHashTableIter {
            parts: Box::new(parts),
            state: PartitionedHashTableIterState::Init,
        }
    }

    fn erase(&mut self, hash: u64, bucket: Bucket<T>) {
        let num_partitions = self.partitions.len();
        let partition_idx = hash as usize % num_partitions;
        let part = &mut self.partitions[partition_idx];
        unsafe {
            part.erase(bucket);
        }
    }
}

struct PartitionedHashTableIter<'a, T> {
    parts: Box<dyn Iterator<Item = &'a RawTable<T>> + 'a>,
    state: PartitionedHashTableIterState<T>,
}

impl<'a, T> Iterator for PartitionedHashTableIter<'a, T> {
    type Item = Bucket<T>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match &mut self.state {
                PartitionedHashTableIterState::Init => {
                    let part_opt = self.parts.next();
                    match part_opt {
                        Some(part) => {
                            let iter = unsafe { part.iter() };
                            self.state = PartitionedHashTableIterState::Polling(iter);
                        }
                        None => {
                            self.state = PartitionedHashTableIterState::Finished;
                            return None;
                        }
                    }
                }
                PartitionedHashTableIterState::Polling(iter) => {
                    let bucket_opt = iter.next();
                    match bucket_opt {
                        Some(bucket) => {
                            return Some(bucket);
                        }
                        None => {
                            self.state = PartitionedHashTableIterState::Init;
                        }
                    }
                }
                PartitionedHashTableIterState::Finished => return None,
            }
        }
    }
}

enum PartitionedHashTableIterState<T> {
    Init,
    Polling(RawIter<T>),
    Finished,
}

pub enum HashTableLike<T> {
    Normal(RawTable<T>),
    Partitioned(PartitionedHashTable<T>),
}

impl<T> HashTableLike<T> {
    pub fn get_partitions(&mut self,  partition_idx:usize) -> &mut RawTable<T> {
        match self {
            HashTableLike::Normal(n) => n,
            HashTableLike::Partitioned(p) => &mut p.partitions[partition_idx],
        }
    }

    pub fn iter(&self) -> Box<dyn Iterator<Item = Bucket<T>> + '_> {
        match self {
            HashTableLike::Normal(n) => unsafe { Box::new(n.iter()) },
            HashTableLike::Partitioned(p) => Box::new(p.iter()),
        }
    }

    pub fn erase(&mut self, hash: u64, bucket: Bucket<T>) {
        match self {
            HashTableLike::Normal(n) => unsafe {
                n.erase(bucket);
            },
            HashTableLike::Partitioned(p) => {
                p.erase(hash, bucket);
            }
        }
    }

    pub fn clear_shrink(&mut self, shrink_size: usize) -> usize {
        match self {
            HashTableLike::Normal(n) => {
                n.clear();
                n.shrink_to(shrink_size, |_| 0); // hasher does not matter since the map is cleared
                n.capacity() * std::mem::size_of::<T>()
            }
            HashTableLike::Partitioned(_) => {
                let new_map = RawTable::with_capacity(shrink_size);
                let new_map_size = new_map.capacity() * std::mem::size_of::<T>();
                *self = HashTableLike::Normal(new_map);

                new_map_size
            }
        }
    }

    pub fn num_partitions(&self) -> usize {
        match self {
            HashTableLike::Normal(_) => 1,
            HashTableLike::Partitioned(p) => p.partitions.len(),
        }
    }
}
