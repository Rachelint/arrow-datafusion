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

//! Aggregation intermediate results blocks in blocked approach

use std::{
    collections::VecDeque,
    fmt::Debug,
    iter, mem,
    ops::{Index, IndexMut},
    usize,
};

use datafusion_expr_common::groups_accumulator::EmitTo;

// ========================================================================
// Basic abstractions: `Blocks` and `Block`
// ========================================================================

/// Structure used to store aggregation intermediate results in `blocked approach`
///
/// Aggregation intermediate results will be stored as multiple [`Block`]s
/// (simply you can think a [`Block`] as a `Vec`). And `Blocks` is the structure
/// to represent such multiple [`Block`]s.
///
/// More details about `blocked approach` can see in: [`GroupsAccumulator::supports_blocked_groups`].
///
/// [`GroupsAccumulator::supports_blocked_groups`]: datafusion_expr_common::groups_accumulator::GroupsAccumulator::supports_blocked_groups
///
#[derive(Debug)]
pub struct Blocks<B: Block, E: EmitBlockBuilder> {
    /// Data in blocks
    inner: Vec<B>,

    /// Block size
    ///
    /// It states:
    ///   - `Some(blk_size)`, it represents multiple block exists, each one
    ///     has the `blk_size` len.
    ///   - `None` , only single block exists.
    block_size: Option<usize>,

    /// Total groups number in blocks
    total_num_groups: usize,

    emit_state: EmitBlocksState<E>,
}

impl<B: Block, E: EmitBlockBuilder<B = B>> Blocks<B, E> {
    #[inline]
    pub fn new(block_size: Option<usize>) -> Self {
        Self {
            inner: Vec::new(),
            total_num_groups: 0,
            block_size,
            emit_state: EmitBlocksState::Init,
        }
    }

    /// Expand blocks to make it large enough to store `total_num_groups` groups,
    /// and we fill the new allocated block with `default_val`
    pub fn expand(&mut self, total_num_groups: usize, default_val: B::T) {
        assert!(!self.is_emitting(), "can not update groups during emitting");
        if self.total_num_groups >= total_num_groups {
            return;
        }

        // We compute how many blocks we need to store the `total_num_groups` groups.
        // And if found the `exist_blocks` are not enough, we allocate more.
        let needed_blocks =
            total_num_groups.div_ceil(self.block_size.unwrap_or(usize::MAX));
        let exist_blocks = self.inner.len();
        if exist_blocks < needed_blocks {
            let allocated_blocks = needed_blocks - exist_blocks;
            self.inner.extend(
                iter::repeat_with(|| {
                    let build_ctx = self.block_size.map(|blk_size| {
                        BuildBlockContext::new(blk_size, default_val.clone())
                    });
                    B::build(build_ctx)
                })
                .take(allocated_blocks),
            );
        }

        // If in `blocked approach`, we can return now.
        // But In `flat approach`, we keep only `single block`, if found the
        // `single block` not large enough, we allocate a larger one and copy
        // the exist data to it(such copy is really expansive).
        if self.block_size.is_none() {
            let single_block = self.inner.last_mut().unwrap();
            single_block.expand(total_num_groups, default_val.clone());
        }

        self.total_num_groups = total_num_groups;
    }

    /// Push block
    pub fn push_block(&mut self, block: B) {
        assert!(!self.is_emitting(), "can not update groups during emitting");
        let block_len = block.len();
        self.inner.push(block);
        self.total_num_groups += block_len;
    }

    /// Emit block
    ///
    /// Because we don't know few about how to init the[`EmitBlockBuilder`],
    /// so we expose `init_block_builder` to let the caller define it.
    pub fn emit_block<F>(&mut self, mut init_block_builder: F) -> Option<B>
    where
        F: FnMut(&mut Vec<B>) -> E,
    {
        let (total_num_groups, block_size) = if !self.is_emitting() {
            (self.total_num_groups, self.block_size.unwrap_or(usize::MAX))
        } else {
            (0, 0)
        };

        let emit_block = self
            .emit_state
            .emit_block(total_num_groups, block_size, || {
                init_block_builder(&mut self.inner)
            });

        self.total_num_groups -= emit_block.len();

        Some(emit_block)
    }

    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn total_num_groups(&self) -> usize {
        self.total_num_groups
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &B> {
        self.inner.iter()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear();
        self.total_num_groups = 0;
    }

    #[inline]
    fn is_emitting(&self) -> bool {
        self.emit_state.is_emitting()
    }
}

impl<B: Block, E: EmitBlockBuilder> Index<usize> for Blocks<B, E> {
    type Output = B;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.inner[index]
    }
}

impl<B: Block, E: EmitBlockBuilder> IndexMut<usize> for Blocks<B, E> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.inner[index]
    }
}

/// The abstraction to represent one aggregation intermediate result block
/// in `blocked approach`, multiple blocks compose a [`Blocks`]
///
/// Many types of aggregation intermediate result exist, and we define an interface
/// to abstract the necessary behaviors of various intermediate result types.
///
pub trait Block: Debug + Default {
    type T: Clone;

    /// How to build the block
    fn build(build_ctx: Option<BuildBlockContext<Self::T>>) -> Self;

    /// Expand the block to `new_len` with `default_val`
    ///
    /// In `flat approach`, we will only keep single block, and need to
    /// expand it when it is not large enough.
    fn expand(&mut self, new_len: usize, default_val: Self::T);

    /// Truncate the block to `new_len`
    fn truncate(&mut self, new_len: usize);

    /// Block len
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct BuildBlockContext<T> {
    block_size: usize,
    default_val: T,
}

impl<T> BuildBlockContext<T> {
    pub fn new(block_size: usize, default_val: T) -> Self {
        Self {
            block_size,
            default_val,
        }
    }
}

// ========================================================================
// The common blocks emitting logic
// ========================================================================

/// Emit blocks state
///
/// There are two states:
///   - Init, we can only update groups of [`BlockedNullState`]
///     in this state
///
///   - Emitting, we can't update groups in this state until all
///     blocks are emitted, and the state is reset to `Init`
///
#[derive(Debug)]
pub enum EmitBlocksState<E: EmitBlockBuilder> {
    Init,
    Emitting(EmitBlocksContext<E>),
}

/// Emit blocks context
#[derive(Debug)]
pub struct EmitBlocksContext<E: EmitBlockBuilder> {
    /// Index of next emitted block
    pub next_emit_index: usize,

    pub block_size: usize,

    /// Number of blocks needed to emit
    pub num_blocks: usize,

    /// The len of last block
    ///
    /// Due to the last block is possibly non-full, so we compute
    /// and store its len.
    ///
    pub last_block_len: usize,

    /// Extension context
    pub block_builder: E,
}

impl<E: EmitBlockBuilder> EmitBlocksState<E> {
    pub fn emit_block<F>(
        &mut self,
        total_num_groups: usize,
        block_size: usize,
        mut init_block_builder: F,
    ) -> E::B
    where
        F: FnMut() -> E,
    {
        let emit_block = loop {
            match self {
                Self::Init => {
                    // Init needed contexts
                    let num_blocks = total_num_groups.div_ceil(block_size);
                    let mut last_block_len = total_num_groups % block_size;
                    last_block_len = if last_block_len > 0 {
                        last_block_len
                    } else {
                        block_size
                    };

                    let block_builder = init_block_builder();

                    let emit_ctx = EmitBlocksContext {
                        next_emit_index: 0,
                        block_size,
                        num_blocks,
                        last_block_len,
                        block_builder,
                    };

                    *self = Self::Emitting(emit_ctx);
                }

                Self::Emitting(EmitBlocksContext {
                    next_emit_index,
                    block_size,
                    num_blocks,
                    last_block_len,
                    block_builder,
                }) => {
                    // Found empty blocks, return and reset directly
                    if next_emit_index == num_blocks {
                        let emit_block = block_builder.build(0, *block_size, true, 0);
                        *self = Self::Init;
                        break emit_block;
                    }

                    // Get current emit block idx
                    let emit_index = *next_emit_index;
                    // And then we advance the block idx
                    *next_emit_index += 1;

                    // Process and generate the emit block
                    let is_last_block = next_emit_index == num_blocks;
                    let emit_block = block_builder.build(
                        emit_index,
                        *block_size,
                        is_last_block,
                        *last_block_len,
                    );

                    // Finally we check if all blocks emitted, if so, we reset the
                    // emit context to allow new updates
                    if next_emit_index == num_blocks {
                        *self = Self::Init;
                    }

                    break emit_block;
                }
            }
        };

        emit_block
    }

    #[inline]
    pub fn is_emitting(&self) -> bool {
        !matches!(self, Self::Init)
    }
}

pub trait EmitBlockBuilder: Debug {
    type B;

    fn build(
        &mut self,
        emit_index: usize,
        block_size: usize,
        is_last_block: bool,
        last_block_len: usize,
    ) -> Self::B;
}

// ========================================================================
// The most commonly used implementation `GeneralBlocks<T>`
// ========================================================================

/// Usually we use `Vec` to represent `Block`, so we define `Blocks<Vec<T>>`
/// as the `GeneralBlocks<T>`
pub type GeneralBlocks<T> = Blocks<Vec<T>, Vec<Vec<T>>>;

/// As mentioned in [`GeneralBlocks`], we usually use `Vec` to represent `Block`,
/// so we implement `Block` trait for `Vec`
impl<Ty: Clone + Debug> Block for Vec<Ty> {
    type T = Ty;

    fn build(build_ctx: Option<BuildBlockContext<Self::T>>) -> Self {
        if let Some(BuildBlockContext {
            block_size,
            default_val,
        }) = build_ctx
        {
            vec![default_val; block_size]
        } else {
            Vec::new()
        }
    }

    #[inline]
    fn expand(&mut self, new_len: usize, default_val: Self::T) {
        self.resize(new_len, default_val);
    }

    #[inline]
    fn truncate(&mut self, new_len: usize) {
        self.truncate(new_len);
    }

    #[inline]
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T: Clone + Debug> GeneralBlocks<T> {
    pub fn emit(&mut self, emit_to: EmitTo) -> Vec<T> {
        let init_block_builder = |inner: &mut Vec<Vec<T>>| mem::take(inner);

        if matches!(emit_to, EmitTo::NextBlock) {
            assert!(
                self.block_size.is_some(),
                "only support emit next block in blocked groups"
            );
            self.emit_block(init_block_builder)
                .expect("should not call emit for empty blocks")
        } else {
            // TODO: maybe remove `EmitTo::take_needed` and move the
            // pattern matching codes here after supporting blocked approach
            // for all exist accumulators, to avoid matching twice
            assert!(
                self.block_size.is_none(),
                "only support emit all/first in flat groups"
            );

            // We perform single block emitting through steps:
            //   - Pop the `block` firstly
            //   - Take `need rows` from `block`
            //   - Push back the `block` if still some rows in it
            let mut block = self
                .emit_block(init_block_builder)
                .expect("should not call emit for empty blocks");

            let emit_block = emit_to.take_needed(&mut block);

            if !block.is_empty() {
                self.push_block(block);
            }

            emit_block
        }
    }
}

impl<T: Debug> EmitBlockBuilder for Vec<Vec<T>> {
    type B = Vec<T>;

    fn build(
        &mut self,
        emit_index: usize,
        _block_size: usize,
        is_last_block: bool,
        last_block_len: usize,
    ) -> Self::B {
        let mut emit_block = mem::take(&mut self[emit_index]);
        if is_last_block {
            emit_block.truncate(last_block_len);
        }
        emit_block
    }
}

#[cfg(test)]
mod test {
    use super::EmitBlockBuilder;
    use crate::aggregate::groups_accumulator::blocks::Blocks;

    // type TestBlocks = Blocks<Vec<u32>>;

    // #[test]
    // fn test() {
    //     let mut a = vec![vec!["".to_string()]];
    //     a.build(0, 0, true, 0);
    // }
    // #[test]
    // fn test_single_block_resize() {
    //     let new_block = |block_size: Option<usize>| {
    //         let cap = block_size.unwrap_or(0);
    //         Vec::with_capacity(cap)
    //     };

    //     let mut blocks = TestBlocks::new(None);
    //     assert_eq!(blocks.len(), 0);

    //     for _ in 0..2 {
    //         // Should have single block, 5 block len, all data are 42
    //         blocks.resize(5, new_block, 42);
    //         assert_eq!(blocks.len(), 1);
    //         assert_eq!(blocks[0].len(), 5);
    //         blocks[0].iter().for_each(|num| assert_eq!(*num, 42));

    //         // Resize to a larger block
    //         // Should still have single block, 10 block len, all data are 42
    //         blocks.resize(10, new_block, 42);
    //         assert_eq!(blocks.len(), 1);
    //         assert_eq!(blocks[0].len(), 10);
    //         blocks[0].iter().for_each(|num| assert_eq!(*num, 42));

    //         // Clear
    //         // Should have nothing after clearing
    //         blocks.clear();
    //         assert_eq!(blocks.len(), 0);

    //         // Test resize after clear in next round
    //     }
    // }

    // #[test]
    // fn test_multi_blocks_resize() {
    //     let new_block = |block_size: Option<usize>| {
    //         let cap = block_size.unwrap_or(0);
    //         Vec::with_capacity(cap)
    //     };

    //     let mut blocks = TestBlocks::new(Some(3));
    //     assert_eq!(blocks.len(), 0);

    //     for _ in 0..2 {
    //         // Should have:
    //         //  - 2 blocks
    //         //  - `block 0` of 3 len
    //         //  - `block 1` of 2 len
    //         //  - all data are 42
    //         blocks.resize(5, new_block, 42);
    //         assert_eq!(blocks.len(), 2);
    //         assert_eq!(blocks[0].len(), 3);
    //         blocks[0].iter().for_each(|num| assert_eq!(*num, 42));
    //         assert_eq!(blocks[1].len(), 2);
    //         blocks[1].iter().for_each(|num| assert_eq!(*num, 42));

    //         // Resize to larger blocks
    //         // Should have:
    //         //  - 4 blocks
    //         //  - `block 0` of 3 len
    //         //  - `block 1` of 3 len
    //         //  - `block 2` of 3 len
    //         //  - `block 3` of 1 len
    //         //  - all data are 42
    //         blocks.resize(10, new_block, 42);
    //         assert_eq!(blocks.len(), 4);
    //         assert_eq!(blocks[0].len(), 3);
    //         blocks[0].iter().for_each(|num| assert_eq!(*num, 42));
    //         assert_eq!(blocks[1].len(), 3);
    //         blocks[1].iter().for_each(|num| assert_eq!(*num, 42));
    //         assert_eq!(blocks[2].len(), 3);
    //         blocks[2].iter().for_each(|num| assert_eq!(*num, 42));
    //         assert_eq!(blocks[3].len(), 1);
    //         blocks[3].iter().for_each(|num| assert_eq!(*num, 42));

    //         // Clear
    //         // Should have nothing after clearing
    //         blocks.clear();
    //         assert_eq!(blocks.len(), 0);

    //         // Test resize after clear in next round
    //     }
    // }
}
