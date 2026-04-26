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

//! Defines an execution plan that coalesces subsets of input partitions into a
//! smaller number of output partitions.

use std::sync::Arc;

use super::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use super::stream::{ObservedStream, RecordBatchReceiverStream};
use super::{
    DisplayAs, ExecutionPlanProperties, PlanProperties, SendableRecordBatchStream,
    Statistics,
};
use crate::execution_plan::{CardinalityEffect, EvaluationType, SchedulingType};
use crate::{DisplayFormatType, ExecutionPlan, Partitioning, check_if_same_properties};
use datafusion_common::{Result, assert_or_internal_err, internal_err};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::PhysicalExpr;

/// Coalesces groups of input partitions into fewer output partitions.
///
/// Output partition `p` consumes input partitions:
/// `p`, `p + output_partitions`, `p + 2 * output_partitions`, ...
#[derive(Debug, Clone)]
pub struct CoalescePartitionGroupsExec {
    /// Input execution plan
    input: Arc<dyn ExecutionPlan>,
    /// Number of output partitions produced by this exec
    output_partitions: usize,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// Cached plan properties
    cache: Arc<PlanProperties>,
}

impl CoalescePartitionGroupsExec {
    /// Creates a new [`CoalescePartitionGroupsExec`].
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        output_partitions: usize,
    ) -> Result<Self> {
        let input_partitions = input.output_partitioning().partition_count();
        assert_or_internal_err!(
            output_partitions > 0,
            "CoalescePartitionGroupsExec requires at least one output partition"
        );
        assert_or_internal_err!(
            input_partitions > 0,
            "CoalescePartitionGroupsExec requires at least one input partition"
        );
        assert_or_internal_err!(
            input_partitions % output_partitions == 0,
            "CoalescePartitionGroupsExec requires input partitions ({input_partitions}) to be divisible by output partitions ({output_partitions})"
        );

        let cache = Self::compute_properties(&input, output_partitions);
        Ok(Self {
            input,
            output_partitions,
            metrics: ExecutionPlanMetricsSet::new(),
            cache: Arc::new(cache),
        })
    }

    /// Input execution plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Number of output partitions
    pub fn output_partitions(&self) -> usize {
        self.output_partitions
    }

    /// Number of input partitions folded into each output partition.
    pub fn group_size(&self) -> usize {
        self.input.output_partitioning().partition_count() / self.output_partitions
    }

    fn with_new_children_and_same_properties(
        &self,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Self {
        Self {
            input: children.swap_remove(0),
            metrics: ExecutionPlanMetricsSet::new(),
            ..Self::clone(self)
        }
    }

    fn compute_properties(
        input: &Arc<dyn ExecutionPlan>,
        output_partitions: usize,
    ) -> PlanProperties {
        let input_partitions = input.output_partitioning().partition_count();
        let output_partitioning = match input.output_partitioning() {
            Partitioning::Hash(exprs, _) => {
                Partitioning::Hash(exprs.clone(), output_partitions)
            }
            _ => Partitioning::UnknownPartitioning(output_partitions),
        };

        let (drive, scheduling) = if input_partitions > output_partitions {
            (EvaluationType::Eager, SchedulingType::Cooperative)
        } else {
            (
                input.properties().evaluation_type,
                input.properties().scheduling_type,
            )
        };

        let mut eq_properties = input.equivalence_properties().clone();
        eq_properties.clear_orderings();
        if input_partitions > output_partitions {
            eq_properties.clear_per_partition_constants();
        }

        PlanProperties::new(
            eq_properties,
            output_partitioning,
            input.pipeline_behavior(),
            input.boundedness(),
        )
        .with_evaluation_type(drive)
        .with_scheduling_type(scheduling)
    }
}

impl DisplayAs for CoalescePartitionGroupsExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let input_partitions = self.input.output_partitioning().partition_count();
        let group_size = self.group_size();
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "CoalescePartitionGroupsExec: input_partitions={input_partitions}, output_partitions={}, group_size={group_size}",
                self.output_partitions,
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "partition_count(in->out)={input_partitions} -> {}", self.output_partitions)?;
                write!(f, "group_size={group_size}")
            }
        }
    }
}

impl ExecutionPlan for CoalescePartitionGroupsExec {
    fn name(&self) -> &'static str {
        "CoalescePartitionGroupsExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&dyn PhysicalExpr) -> Result<datafusion_common::tree_node::TreeNodeRecursion>,
    ) -> Result<datafusion_common::tree_node::TreeNodeRecursion> {
        Ok(datafusion_common::tree_node::TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        check_if_same_properties!(self, children);
        Ok(Arc::new(Self::try_new(
            children.swap_remove(0),
            self.output_partitions,
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        assert_or_internal_err!(
            partition < self.output_partitions,
            "CoalescePartitionGroupsExec invalid partition {partition}"
        );

        let input_partitions = self.input.output_partitioning().partition_count();
        let group_size = self.group_size();
        match group_size {
            0 => internal_err!(
                "CoalescePartitionGroupsExec requires at least one input partition per output partition"
            ),
            1 => self.input.execute(partition, context),
            _ => {
                let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);
                let elapsed_compute = baseline_metrics.elapsed_compute().clone();
                let _timer = elapsed_compute.timer();

                let mut builder =
                    RecordBatchReceiverStream::builder(self.schema(), group_size);

                for input_partition in (partition..input_partitions).step_by(self.output_partitions)
                {
                    builder.run_input(
                        Arc::clone(&self.input),
                        input_partition,
                        Arc::clone(&context),
                    );
                }

                let stream = builder.build();
                Ok(Box::pin(ObservedStream::new(
                    stream,
                    baseline_metrics,
                    None,
                )))
            }
        }
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        if partition.is_some() {
            let partition_count = self.output_partitions;
            let mut stats = Arc::unwrap_or_clone(self.input.partition_statistics(None)?);

            stats.num_rows = stats
                .num_rows
                .get_value()
                .map(|rows| rows / partition_count)
                .map(datafusion_common::stats::Precision::Inexact)
                .unwrap_or(datafusion_common::stats::Precision::Absent);
            stats.total_byte_size = stats
                .total_byte_size
                .get_value()
                .map(|bytes| bytes / partition_count)
                .map(datafusion_common::stats::Precision::Inexact)
                .unwrap_or(datafusion_common::stats::Precision::Absent);
            stats.column_statistics = stats
                .column_statistics
                .iter()
                .map(|_| datafusion_common::ColumnStatistics::new_unknown())
                .collect();

            Ok(Arc::new(stats))
        } else {
            self.input.partition_statistics(None)
        }
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common;
    use crate::expressions::Column;
    use crate::repartition::RepartitionExec;
    use crate::test;
    use datafusion_common::Result;

    #[tokio::test]
    async fn merge_partition_groups() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
        let input = test::scan_partitioned(8);
        let coalesce = CoalescePartitionGroupsExec::try_new(input, 2)?;

        assert_eq!(coalesce.properties().output_partitioning().partition_count(), 2);
        assert_eq!(coalesce.group_size(), 4);

        for partition in 0..2 {
            let batches = common::collect(coalesce.execute(partition, Arc::clone(&task_ctx))?).await?;
            assert_eq!(batches.len(), 4);
            let row_count: usize = batches.iter().map(|batch| batch.num_rows()).sum();
            assert_eq!(row_count, 400);
        }

        Ok(())
    }

    #[tokio::test]
    async fn preserves_hash_partitioning_metadata() -> Result<()> {
        let input = test::scan_partitioned(1);
        let schema = input.schema();
        let hash_expr = Arc::new(Column::new_with_schema("i", &schema)?) as Arc<dyn PhysicalExpr>;
        let repartition = Arc::new(RepartitionExec::try_new(
            input,
            Partitioning::Hash(vec![Arc::clone(&hash_expr)], 8),
        )?) as Arc<dyn ExecutionPlan>;

        let coalesce = CoalescePartitionGroupsExec::try_new(repartition, 2)?;
        match coalesce.properties().output_partitioning() {
            Partitioning::Hash(exprs, 2) => {
                assert_eq!(exprs.len(), 1);
                assert_eq!(exprs[0].to_string(), hash_expr.to_string());
            }
            other => panic!("expected hash partitioning, got {other:?}"),
        }

        Ok(())
    }
}
