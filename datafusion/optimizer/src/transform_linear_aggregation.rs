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

//! [`EliminateGroupByConstant`] removes constant expressions from `GROUP BY` clause
use std::sync::Arc;

use crate::optimizer::ApplyOrder;
use crate::{OptimizerConfig, OptimizerRule};

use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{internal_err, DFSchema, Result};
use datafusion_expr::expr::{AggregateFunction, AggregateFunctionParams, Alias};
use datafusion_expr::{
    Aggregate, BinaryExpr, Expr, ExprSchemable, LogicalPlan, LogicalPlanBuilder,
    Operator, Volatility,
};

/// Optimizer rule that removes constant expressions from `GROUP BY` clause
/// and places additional projection on top of aggregation, to preserve
/// original schema
#[derive(Default, Debug)]
pub struct TransformLinearAggregation {}

impl TransformLinearAggregation {
    pub fn new() -> Self {
        Self {}
    }
}

impl OptimizerRule for TransformLinearAggregation {
    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        match plan {
            LogicalPlan::Aggregate(aggregate) => {
                // TODO: transform linear udaf in group by cases
                // Currently, we only try to transform `aggr_expr` in no grouping cases,
                // and surely `aggr_expr` should exist
                if !aggregate.group_expr.is_empty() || aggregate.aggr_expr.is_empty() {
                    return Ok(Transformed::no(LogicalPlan::Aggregate(aggregate)));
                }

                // Try to transform `aggr_expr`
                let mut new_aggr_exprs = Vec::with_capacity(aggregate.aggr_expr.len());
                let mut new_proj_exprs = Vec::new();
                let mut transformed = false;
                let input_schema = aggregate.input.schema();
                for expr in aggregate.aggr_expr.iter() {
                    transformed = maybe_transform_aggr_expr(
                        expr,
                        input_schema,
                        &mut new_aggr_exprs,
                        &mut new_proj_exprs,
                    )?;
                }

                // If transform happened, we need to rewrite the old `Aggregate`
                if !transformed {
                    return Ok(Transformed::no(LogicalPlan::Aggregate(aggregate)));
                }

                let transformed_aggregate = LogicalPlan::Aggregate(Aggregate::try_new(
                    aggregate.input,
                    aggregate.group_expr.clone(),
                    new_aggr_exprs,
                )?);

                let projection_expr =
                    aggregate.group_expr.into_iter().chain(new_proj_exprs);

                let projection = LogicalPlanBuilder::from(transformed_aggregate)
                    .project(projection_expr)?
                    .build()?;

                Ok(Transformed::yes(projection))
            }
            _ => Ok(Transformed::no(plan)),
        }
    }

    fn name(&self) -> &str {
        "transform_linear_aggregation"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }
}

/// Try to transform the udaf
///
/// It will try to transform if `aggr_expr` satisfies following requirements:
///   - It is a `linear` udaf
///   - It should be `not distinct`, `not filter exists`, `not order by exists`
///   - Only one arg exist in `args`, and it is `non-nullable`
///
/// And
fn maybe_transform_aggr_expr(
    aggr_expr: &Expr,
    input_schema: &DFSchema,
    new_aggr_exprs: &mut Vec<Expr>,
    new_proj_exprs: &mut Vec<Expr>,
) -> Result<bool> {
    let Expr::AggregateFunction(AggregateFunction { func, params }) = aggr_expr else {
        return internal_err!(
            "expect Expr::AggregateFunction in aggr_expr, but found:{aggr_expr:?}"
        );
    };

    let AggregateFunctionParams {
        args,
        distinct,
        filter,
        order_by,
        null_treatment,
    } = params;

    // Check if we should try to transform
    if !func.is_linear()
        || *distinct
        || filter.is_some()
        || order_by.is_some()
        || args.len() != 1
        || args[0].nullable(input_schema)?
    {
        dbg!("return in check transforming failed");
        new_aggr_exprs.push(aggr_expr.clone());
        return Ok(false);
    }

    // For simplicity, we just process the simple situation `aggr(a + b)` in demo
    // We split it into two aggr exprs: `aggr(a) + aggr(b)`.
    // And NOTICE, we need to use `Projection` to alias it back to `aggr(a + b)` to
    // let it parent can still refer to it.
    let Expr::BinaryExpr(BinaryExpr {
        left,
        op: Operator::Plus,
        right,
    }) = &args[0]
    else {
        dbg!("return in tranform failed");
        new_aggr_exprs.push(aggr_expr.clone());
        return Ok(false);
    };

    // Build the two `new aggr exprs`
    let aggr1 = Expr::AggregateFunction(AggregateFunction::new_udf(
        Arc::clone(&func),
        vec![*left.clone()],
        *distinct,
        filter.clone(),
        order_by.clone(),
        null_treatment.clone(),
    ));
    let aggr2 = Expr::AggregateFunction(AggregateFunction::new_udf(
        Arc::clone(&func),
        vec![*right.clone()],
        *distinct,
        filter.clone(),
        order_by.clone(),
        null_treatment.clone(),
    ));
    new_aggr_exprs.extend([aggr1.clone(), aggr2.clone()]);

    // Build the `new proj expr`
    let (_, proj_name) = aggr_expr.qualified_name();
    let new_proj_expr = (aggr1 + aggr2).alias(proj_name);
    new_proj_exprs.push(new_proj_expr);

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_optimized_plan_eq_snapshot;
    use crate::test::*;
    use crate::Optimizer;
    use crate::OptimizerContext;

    use arrow::datatypes::DataType;
    use datafusion_common::Result;
    use datafusion_expr::expr::ScalarFunction;
    use datafusion_expr::{
        col, lit, ColumnarValue, LogicalPlanBuilder, ScalarFunctionArgs, ScalarUDF,
        ScalarUDFImpl, Signature, TypeSignature,
    };

    use datafusion_functions_aggregate::expr_fn::count;
    use datafusion_functions_aggregate::expr_fn::sum;

    use std::sync::Arc;

    macro_rules! assert_optimized_plan_equal {
            (
                $plan:expr,
                @ $expected:literal $(,)?
            ) => {{
                let rule: Arc<dyn crate::OptimizerRule + Send + Sync> = Arc::new(TransformLinearAggregation::new());
                assert_optimized_plan_eq_snapshot!(
                    rule,
                    $plan,
                    @ $expected,
                )
            }};
        }

    #[test]
    fn test_eliminate_gby_literal() {
        let scan = test_table_scan().unwrap();
        let plan = LogicalPlanBuilder::from(scan)
            .aggregate(Vec::<Expr>::new(), vec![sum(col("c") + col("b"))])
            .unwrap()
            .build()
            .unwrap();

        let rule: Arc<dyn OptimizerRule + Send + Sync> =
            Arc::new(TransformLinearAggregation::new());
        let opt_context = OptimizerContext::new().with_max_passes(1);
        let optimizer = Optimizer::with_rules(vec![Arc::clone(&rule)]);
        let optimized_plan = optimizer.optimize(plan, &opt_context, |_, _| {}).unwrap();
        println!("{optimized_plan}");

        // assert_optimized_plan_equal!(plan, @r"
        // Projection: test.a, UInt32(1), count(test.c)
        //   Aggregate: groupBy=[[test.a]], aggr=[[count(test.c)]]
        //     TableScan: test
        // ")
    }

    //     #[test]
    //     fn test_eliminate_constant() -> Result<()> {
    //         let scan = test_table_scan()?;
    //         let plan = LogicalPlanBuilder::from(scan)
    //             .aggregate(vec![lit("test"), lit(123u32)], vec![count(col("c"))])?
    //             .build()?;

    //         assert_optimized_plan_equal!(plan, @r#"
    //         Projection: Utf8("test"), UInt32(123), count(test.c)
    //           Aggregate: groupBy=[[]], aggr=[[count(test.c)]]
    //             TableScan: test
    //         "#)
    //     }
}
