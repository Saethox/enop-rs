use std::slice::from_mut;
use mahf::{Individual, Problem, State};
use mahf::problems::{Evaluate, LimitedVectorProblem, VectorProblem};
use enop_rs::{EngineeringOptimizationEvaluator, EngineeringOptimizationProblem};

fn main() {
    let problem = EngineeringOptimizationProblem::heat_exchanger_network_design_case1();
    let mut evaluator = EngineeringOptimizationEvaluator::new(&problem);

    println!("Name: {}", problem.name());
    println!("Dimensionality: {}", problem.dimension());
    println!("Domain: {:?}", problem.domain());

    let solution = problem.domain().into_iter().map(|range| range.start + (range.end - range.start) / 2.0).collect();
    let mut individual = Individual::new_unevaluated(solution);
    evaluator.evaluate(&problem, &mut State::new(), from_mut(&mut individual));

    println!("f({:?}) = {}", individual.solution(), individual.objective().value());
}