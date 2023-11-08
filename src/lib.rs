use std::ops::Range;

use mahf::{
    problems::{Evaluate, LimitedVectorProblem, VectorProblem},
    ExecResult, Individual, Problem, SingleObjective, State,
};
use numpy::{ndarray::Array1, IntoPyArray};
use pyo3::{IntoPy, PyObject, Python};

pub struct EngineeringOptimizationProblem {
    name: String,
    dim: usize,
    domain: Vec<Range<f64>>,
}

impl EngineeringOptimizationProblem {
    pub fn new(name: impl AsRef<str>) -> ExecResult<Self> {
        Python::with_gil(|py| {
            let problems = Python::import(py, "enoppy.paper_based.rwco_2020")?;
            let py_problem_class = problems.getattr(name.as_ref())?;
            let py_problem = py_problem_class.call0()?;
            let dim = py_problem.getattr("n_dims")?.extract::<usize>()?;
            let bounds = py_problem.getattr("bounds")?.extract::<Vec<Vec<f64>>>()?;
            let domain = bounds.into_iter().map(|bound| bound[0]..bound[1]).collect();

            let problem = Self {
                name: name.as_ref().to_string(),
                dim,
                domain,
            };

            Ok(problem)
        })
    }

    pub fn heat_exchanger_network_design_case1() -> Self {
        Self::new("HeatExchangerNetworkDesignCase1Problem").unwrap()
    }

    pub fn heat_exchanger_network_design_case2() -> Self {
        Self::new("HeatExchangerNetworkDesignCase2Problem").unwrap()
    }

    pub fn haverly_pooling() -> Self {
        Self::new("HaverlyPoolingProblem").unwrap()
    }

    pub fn blending_pooling_separation() -> Self {
        Self::new("BlendingPoolingSeparationProblem").unwrap()
    }

    pub fn propane_isobutane_n_butane_nonsharp_separation() -> Self {
        Self::new("PropaneIsobutaneNButaneNonsharpSeparationProblem").unwrap()
    }

    pub fn optimal_operation_alkylation_unit() -> Self {
        Self::new("OptimalOperationAlkylationUnitProblem").unwrap()
    }

    pub fn reactor_network_design() -> Self {
        Self::new("ReactorNetworkDesignProblem").unwrap()
    }

    pub fn process_synthesis_01() -> Self {
        Self::new("ProcessSynthesis01Problem").unwrap()
    }

    pub fn process_synthesis_02() -> Self {
        Self::new("ProcessSynthesis02Problem").unwrap()
    }

    pub fn process_design() -> Self {
        Self::new("ProcessDesignProblem").unwrap()
    }

    pub fn process_synthesis_and_design() -> Self {
        Self::new("ProcessSynthesisAndDesignProblem").unwrap()
    }

    pub fn process_flow_sheeting() -> Self {
        Self::new("ProcessFlowSheetingProblem").unwrap()
    }

    pub fn two_reactor() -> Self {
        Self::new("TwoReactorProblem").unwrap()
    }

    pub fn multi_product_batch_plant() -> Self {
        Self::new("MultiProductBatchPlantProblem").unwrap()
    }

    pub fn weight_minimization_speed_reducer() -> Self {
        Self::new("WeightMinimizationSpeedReducerProblem").unwrap()
    }

    pub fn optimal_design_industrial_refrigeration_system() -> Self {
        Self::new("OptimalDesignIndustrialRefrigerationSystemProblem").unwrap()
    }

    pub fn tension_compression_spring_design() -> Self {
        Self::new("TensionCompressionSpringDesignProblem").unwrap()
    }

    pub fn pressure_vessel_design() -> Self {
        Self::new("PressureVesselDesignProblem").unwrap()
    }

    pub fn welded_beam_design() -> Self {
        Self::new("WeldedBeamDesignProblem").unwrap()
    }

    pub fn three_bar_truss_design() -> Self {
        Self::new("ThreeBarTrussDesignProblem").unwrap()
    }

    pub fn multiple_disk_clutch_brake_design() -> Self {
        Self::new("MultipleDiskClutchBrakeDesignProblem").unwrap()
    }

    pub fn planetary_gear_train_design() -> Self {
        Self::new("PlanetaryGearTrainDesignOptimizationProblem").unwrap()
    }

    pub fn step_cone_pulley() -> Self {
        Self::new("StepConePulleyProblem").unwrap()
    }
}

impl Problem for EngineeringOptimizationProblem {
    type Encoding = Vec<f64>;
    type Objective = SingleObjective;

    fn name(&self) -> &str {
        self.name.as_str()
    }
}

impl VectorProblem for EngineeringOptimizationProblem {
    type Element = f64;

    fn dimension(&self) -> usize {
        self.dim
    }
}

impl LimitedVectorProblem for EngineeringOptimizationProblem {
    fn domain(&self) -> Vec<Range<Self::Element>> {
        self.domain.clone()
    }
}

#[derive(Clone)]
pub struct EngineeringOptimizationEvaluator {
    inner: PyObject,
}

impl EngineeringOptimizationEvaluator {
    pub fn new(problem: &EngineeringOptimizationProblem) -> Self {
        Python::with_gil(|py| {
            let problems = Python::import(py, "enoppy.paper_based.rwco_2020")?;
            let py_problem_class = problems.getattr(problem.name.as_str())?;
            let inner = py_problem_class.call0()?.into_py(py);

            let evaluator = Self { inner };

            ExecResult::Ok(evaluator)
        })
        .unwrap()
    }
}

impl Evaluate for EngineeringOptimizationEvaluator {
    type Problem = EngineeringOptimizationProblem;

    fn evaluate(
        &mut self,
        _problem: &Self::Problem,
        _state: &mut State<Self::Problem>,
        individuals: &mut [Individual<Self::Problem>],
    ) {
        Python::with_gil(|py| {
            for individual in individuals {
                let solution = Array1::from_vec(individual.solution().clone());
                let np_solution = solution.into_pyarray(py);
                let problem = self.inner.as_ref(py);
                let fitness = problem
                    .call_method1("evaluate", (np_solution,))
                    .unwrap()
                    .extract::<f64>()
                    .unwrap_or(f64::INFINITY);
                let objective_value = SingleObjective::try_from(fitness).unwrap_or_default();
                individual.set_objective(objective_value);
            }
        });
    }
}
