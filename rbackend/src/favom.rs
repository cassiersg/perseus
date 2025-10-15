use std::collections::HashMap;

use pyo3::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

// These files implement the verification algorithms.
pub(crate) mod expr;
pub mod state;
mod utils;

// We provide a python wrapper here.
#[pyclass]
pub struct FavomExprs {
    pub expr_graph: expr::ExprGraph,
    operators: HashMap<String, expr::OperatorId>,
}

#[pyclass]
pub struct Secret(expr::SecretId);

#[pymethods]
impl Secret {
    fn __repr__(&self) -> String {
        format!("Secret({})", self.0)
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }
    fn __hash__(&self) -> usize {
        self.0.idx()
    }
}

#[pyclass]
#[derive(Copy, Clone)]
pub struct Expr(pub expr::ExprId);

impl FavomExprs {
    fn add_operator_inner(
        &mut self,
        name: String,
        arity: expr::OpArity,
        bijective: bool,
        commutative: bool,
        kind: expr::OpKind,
        infix: Option<String>,
    ) {
        let operator = expr::Operator {
            name: name.clone(),
            arity,
            bijective,
            commutative,
            kind,
            infix,
        };
        let op_id = self.expr_graph.op_store.make(operator);
        self.operators.insert(name, op_id);
    }
}

#[pymethods]
impl FavomExprs {
    #[new]
    fn new() -> FavomExprs {
        let mut res = FavomExprs {
            expr_graph: expr::ExprGraph::new(),
            operators: HashMap::new(),
        };
        use expr::{OpArity::*, OpKind::*};
        res.add_operator_inner(
            String::from("add"),
            Unbounded,
            true,
            true,
            Add,
            Some("+".to_owned()),
        );
        res.add_operator_inner(
            String::from("mul"),
            Unbounded,
            false,
            true,
            Mul,
            Some("*".to_owned()),
        );
        res.add_operator_inner(
            String::from("neg"),
            Fixed(1),
            true,
            false,
            Neg,
            Some("-".to_owned()),
        );
        res.add_operator_inner(String::from("tuple"), Unbounded, false, true, Tuple, None);
        res
    }

    pub fn add_operator(
        &mut self,
        name: String,
        fixed_arity: bool,
        arity: u32,
        bijective: bool,
        commutative: bool,
    ) {
        assert!(!self.operators.contains_key(&name));
        let arity = if fixed_arity {
            expr::OpArity::Fixed(arity)
        } else {
            expr::OpArity::Unbounded
        };
        self.add_operator_inner(
            name,
            arity,
            bijective,
            commutative,
            expr::OpKind::Other,
            None,
        );
    }

    pub fn add_secret(&mut self, name: String) -> (Expr, Secret) {
        let (e, s) = self.expr_graph.add_secret(name.clone());
        (Expr(e), Secret(s))
    }

    pub fn add_random(&mut self, name: String) -> Expr {
        Expr(self.expr_graph.add_random(name))
    }

    pub fn add_pub(&mut self, name: String) -> Expr {
        Expr(self.expr_graph.add_pub(name))
    }

    pub fn add_op_expr(&mut self, op: &str, operands: &Bound<'_, pyo3::types::PyList>) -> Expr {
        assert!(self.operators.contains_key(op), "Unknown operator {op}");
        let operands = operands.iter().map(|e| e.extract::<Expr>().unwrap().0);
        Expr(self.expr_graph.add_op(self.operators[op], operands))
    }
}

#[pyclass]
pub struct FavomState {
    state: state::State,
}

#[pymethods]
impl FavomState {
    #[new]
    pub fn new(n_shares: u32, exprs: &FavomExprs, top_exprs: Vec<Expr>) -> FavomState {
        Self {
            state: state::State::new(
                &exprs.expr_graph,
                top_exprs.into_iter().map(|e| e.0).collect(),
            ),
        }
    }
    pub fn simplify1(&mut self, exprs: &FavomExprs) -> bool {
        self.state.simplify1(&exprs.expr_graph)
    }
    pub fn simplify_until(&mut self, t: u32, exprs: &FavomExprs) -> bool {
        self.state.simplify_until(t, &exprs.expr_graph)
    }
    pub fn continue_simpl(&self, t: u32) -> bool {
        self.state.continue_simpl(t)
    }

    pub fn used_secrets(&self) -> Vec<Secret> {
        self.state.used_secrets().into_iter().map(Secret).collect()
    }

    pub fn n_bij(&self) -> usize {
        self.state.n_bij()
    }

    pub fn pprint(&self, exprs: &FavomExprs) -> String {
        self.state.pprint(&exprs.expr_graph)
    }
}

pub fn check_tuple(exprs: &FavomExprs, top_exprs: Vec<expr::ExprId>) -> bool {
    let exprs = &exprs.expr_graph;
    let mut state = state::State::new(exprs, top_exprs);
    state.simplify_until(0, exprs);
    state.used_secrets().is_empty()
}

pub fn check_tuples_inner<I>(exprs: &FavomExprs, top_exprs: I) -> Vec<bool>
where
    I: rayon::iter::IndexedParallelIterator<Item = Vec<Expr>>,
{
    let nexprs = top_exprs.len();
    crate::progress_bar::with_progress(
        |it_cnt| {
            top_exprs
                .into_par_iter()
                .with_min_len(std::thread::available_parallelism().unwrap().get() * 16)
                .map(|t_e| {
                    let res = check_tuple(exprs, t_e.into_iter().map(|e| e.0).collect());
                    it_cnt.inc(1);
                    res
                })
                .collect()
        },
        nexprs as u64,
        "Favom tuples",
        std::time::Duration::from_millis(500),
    )
}

#[pyfunction]
pub fn check_tuples(exprs: &FavomExprs, top_exprs: Vec<Vec<Expr>>) -> Vec<bool> {
    check_tuples_inner(exprs, top_exprs.into_par_iter())
}
