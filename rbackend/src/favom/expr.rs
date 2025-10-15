use index_vec::Idx;
use std::collections::{hash_map, BTreeSet, HashMap};
use std::ops::AddAssign;

use super::utils::new_id;
use smallvec::SmallVec;

// TODO: prevent name collisions between rnd, secrets and pub.

new_id!(SecretId, SecretVec);
new_id!(RndId, RndVec);
new_id!(PubId, PubVec);
new_id!(OperatorId, OperatorVec);
new_id!(ExprId, ExprVec);

#[derive(Debug)]
pub struct VarStore<Id: Idx> {
    next_id: Id,
    names: index_vec::IndexVec<Id, String>,
    id_of: HashMap<String, Id>,
}

impl<Id: Idx> VarStore<Id> {
    pub fn new() -> Self {
        Self {
            next_id: Id::from_usize(0),
            names: index_vec::IndexVec::new(),
            id_of: HashMap::new(),
        }
    }
    pub fn make(&mut self, name: String) -> Id {
        let id = self.next_id;
        self.next_id = Id::from_usize(self.next_id.index() + 1);
        match self.id_of.entry(name.clone()) {
            hash_map::Entry::Occupied(_) => {
                panic!("Already used entry");
            }
            hash_map::Entry::Vacant(entry) => {
                entry.insert(id);
            }
        }
        self.names.push(name);
        id
    }
    pub fn len(&self) -> usize {
        self.names.len()
    }
}

impl<Id: Idx> std::ops::Index<Id> for VarStore<Id> {
    type Output = String;
    fn index(&self, index: Id) -> &String {
        &self.names[index]
    }
}

pub type SecretStore = VarStore<SecretId>;
pub type RndStore = VarStore<RndId>;
pub type PubStore = VarStore<PubId>;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum OpKind {
    Add,
    Mul,
    Neg,
    Other,
    Tuple,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum OpArity {
    Fixed(u32),
    Unbounded,
}

#[derive(Debug)]
pub struct Operator {
    pub name: String,
    pub arity: OpArity,
    pub bijective: bool,
    pub commutative: bool,
    pub kind: OpKind,
    pub infix: Option<String>,
}

impl Operator {
    pub fn bij(&self) -> bool {
        self.bijective
    }
}

#[derive(Debug)]
pub struct OpStore {
    next_id: OperatorId,
    operators: OperatorVec<Operator>,
    id_of: HashMap<String, OperatorId>,
}

impl OpStore {
    pub fn new() -> Self {
        Self {
            next_id: OperatorId::new(0),
            operators: OperatorVec::new(),
            id_of: HashMap::new(),
        }
    }
    pub fn make(&mut self, operator: Operator) -> OperatorId {
        let id = self.next_id;
        self.next_id.add_assign(1);
        match self.id_of.entry(operator.name.clone()) {
            hash_map::Entry::Occupied(_) => {
                panic!("Already used entry");
            }
            hash_map::Entry::Vacant(entry) => {
                entry.insert(id);
            }
        }
        self.operators.push(operator);
        return id;
    }
}

impl std::ops::Index<OperatorId> for OpStore {
    type Output = Operator;
    fn index(&self, index: OperatorId) -> &Operator {
        &self.operators[index]
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum Constant {
    Zero,
    One,
}

impl std::fmt::Display for Constant {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Constant::Zero => write!(fmt, "0"),
            Constant::One => write!(fmt, "1"),
        }
    }
}

pub type Operands = smallvec::SmallVec<[ExprId; 2]>;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Share {
    share_idx: u32,
    secret_var: SecretId,
}
impl Share {
    pub fn secret(&self) -> SecretId {
        self.secret_var
    }
    pub fn idx(&self) -> u32 {
        self.share_idx
    }
    pub fn pp<'a>(&self, sec_store: &'a SecretStore) -> String {
        format!("{}[{}]", sec_store[self.secret_var], self.share_idx)
    }
}
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Operation {
    operator: OperatorId,
    operands: Operands,
}
impl Operation {
    pub fn ops(&self) -> &Operands {
        &self.operands
    }
    pub fn operator(&self) -> OperatorId {
        self.operator
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum ExprNode {
    Rnd(RndId),
    Share(Share),
    Pub(PubId),
    Op(Operation),
    Const(Constant),
}

impl ExprNode {
    pub fn new_rnd(var: RndId) -> Self {
        ExprNode::Rnd(var)
    }
    pub fn new_share(share_idx: u32, secret_var: SecretId) -> Self {
        ExprNode::Share(Share {
            share_idx,
            secret_var,
        })
    }
    pub fn new_pub(var: PubId) -> Self {
        ExprNode::Pub(var)
    }
    pub fn new_const(constant: Constant) -> Self {
        ExprNode::Const(constant)
    }
    pub fn new_op(
        operator: OperatorId,
        operands: impl IntoIterator<Item = ExprId>,
        op_store: &OpStore,
    ) -> Self {
        let res = ExprNode::Op(Operation {
            operator,
            operands: operands.into_iter().collect(),
        });
        res.sort_ops(op_store)
    }
    fn new_op_dedup(
        operator: OperatorId,
        operands: impl IntoIterator<Item = ExprId>,
        op_store: &OpStore,
    ) -> Self {
        let mut operands = operands.into_iter().collect::<Vec<_>>();
        operands.sort_unstable();
        operands.dedup();
        Self::new_op(operator, operands, op_store)
    }

    fn sort_ops(self, op_store: &OpStore) -> Self {
        if let ExprNode::Op(Operation {
            operator,
            mut operands,
        }) = self
        {
            if op_store[operator].commutative {
                operands.sort_unstable()
            };
            ExprNode::Op(Operation { operator, operands })
        } else {
            self
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum SExpr {
    Rnd(RndId),
    Share(Share),
    Pub(PubId),
    Const(Constant),
    Op1 {
        operator: OperatorId,
        op: ExprId,
    },
    Op2 {
        operator: OperatorId,
        ops: [ExprId; 2],
    },
}

impl SExpr {
    pub fn from_expr(e_id: ExprId, expr_graph: &ExprGraph) -> SExpr {
        match &expr_graph.expr_store[e_id] {
            ExprNode::Rnd(var_id) => SExpr::Rnd(*var_id),
            ExprNode::Share(share) => SExpr::Share(*share),
            ExprNode::Pub(var_id) => SExpr::Pub(*var_id),
            ExprNode::Const(constant) => SExpr::Const(*constant),
            ExprNode::Op(op) => {
                let operands = op.ops();
                match operands.len() {
                    1 => SExpr::Op1 {
                        operator: op.operator(),
                        op: operands[0],
                    },
                    2 => SExpr::Op2 {
                        operator: op.operator(),
                        ops: [operands[0], operands[1]],
                    },
                    l => {
                        let op_name = &expr_graph.op_store[op.operator()].name;
                        panic!("Invalid operation '{op_name}' with {l} operands (expected 1 or 2).")
                    }
                }
            }
        }
    }
    pub fn parents_with_dup(&self) -> &[ExprId] {
        use SExpr::*;
        match self {
            Rnd(_) | Share(_) | Pub(_) | Const(_) => &[],
            Op1 { op, .. } => std::slice::from_ref(op),
            Op2 { ops, .. } => ops.as_slice(),
        }
    }
}

#[derive(Debug)]
pub struct ExprStore {
    next_id: ExprId,
    exprs: ExprVec<ExprNode>,
    node2id: HashMap<ExprNode, ExprId>,
}
impl ExprStore {
    pub fn new() -> Self {
        Self {
            next_id: ExprId::new(0),
            exprs: ExprVec::new(),
            node2id: HashMap::new(),
        }
    }
    pub fn make(&mut self, expr: ExprNode) -> ExprId {
        if let Some(id) = self.node2id.get(&expr) {
            return *id;
        }
        let id = self.next_id;
        if let ExprNode::Op(Operation { operands, .. }) = &expr {
            assert!(operands.is_sorted());
            assert!(operands.iter().all(|op| *op < id));
        }
        self.next_id.add_assign(1);
        self.exprs.push(expr.clone());
        self.node2id.insert(expr, id);
        return id;
    }
    pub fn find_sub_exprs_es(&self, es: &[ExprId]) -> Vec<(ExprId, bool)> {
        let mut res = Vec::new();
        let mut todo = std::collections::BTreeMap::new();
        for e in es {
            todo.insert(*e, true);
        }
        while let Some((e, is_top)) = todo.pop_last() {
            res.push((e, is_top));
            if let ExprNode::Op(Operation { operands, .. }) = &self[e] {
                for operand in operands {
                    todo.entry(*operand).or_insert(false);
                }
            }
        }
        res.reverse();
        return res;
    }
    pub fn find_sub_exprs(&self, e: ExprId) -> Vec<(ExprId, bool)> {
        self.find_sub_exprs_es(&[e])
    }
    pub fn len(&self) -> usize {
        self.exprs.len()
    }
}

impl std::ops::Index<ExprId> for ExprStore {
    type Output = ExprNode;
    fn index(&self, index: ExprId) -> &ExprNode {
        &self.exprs[index]
    }
}

#[derive(Debug)]
pub struct ExprGraph {
    pub sec_store: SecretStore,
    pub rnd_store: RndStore,
    pub pub_store: PubStore,
    pub op_store: OpStore,
    pub expr_store: ExprStore,
    pub sexprs: ExprVec<SExpr>,
    pub children: ExprVec<SmallVec<[ExprId; 4]>>,
}

impl ExprGraph {
    pub fn new() -> Self {
        Self {
            sec_store: SecretStore::new(),
            rnd_store: RndStore::new(),
            pub_store: PubStore::new(),
            op_store: OpStore::new(),
            expr_store: ExprStore::new(),
            sexprs: ExprVec::new(),
            children: ExprVec::new(),
        }
    }
    fn add_expr(&mut self, e: ExprNode) -> ExprId {
        let id = self.expr_store.make(e);
        self.sexprs.push(SExpr::from_expr(id, self));
        self.children.push(SmallVec::new());
        for p in self.sexprs[id]
            .parents_with_dup()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
        {
            self.children[p].push(id);
        }
        id
    }
    pub fn add_secret(&mut self, name: String) -> (ExprId, SecretId) {
        let secret_var_id = self.sec_store.make(name.clone());
        let expr = ExprNode::new_share(0, secret_var_id);
        (self.add_expr(expr), secret_var_id)
    }
    pub fn add_random(&mut self, name: String) -> ExprId {
        let expr = ExprNode::Rnd(self.rnd_store.make(name));
        self.add_expr(expr)
    }
    pub fn add_pub(&mut self, name: String) -> ExprId {
        let expr = ExprNode::Pub(self.pub_store.make(name));
        self.add_expr(expr)
    }
    pub fn add_op(&mut self, op: OperatorId, operands: impl IntoIterator<Item = ExprId>) -> ExprId {
        let operands = operands.into_iter().collect::<Vec<_>>();
        if self.op_store[op].bij() && operands.len() == 1 {
            // We can simply use the operand, since we don't do any arithmetic simplification
            // afterwiards.
            operands[0]
        } else {
            let expr = ExprNode::new_op(op, operands, &self.op_store);
            self.add_expr(expr)
        }
    }
    pub fn print_expr(&self, e: ExprId) -> String {
        match &self.expr_store[e] {
            ExprNode::Rnd(v) => self.rnd_store[*v].clone(),
            ExprNode::Pub(v) => self.pub_store[*v].clone(),
            ExprNode::Share(sh) => sh.pp(&self.sec_store),
            ExprNode::Const(Constant::Zero) => "0".to_owned(),
            ExprNode::Const(Constant::One) => "1".to_owned(),
            ExprNode::Op(op) => {
                if let Some(op_infix) = &self.op_store[op.operator].infix {
                    match op.operands.len() {
                        1 => {
                            return format!("({}{})", op_infix, self.print_expr(op.operands[0]));
                        }
                        2 => {
                            return format!(
                                "({} {} {})",
                                self.print_expr(op.operands[0]),
                                op_infix,
                                self.print_expr(op.operands[1]),
                            );
                        }
                        _ => {}
                    }
                }
                return format!(
                    "{}({})",
                    self.op_store[op.operator].name,
                    op.operands
                        .iter()
                        .map(|o| self.print_expr(*o))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }
    }
}
