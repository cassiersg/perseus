// TODO optimizations:
// - cleanup
// - make remove_node/remove_child non-recursive

use super::expr;

use expr::{ExprId, SExpr};

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct Node {
    /// The current value of the node
    descriptor: SExpr,
    n_children: u32,
    /// true for top probed nodes
    is_top: bool,
    /// has been rewritten as a random
    rewritten: bool,
}

impl Node {
    fn is_rnd(&self) -> bool {
        matches!(self.descriptor, SExpr::Rnd(_))
    }
    fn unused(&self) -> bool {
        self.n_children == 0 && !self.is_top
    }
    fn has_single_child(&self) -> bool {
        self.n_children == 1 && !self.is_top
    }
}

#[derive(Debug, Clone)]
struct SecretInfo {
    n_used_shares: u32,
}

impl SecretInfo {
    fn new() -> Self {
        Self { n_used_shares: 0 }
    }
    fn incr(&mut self) {
        self.n_used_shares += 1;
    }
    fn decr(&mut self) {
        self.n_used_shares -= 1;
    }
}

#[derive(Debug, Clone)]
pub struct State {
    nodes: expr::ExprVec<Option<Node>>,
    sharings: expr::SecretVec<SecretInfo>,
    /// Next randoms to eliminate
    todo: Vec<ExprId>,
    /// top tuple node
    top: Vec<ExprId>,
    /// tracking all bijections for record
    bij: Vec<(ExprId, ExprId)>,
}

impl State {
    pub fn new(expr_graph: &expr::ExprGraph, top_exprs: Vec<expr::ExprId>) -> Self {
        let sharings = expr::SecretVec::from(vec![SecretInfo::new(); expr_graph.sec_store.len()]);
        let mut res = Self {
            nodes: expr::ExprVec::from(vec![None; expr_graph.expr_store.len()]),
            sharings,
            todo: Vec::new(),
            top: Vec::new(),
            bij: Vec::new(),
        };
        let mut used_nodes = expr::ExprVec::from(vec![false; expr_graph.expr_store.len()]);
        let mut top_nodes = expr::ExprVec::from(vec![false; expr_graph.expr_store.len()]);
        let mut randoms = Vec::new();
        for e in top_exprs.iter() {
            top_nodes[*e] = true;
            used_nodes[*e] = true;
        }
        for (e, n) in expr_graph.sexprs.iter_enumerated().rev() {
            if used_nodes[e] {
                for p in n.parents_with_dup() {
                    used_nodes[*p] = true;
                }
                if let SExpr::Rnd(_) = n {
                    randoms.push(e);
                }
            }
        }
        for ((e, used), is_top) in used_nodes.iter_enumerated().zip(top_nodes.iter()) {
            if *used {
                res.add_expr_assume_parents(e, expr_graph, *is_top);
            }
        }
        res.top = top_exprs;
        //res.check_children(expr_graph);
        res.init_todo(&randoms);
        res
    }
    fn node(&self, n: ExprId) -> &Node {
        self.nodes[n].as_ref().unwrap()
    }
    fn node_mut(&mut self, n: ExprId) -> &mut Node {
        self.nodes[n].as_mut().unwrap()
    }
    fn add_used_share(&mut self, secret: expr::SecretId) {
        self.sharings[secret].incr()
    }
    fn rm_used_share(&mut self, secret: expr::SecretId) {
        self.sharings[secret].decr()
    }
    fn add_child(&mut self, parent: ExprId) {
        let p_node = self.node(parent);
        if let SExpr::Share(share) = &p_node.descriptor {
            if p_node.unused() {
                self.add_used_share(share.secret());
            }
        }
        self.node_mut(parent).n_children += 1;
    }
    fn set_parent(&mut self, n: ExprId) {
        match self.node(n).descriptor {
            SExpr::Op1 { op, .. } => {
                self.add_child(op);
            }
            SExpr::Op2 {
                ops: [op1, op2], ..
            } => {
                self.add_child(op1);
                if op2 != op1 {
                    self.add_child(op2);
                }
            }
            _ => {}
        }
    }
    fn add_expr_assume_parents(
        &mut self,
        e_id: expr::ExprId,
        expr_graph: &expr::ExprGraph,
        is_top: bool,
    ) {
        let node = Node {
            descriptor: expr_graph.sexprs[e_id],
            n_children: 0,
            is_top,
            rewritten: false,
        };
        assert!(
            self.nodes[e_id].is_none(),
            "old node: {:?}, new: {:?}",
            self.node(e_id),
            node
        );
        self.nodes[e_id] = Some(node);
        self.set_parent(e_id);
    }
    fn check_children(&self, expr_graph: &expr::ExprGraph) {
        for (nid, n) in self.nodes.iter_enumerated() {
            if let Some(n) = n {
                assert_eq!(
                    n.n_children as usize,
                    self.children(nid, expr_graph).count(),
                    "fail for node {nid} (all chilrden: {:?}, active children: {:?})",
                    expr_graph.children[nid],
                    self.children(nid, expr_graph).collect::<Vec<_>>()
                );
            }
        }
    }
    fn init_todo(&mut self, randoms: &[ExprId]) {
        self.todo = randoms
            .iter()
            .copied()
            .filter(|n| self.node(*n).has_single_child())
            .collect();
    }
    fn remove_child(&mut self, parent: ExprId) {
        let mut parents = vec![parent];
        while let Some(parent) = parents.pop() {
            assert!(self.node(parent).n_children > 0);
            self.node_mut(parent).n_children -= 1;
            if !self.node(parent).is_top {
                match self.node(parent).n_children {
                    0 => match self.nodes[parent].take().unwrap().descriptor {
                        SExpr::Share(share) => self.rm_used_share(share.secret()),
                        SExpr::Op1 { op, .. } => parents.push(op),
                        SExpr::Op2 {
                            ops: [op1, op2], ..
                        } => {
                            parents.push(op1);
                            if op1 != op2 {
                                parents.push(op2);
                            }
                        }
                        _ => {}
                    },
                    1 if self.node(parent).is_rnd() => {
                        self.todo.push(parent);
                    }
                    _ => {}
                }
            }
        }
    }
    fn remove_other_parent(&mut self, parent: ExprId, child: ExprId) {
        match self.node(child).descriptor {
            SExpr::Op1 { .. } => {}
            SExpr::Op2 {
                ops: [op1, op2], ..
            } => {
                assert!(parent == op1 || parent == op2);
                self.remove_child(if parent == op1 { op2 } else { op1 });
            }
            ref descriptor => {
                panic!(
                    "Invalid node descriptor in remove_other_parent {:?}",
                    descriptor
                );
            }
        }
    }
    fn children<'s>(
        &'s self,
        n: ExprId,
        expr_graph: &'s expr::ExprGraph,
    ) -> impl Iterator<Item = ExprId> + 's {
        expr_graph.children[n]
            .iter()
            .copied()
            .filter(|n| self.nodes[*n].as_ref().is_some_and(|n| !n.rewritten))
    }
    fn is_rnd_for_bij(&self, rid: ExprId, expr_graph: &expr::ExprGraph) -> bool {
        let r = self.node(rid);
        if !(r.is_rnd() && r.has_single_child()) {
            return false;
        }
        //assert_eq!(
        //    self.children(rid, expr_graph).count(),
        //    self.node(rid).n_children as usize
        //);
        let mut children = self.children(rid, expr_graph);
        let child = children.next().unwrap();
        assert!(children.next().is_none());
        match self.node(child).descriptor {
            SExpr::Op1 { operator, .. } => expr_graph.op_store[operator].bij(),
            SExpr::Op2 {
                operator,
                ops: [n1, n2],
            } => expr_graph.op_store[operator].bij() && n1 != n2,
            _ => false,
        }
    }
    fn apply_bij(&mut self, r: ExprId, expr_graph: &expr::ExprGraph) {
        let child = {
            let mut children = self.children(r, expr_graph);
            let child = children.next().unwrap();
            assert!(children.next().is_none());
            child
        };
        assert_eq!(self.node(r).n_children, 1);
        self.node_mut(r).n_children = 0;
        self.remove_other_parent(r, child);
        self.node_mut(child).descriptor = self.node(r).descriptor;
        self.bij.push((r, child));
        self.node_mut(child).rewritten = true;
        if self.is_rnd_for_bij(child, expr_graph) {
            self.todo.push(child);
        }
    }
    pub fn simplify1(&mut self, expr_graph: &expr::ExprGraph) -> bool {
        while let Some(r) = self.todo.pop() {
            if self.nodes[r].is_some() && self.is_rnd_for_bij(r, expr_graph) {
                //println!("is_rnd_for_bij, applying bij");
                self.apply_bij(r, expr_graph);
                return true;
            }
            //println!("not is_rnd_for_bij");
        }
        false
    }
    pub fn simplify(&mut self, expr_graph: &expr::ExprGraph) -> bool {
        let mut simplified = false;
        while self.simplify1(expr_graph) {
            simplified = true;
        }
        simplified
    }
    fn find_share(&self, t: u32) -> Option<expr::SecretId> {
        self.sharings
            .iter_enumerated()
            .filter_map(|(secret, info)| (info.n_used_shares > t).then_some(secret))
            .next()
    }
    pub fn continue_simpl(&self, t: u32) -> bool {
        self.find_share(t).is_some()
    }
    pub fn simplify_until(&mut self, t: u32, expr_graph: &expr::ExprGraph) -> bool {
        //println!("State before simplify_until: {:#?}", self);
        while self.continue_simpl(t) && self.simplify1(expr_graph) {}
        //println!("State after simplify_until: {:?}", self);
        !self.continue_simpl(t)
    }
    pub fn used_secrets(&self) -> Vec<expr::SecretId> {
        self.sharings
            .iter_enumerated()
            .filter(|(_, s)| s.n_used_shares != 0)
            .map(|(v, _)| v)
            .collect()
    }
    pub fn n_bij(&self) -> usize {
        self.bij.len()
    }
    fn used_nodes(&self) -> expr::ExprVec<bool> {
        let mut res = expr::ExprVec::from(vec![false; self.nodes.len()]);
        let mut to_explore = self.top.clone();
        while let Some(node) = to_explore.pop() {
            if !res[node] {
                res[node] = true;
                match &self.node(node).descriptor {
                    SExpr::Op1 { op, .. } => {
                        to_explore.push(*op);
                    }
                    SExpr::Op2 { ops, .. } => {
                        to_explore.push(ops[0]);
                        to_explore.push(ops[1]);
                    }
                    SExpr::Rnd(_) | SExpr::Share(_) | SExpr::Pub(_) | SExpr::Const(_) => {}
                }
            }
        }
        res
    }
    fn pp_node(&self, node: ExprId, expr_graph: &expr::ExprGraph) -> String {
        match self.node(node).descriptor {
            SExpr::Rnd(var) => expr_graph.rnd_store[var].clone(),
            SExpr::Pub(var) => expr_graph.pub_store[var].clone(),
            SExpr::Share(share) => share.pp(&expr_graph.sec_store).to_owned(),
            SExpr::Const(c) => format!("'{}'", c),
            SExpr::Op1 { .. } | SExpr::Op2 { .. } => {
                format!("{:?}", node)
            }
        }
    }
    pub fn pprint(&self, expr_graph: &expr::ExprGraph) -> String {
        let mut res = Vec::new();
        for ((nid, node), used) in self
            .nodes
            .iter_enumerated()
            .zip(self.used_nodes().into_iter())
        {
            if used {
                let node = node.as_ref().unwrap();
                use SExpr::*;
                let expr_s = match &node.descriptor {
                    Rnd(_) | Share(_) | Pub(_) | Const(_) => None,
                    Op1 { operator, op } => {
                        let op_s = self.pp_node(*op, expr_graph);
                        match &expr_graph.op_store[*operator].infix {
                            Some(s) => Some(format!("{}{}", s, op_s)),
                            None => {
                                Some(format!("{}({})", expr_graph.op_store[*operator].name, op_s))
                            }
                        }
                    }
                    Op2 { operator, ops } => {
                        let op0 = self.pp_node(ops[0], expr_graph);
                        let op1 = self.pp_node(ops[1], expr_graph);
                        match &expr_graph.op_store[*operator].infix {
                            Some(s) => Some(format!("{} {} {}", op0, s, op1)),
                            None => Some(format!(
                                "{} {} {}",
                                op0, expr_graph.op_store[*operator].name, op1
                            )),
                        }
                    }
                };
                if let Some(expr_s) = expr_s {
                    res.push(format!("{:?} = {}", nid, expr_s));
                }
            }
        }
        res.join("\n")
    }
}
