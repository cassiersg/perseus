use rand::seq::IndexedRandom;
use rand::SeedableRng;
use rand_distr::Distribution;

use pyo3::prelude::*;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use std::sync::Arc;
pub(crate) mod favom;
pub(crate) mod progress_bar;

/// Sample probes given the number of probes in each gadget.
#[pyclass]
#[derive(Clone)]
struct ProbeSampler {
    /// Outer Vec corresponds to gadgets
    /// Middle Vec corresponds to probes in a gadget
    /// Inner Vec corresponds to Expr in a single extended probe
    gadgets_eprobes: Vec<Vec<Vec<favom::Expr>>>,
}
impl ProbeSampler {
    fn sample_probes<R: rand::Rng>(
        &self,
        rng: &mut R,
        n_probes: Vec<u32>,
    ) -> Vec<favom::expr::ExprId> {
        // Need at least that size, more if gate leakage model.
        let mut res = Vec::with_capacity(n_probes.iter().map(|x| *x as usize).sum());
        for (exprs, n_probes) in self.gadgets_eprobes.iter().zip(n_probes.iter()) {
            res.extend(
                exprs
                    .choose_multiple(rng, *n_probes as usize)
                    .flat_map(|eprobe| eprobe.iter().map(|e| e.0)),
            );
        }
        res
    }
}
#[pymethods]
impl ProbeSampler {
    #[new]
    pub fn new(gadgets_eprobes: Vec<Vec<Vec<favom::Expr>>>) -> ProbeSampler {
        Self { gadgets_eprobes }
    }
}
#[cfg(debug_assertions)]
const RELEASE: bool = false;

#[cfg(not(debug_assertions))]
const RELEASE: bool = true;

#[pyfunction]
fn is_release() -> bool {
    RELEASE
}

use rand_distr::weighted::WeightedAliasIndex as ChoiceDistr;

/// For a set of gadgets bound by an Inequality, sample the number of probes in thoss gadgets,
/// conditioned on violating the inequality.
struct JointSizesSamplerInner {
    idx_distr: ChoiceDistr<f64>,
    idx2sizes: Vec<Vec<u32>>,
}

impl JointSizesSamplerInner {
    fn new(weights: Vec<f64>, sizes: Vec<Vec<u32>>) -> Self {
        Self {
            idx_distr: ChoiceDistr::new(weights).unwrap(),
            idx2sizes: sizes,
        }
    }
    fn sample<R: rand::Rng>(&self, rng: &mut R) -> &Vec<u32> {
        let idx = self.idx_distr.sample(rng);
        &self.idx2sizes[idx]
    }
}

#[pyclass]
#[derive(Clone)]
struct JointSizesSampler(Arc<JointSizesSamplerInner>);
#[pymethods]
impl JointSizesSampler {
    #[new]
    fn new(weights: Vec<f64>, sizes: Vec<Vec<u32>>) -> Self {
        Self(Arc::new(JointSizesSamplerInner::new(weights, sizes)))
    }
    fn sample(&mut self, seed: u64) -> Vec<u32> {
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(seed);
        self.0.sample(&mut rng).clone()
    }
}

/// Sample the number of probes in each gadget possiblty conditioned on the failure of the SNI
/// composition proof strategy.
#[pyclass]
struct TupleSizeSampler {
    joint_size_samplers: Vec<Arc<JointSizesSamplerInner>>,
    gadget_n_leak: Vec<u32>,
    ineq2gidx: Vec<Vec<usize>>,
    ineq_t: Vec<u32>,
    ineq_distr: ChoiceDistr<f64>,
    size_distributions: Vec<rand_distr::Binomial>,
}

impl TupleSizeSampler {
    fn sample_in<const N: usize, R: rand::Rng>(&self, rng: &mut R, res: &mut [[u32; N]]) {
        for (res, distr) in res.iter_mut().zip(self.size_distributions.iter()) {
            for res in res.iter_mut() {
                *res = distr.sample(rng).try_into().unwrap();
            }
        }
    }
    fn n_violated_inequalities<const N: usize>(&self, sizes: &[[u32; N]]) -> [u64; N] {
        let mut res = [0; N];
        for (idxs, t) in self.ineq2gidx.iter().zip(self.ineq_t.iter()) {
            let mut tot = [0; N];
            for idx in idxs {
                for i in 0..N {
                    tot[i] += sizes[*idx][i];
                }
            }
            for i in 0..N {
                if tot[i] > *t {
                    res[i] += 1;
                }
            }
        }
        res
    }
    fn sample_violations<const N: usize, R: rand::Rng>(&self, rng: &mut R) -> u64 {
        let mut tmp = vec![[0; N]; self.gadget_n_leak.len()];
        self.sample_in::<N, _>(rng, &mut tmp);
        self.n_violated_inequalities(&tmp)
            .into_iter()
            .filter(|x| *x != 0)
            .count()
            .try_into()
            .unwrap()
    }
    fn sample_ineq_inner<R: rand::Rng>(&self, rng: &mut R) -> usize {
        self.ineq_distr.sample(rng)
    }
    fn sample_sizes_cond_inner<R: rand::Rng>(&self, rng: &mut R) -> (Vec<u32>, u64) {
        let mut tmp = vec![[0; 1]; self.gadget_n_leak.len()];
        let mut n_rej = 0;
        loop {
            self.sample_in::<1, _>(rng, &mut tmp);
            let ineq = self.sample_ineq_inner(rng);
            for (j, s) in self.ineq2gidx[ineq]
                .iter()
                .zip(self.joint_size_samplers[ineq].sample(rng).iter())
            {
                tmp[*j][0] = *s;
            }
            let violated = self.n_violated_inequalities(&tmp);
            if rand::distr::Bernoulli::from_ratio(1, violated[0].try_into().unwrap())
                .unwrap()
                .sample(rng)
            {
                let tmp = tmp.into_iter().map(|x| x[0]).collect::<Vec<_>>();
                return (tmp, n_rej);
            }
            n_rej += 1;
        }
    }
}

#[pymethods]
impl TupleSizeSampler {
    #[new]
    pub fn new(
        p: f64,
        gadget_n_leak: Vec<u32>,
        ineq2gidx: Vec<Vec<usize>>,
        ineq_t: Vec<u32>,
        ineq_weights: Vec<f64>,
        joint_size_samplers: &Bound<'_, pyo3::types::PyList>,
    ) -> Self {
        let joint_size_samplers = joint_size_samplers
            .iter()
            .map(|e| e.extract::<JointSizesSampler>().unwrap().0)
            .collect::<Vec<_>>();
        Self {
            size_distributions: gadget_n_leak
                .iter()
                .map(|n| rand_distr::Binomial::new((*n).into(), p).unwrap())
                .collect(),
            gadget_n_leak,
            ineq2gidx,
            ineq_t,
            ineq_distr: ChoiceDistr::new(ineq_weights).unwrap(),
            joint_size_samplers,
        }
    }
    pub fn sample_ineq_violation(&self, n_samples: u64, seed: u64) -> u64 {
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(seed);
        const N: usize = 8;
        let rem = n_samples % (N as u64);
        let res_rem: u64 = (0..rem)
            .map(|_| self.sample_violations::<1, _>(&mut rng))
            .sum();
        // TODO: add min size (e.g. 10*num_cores)
        let res_main: u64 = crate::progress_bar::with_progress(
            |it_cnt| {
                it_cnt.inc(rem);
                (0..(n_samples / (N as u64)))
                    .into_par_iter()
                    .map(|i| {
                        let mut rng = rng.clone();
                        rng.advance((i as u128) << 64);
                        let res = self.sample_violations::<N, _>(&mut rng);
                        it_cnt.inc(N as u64);
                        res
                    })
                    .sum()
            },
            n_samples,
            "sample_ineq_violation",
            std::time::Duration::from_millis(500),
        );
        res_main + res_rem
    }
    pub fn sample_sizes_cond(&self, seed: u64) -> (Vec<u32>, u64) {
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(seed);
        self.sample_sizes_cond_inner(&mut rng)
    }
}

#[pyfunction]
fn check_tuples_sample_probes(
    exprs: &favom::FavomExprs,
    probe_sampler: &ProbeSampler,
    tuples_sampler: &TupleSizeSampler,
    n_samples: u64,
    seed: u64,
) -> (u64, u64) {
    let rng = rand_pcg::Pcg64Mcg::seed_from_u64(seed);
    crate::progress_bar::with_progress(
        |it_cnt| {
            (0..(n_samples as usize))
                .into_par_iter()
                .with_min_len(std::thread::available_parallelism().unwrap().get() * 16)
                .map(|i| {
                    let mut rng = rng.clone();
                    rng.advance((i as u128) << 64);
                    let (n_probes, n_rej) = tuples_sampler.sample_sizes_cond_inner(&mut rng);
                    let top_exprs = probe_sampler.sample_probes(&mut rng, n_probes);
                    let res = if favom::check_tuple(exprs, top_exprs) {
                        0
                    } else {
                        1
                    };
                    it_cnt.inc(1);
                    (res, n_rej)
                })
                .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
        },
        n_samples,
        "Favom tuples",
        std::time::Duration::from_millis(500),
    )
}

#[pymodule]
fn rbackend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<favom::FavomExprs>()?;
    m.add_class::<favom::FavomState>()?;
    m.add_class::<favom::Expr>()?;
    m.add_class::<favom::Secret>()?;
    m.add_class::<JointSizesSampler>()?;
    m.add_class::<ProbeSampler>()?;
    m.add_class::<TupleSizeSampler>()?;
    m.add_function(wrap_pyfunction!(check_tuples_sample_probes, m)?)?;
    m.add_function(wrap_pyfunction!(favom::check_tuples, m)?)?;
    m.add_function(wrap_pyfunction!(is_release, m)?)?;
    Ok(())
}
