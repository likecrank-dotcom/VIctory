//! Read-only barrier audit using the candidate heuristic from FibbS6 1052a90.
//! Reports potential control dependence, not a proof of a GPU deadlock.
use std::collections::HashSet;
use victory_shader::{Container, FourCC, ir, structure, native_programs};
use structure::{Atom, LaneExit};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path)?;
        let c = Container::parse(&bytes)?;
        let dxil = c.part(FourCC::DXIL).ok_or("missing DXIL")?;
        for native in native_programs(dxil.data)? {
            let mut p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
            victory_shader::sgpr_spill::forward(&mut p);
            let all: Vec<_> = p.blocks.iter().enumerate().filter(|(_, b)| b.insts.iter().any(|i| i.name()=="s_barrier")).map(|(i,b)|(i,b.offset)).collect();
            let st = structure::structure(&p)?;
            let suspects = divergent_barriers(&p, &st, &HashSet::new());
            let console = structure::structure_console(&p)?;
            let console_suspects = divergent_barriers(&p, &console, &HashSet::new());
            println!("{} stage={:?} wave64={} blocks={} barrier_blocks={:?} flagged_normal={:?} flagged_console={:?}", path, native.stage, native.wave64, p.blocks.len(), all, suspects, console_suspects);
        }
    }
    Ok(())
}

fn divergent_barriers(
    p: &ir::Program,
    st: &structure::Structured,
    skip: &HashSet<usize>,
) -> HashSet<usize> {
    let n = p.blocks.len();
    let barriers: Vec<usize> = (0..n)
        .filter(|b| {
            !skip.contains(b)
                && p.blocks[*b]
                    .insts
                    .iter()
                    .any(|i| i.name() == "s_barrier")
        })
        .collect();
    if barriers.is_empty() {
        return HashSet::new();
    }
    let divergent: Vec<(usize, usize)> = (0..n)
        .filter_map(|u| match st.exits[u] {
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } if taken != not_taken
                && st
                    .bools
                    .support(when)
                    .iter()
                    .any(|a| !matches!(a, Atom::Uniform(_))) =>
            {
                Some((taken, not_taken))
            }
            _ => None,
        })
        .collect();
    if divergent.is_empty() {
        return HashSet::new();
    }
    let ends: Vec<usize> = (0..n)
        .filter(|&b| matches!(st.exits[b], LaneExit::End))
        .collect();
    let mut out = HashSet::new();
    for b in barriers {
        let mut can_reach = vec![false; n];
        let mut stack = vec![b];
        can_reach[b] = true;
        while let Some(cur) = stack.pop() {
            for &pred in &p.blocks[cur].preds {
                if !can_reach[pred] {
                    can_reach[pred] = true;
                    stack.push(pred);
                }
            }
        }
        let mut can_bypass = vec![false; n];
        for &e in &ends {
            if e != b && !can_bypass[e] {
                can_bypass[e] = true;
                stack.push(e);
            }
        }
        while let Some(cur) = stack.pop() {
            for &pred in &p.blocks[cur].preds {
                if pred != b && !can_bypass[pred] {
                    can_bypass[pred] = true;
                    stack.push(pred);
                }
            }
        }
        if divergent
            .iter()
            .any(|&(t, f)| (can_reach[t] && can_bypass[f]) || (can_reach[f] && can_bypass[t]))
        {
            out.insert(b);
        }
    }
    out
}
