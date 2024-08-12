#![feature(fn_traits, associated_type_defaults, const_type_id, ptr_metadata, entry_insert, let_chains)]
#![allow(unused_imports, dead_code, non_upper_case_globals)]

mod tracer;
mod repr;

use repr::{layout, Layout};
use tracer::{Tracer, TracerError};
use std::rc::Rc;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

use rangemap::RangeMap;
use sleigh::{VarnodeData, Decompiler, X86Mode, Opcode, SpaceType};

pub struct Env {
    values: HashMap<String, Item>,
}
pub type Function = dyn Closure<Env, Item>;

pub trait Closure<IN, OUT> {
    extern "C" fn call(&self, _: &mut IN) -> OUT;
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum Item {
    Object(Box<Item>),
    Num(u32),
}


inspect![One, []];
impl Closure<Env, Item> for One::Ty {
    extern "C" fn call(&self, _: &mut Env) -> Item {
        return Item::Num(1);
    }
}


// with usize
//mov rax, qword [rdi + 0x8]
//mov rcx, qword [rdi + 0x10]
//mov rdx, qword [rcx + 0x10]
// without
//mov rax, qword [rdi]
//mov rcx, qword [rdi + 0x8]
//mov rdx, qword [rcx + 0x10]
// because the Rc is a fat pointer right woops
inspect![Inc, [crate::Rc<crate::Function>]];
impl Closure<Env, Item> for Inc::Ty {
    extern "C" fn call(&self, env: &mut Env) -> Item {
        match self.0.call(env) {
            Item::Num(a) => Item::Num(a + 1),
            _ => unreachable!()
        }
    }
}

inspect![Get, [String]];
impl Closure<Env, Item> for Get::Ty {
    extern "C" fn call(&self, env: &mut Env) -> Item {
        return env.values.get(&self.0).unwrap().clone()
    }
}

inspect![Add, [crate::Rc<crate::Function>, crate::Rc<crate::Function>]];
impl Closure<Env, Item> for Add::Ty {
    extern "C" fn call(&self, env: &mut Env) -> Item {
        match (self.0.call(env), self.1.call(env)) {
            (Item::Num(a), Item::Num(b)) => {
                Item::Num(a + b)
            },
            _ => unimplemented!()
        }
    }
}

#[derive(Clone, Hash, Debug, Eq, PartialEq)]
struct Type { layout: repr::Layout, offset: u8 }

use std::borrow::Cow;
#[derive(Clone, Hash, Debug, Eq, PartialEq)]
enum Value {
    Unknown,
    Known(VarnodeData),
    Fixed(Type, Cow<'static, [u8]>),
}

#[derive(Debug, Clone)]
struct Spell {
    pcode: sleigh::PCode,
}

impl std::hash::Hash for Spell {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // we explicitly ignore the address.
        state.write_u32(self.pcode.opcode as u32);
        self.pcode.vars.iter().map(|var| var.hash(state)).for_each(drop);
        self.pcode.outvar.as_ref().map(|var| var.hash(state));
    }
}

impl PartialEq for Spell {
    fn eq(&self, other: &Self) -> bool {
        // we explicitly ignore the address
        self.pcode.opcode.eq(&other.pcode.opcode) &&
        self.pcode.vars.eq(&other.pcode.vars) &&
        self.pcode.outvar.eq(&other.pcode.outvar)
    }
}

impl Eq for Spell {}

struct Witchcraft {
    decompiler: Decompiler,
    /// Code rewriting window
    peephole: lru::LruCache<Spell, u8>,
    /// Code materializing window
    materialize: Vec<Spell>,
    /// Finalized code to emit
    emitted_code: Vec<sleigh::PCode>,
    /// Live values -> (emitted, spell, dependants)
    context: HashMap<VarnodeData, (bool, Value, Spell, HashSet<Spell>)>,
}

impl Witchcraft {
    fn jit<F>(&mut self, f: Box<F>) -> Result<Box<F>, Box<dyn std::error::Error>> where
        F: Closure<Env, Item> + ?Sized
    {
        let call = F::call;
        let mut tracer = Tracer::new().unwrap();
        let sym = tracer.get_symbol_from_address(call as *const ())?;

        let tokens = unsafe { std::slice::from_raw_parts(call as *const u8, sym.st_size as usize) };
        println!("{:x}", call as u64);

        Ok(f)
    }

    fn simplify(&mut self, inst: sleigh::PCode) -> (Value, Spell) {
        let size_mismatch = inst.vars.iter().any(|var| inst.outvar.as_ref().map(|ourvar| ourvar.size != var.size).unwrap_or(true));
        'a:  { match (inst.opcode, &inst.vars[..]){
            (Opcode::Copy, [input]) => {
                if size_mismatch { break 'a };
                if input.space.type_ == SpaceType::Constant {
                    return (Value::Known(input.clone()), Spell { pcode: inst })
                }
                if let Some(exists) = self.context.get(input) {
                    return (exists.1.clone(), exists.2.clone())
                }
            },
            (Opcode::IntAdd | Opcode::IntAnd, [left, right])
                if right.space.type_ == SpaceType::Constant
            => {
                if size_mismatch { break 'a };
                if let Some(left) = self.context.get_mut(left) {
                    if let Value::Known(lvar) = &mut left.1 && lvar.space.type_ == SpaceType::Constant {
                        let const_val = match inst.opcode {
                            Opcode::IntAdd => { lvar.offset + right.offset },
                            Opcode::IntAnd => { (lvar.offset&right.offset) as u64 },
                            _ => panic!(),
                        };
                        lvar.offset = const_val;
                        let new_const = sleigh::PCode {
                            address: 0,
                            opcode: Opcode::Copy,
                            vars: vec![lvar.clone()],
                            outvar: inst.outvar
                        };
                        return (left.1.clone(), Spell { pcode: new_const });
                    }
                }
            }
            _ => {}
        }};
        (Value::Unknown, Spell { pcode: inst })
    }

    fn push_peephole(&mut self, inst: sleigh::PCode) {
        // Try to use an existing instruction if it already exists
        // This is not-quite GVN, since we need the exact same instruction form.
        // For example, we can't de-duplicate the same instruction but output
        // to a different register
        //
        // Peephole invariants are:
        // 1) if we would add an operation that depends
        //    on a value in the peephole, it has to be materialized.
        //    -> this means we never have to recursively materialize spells
        // 2) if we would add an operation that produces a varnode in our context,
        //    all values in the peephole that depend on it have to be materialized
        //    -> this means spells mentioning inputs by-name is safe
        // Additionally, 1 means that removing values from the peephole is safe,
        // because no spell can depend on its output.
        let (val, spell) = self.simplify(inst);
        let mut to_materialize = vec![];
        //let mut to_remove = vec![];
        for dep in &spell.pcode.vars {
            if dep.space.type_ == SpaceType::Constant { continue; }
            if let Some(computes) = self.context.get_mut(dep) {
                // track that this spell is a dependant of its inputs
                computes.3.insert(spell.clone());
                // breaks invariant 1, materialize input but we still need to track uses for 2
                if computes.0 == false {
                    computes.0 = true;
                    to_materialize.push(computes.clone());
                }
            } else {
                //println!("unbound input {}", self.pretty(dep));
            }
        }
        for (emitted, e_val, mat, uses) in to_materialize.drain(..) {
            println!("- materializing {}", self.prettySpell(&mat));
            mat.pcode.outvar.as_ref().map(|outvar|
                self.context.get_mut(&outvar).map(|u| u.0 = true) );
            self.materialize.push(mat);
        }
        spell.pcode.outvar.as_ref().map(|outvar| {
            // breaks invariant 2, materialize dependants
            if let Some((emitted, e_val, exists, uses)) = self.context.insert(outvar.clone(),
                (false, val, spell.clone(), Default::default()))
            {
                if emitted {
                    // if we already emitted it, then it has dependants
                    // if the instruction names an output the same as an input, we don't want to emit it
                    for dependant in uses {
                        if dependant == spell { continue }
                        println!("-- mat old use {}", self.prettySpell(&dependant));
                        let mut needs_mat = false;
                        dependant.pcode.outvar.as_ref().map(|ourvar|
                            self.context.get_mut(&ourvar).map(|u| {
                                if u.0 == false {
                                    needs_mat = true;
                                    u.0 = true
                                }
                            }) );
                        if needs_mat {
                            self.materialize.push(dependant);
                        }
                    }
                } else {
                    // else we don't have any dependants, but remove this instruction
                    // from all spells's uses (since it won't ever be visible)
                    for used in &exists.pcode.vars {
                        if used.space.type_ == SpaceType::Constant { continue }
                        println!("-- remove as use from {}", self.pretty(&used));
                        self.context.get_mut(&used).map(|u| u.3.remove(&exists));
                    }
                }
            }
        });
    }

    fn analyze(&mut self, tokens: &[u8], start: usize) -> Result<(), Box<dyn std::error::Error>>{
        let mut tracer = Tracer::new().unwrap();
        let mut off = 0;
        let mut pcode_count = 0;
        while(off < tokens.len()) {

            let pc = (start + off);
            let (len, mut sleigh) = self.decompiler.translate(&tokens[off..], pc as u64);

            let instructions = tracer.disassemble(
                tokens[off..].as_ptr() as *const (), len)?;
            println!("----------");
            tracer.format(&instructions)?;

            for inst in sleigh.drain(..) {
                //dbg!(inst);
                match (&inst.opcode, &inst.vars[..], inst.outvar.as_slice()) {
                    (sleigh::Opcode::Copy, [input], [output]) => {
                        println!("copy {} -> {}",
                            self.pretty(input), self.pretty(output));
                    },
                    // One operand bitwise operations
                    (sleigh::Opcode::BoolNegate, [var], [out]) => {
                        println!("{:?} {} -> {}", inst.opcode,
                            self.pretty(var), self.pretty(out));
                    },
                    // One operand numeric operations
                    (sleigh::Opcode::PopCount | sleigh::Opcode::IntZExt, [var], [count]) => {
                        println!("{:?} {} -> {}", inst.opcode,
                            self.pretty(var), self.pretty(count));
                    },
                    // Two operand numeric operations
                    (sleigh::Opcode::IntAdd | sleigh::Opcode::IntSub | sleigh::Opcode::IntMult
                     | sleigh::Opcode::IntLess | sleigh::Opcode::IntSBorrow | sleigh::Opcode::IntCarry
                     | sleigh::Opcode::IntSCarry
                     | sleigh::Opcode::IntSLess | sleigh::Opcode::IntEqual
                     | sleigh::Opcode::IntAnd | sleigh::Opcode::IntOr | sleigh::Opcode::IntXor,
                     [left, right], [out]) => {
                        println!("{:?} {} {} -> {}", inst.opcode,
                            self.pretty(left), self.pretty(right), self.pretty(out));
                    },
                    (sleigh::Opcode::Store, [spaceid, ptr, val], []) => {
                        //let addrspace = decompiler.getAddressSpace(spaceid.offset as i32);
                        let addrspace: *const sleigh_sys::ffi::AddrSpace = spaceid.offset as _;
                        let addrspace_name = unsafe { addrspace.as_ref().unwrap().getName() };
                        println!("store {}:[{}] <- {}", addrspace_name,
                            self.pretty(ptr),
                            self.pretty(val)
                        );
                    },
                    (sleigh::Opcode::Load, [spaceid, ptr], [val]) => {
                        let addrspace: *const sleigh_sys::ffi::AddrSpace = spaceid.offset as _;
                        let addrspace_name = unsafe { addrspace.as_ref().unwrap().getName() };
                        println!("load {}:[{}] -> {}", addrspace_name, self.pretty(ptr), self.pretty(val));
                    },
                    // Control flow
                    (sleigh::Opcode::CallInd, [ptr], []) => {
                        println!("callind {}", self.pretty(ptr));
                    },
                    (sleigh::Opcode::Call, [target], []) => {
                        println!("call {}", self.pretty(target));
                    },
                    (sleigh::Opcode::CallOther, args, ret) => {
                        println!("callother {:?} {:?}", args, ret);
                    },
                    (sleigh::Opcode::Return, [reg], []) => {
                        println!("return {}", self.pretty(reg));
                    },
                    (sleigh::Opcode::Branch, [target], []) => {
                        println!("branch {}", self.pretty(target));
                    },
                    (sleigh::Opcode::CBranch, [cond, target], []) => {
                        println!("cbranch {} {}",
                            self.pretty(cond),
                            self.pretty(target),
                        );
                    },
                    x => unreachable!("unimplemented pcode op {:?}", x)
                }
                self.push_peephole(inst);
                pcode_count += 1;
            }
            // unique space varnodes aren't ever used across instructions, so we
            // can clear them from our context in order to avoid materializing unused
            // results
            off += len;
        }
        println!("saw {} pcode ops", pcode_count);
        Ok(())
    }

    fn materialize(&mut self, loc: &VarnodeData) {
        // We have an instruction that is referencing an instruction that we need to materialize:
        // we walk the data dependency chain for our location, adding the instructions to the
        // peephole if we don't already have them emitted. If possible, we try to rewrite
        // instructions in the peephole for constant folding, and flush the peephole buffer at the
        // end. Whenever an instruction falls out we try to emit the instruction that fell out.
        // This means that we only rewrite intermediate products that haven't been reference by any
        // other materialized values yet.
        // TODO: how do we deal with clobbers? do we just say these materialized instructions are
        // unordered, and then have instruction scheduling after? probably yeah
    }

    fn speedup<F>(f: Box<F>) -> Result<Box<F>, Box<dyn std::error::Error>> where
        F: Closure<Env, Item> + ?Sized + 'static
    {
        let [obj, vtable] = unsafe { *(f.as_ref() as *const _ as *const [*const (); 2]) };
        dbg!(obj, vtable);
        // TODO: get the size of the vtable from symbols/dwarf? hardcode it to 0x20 instead? idk
        let vtable_slice = unsafe { std::slice::from_raw_parts(vtable as *const u8, 0x1000) };


        let mut witch = Witchcraft::new();
            //values: vec![
            //    // stupid vtable hack
            //    (RegSpec::rdi(), Value::Fixed(Type {
            //        layout: Layout(layout::<*const ()>(0)),
            //        offset:0
            //    }, Cow::Borrowed(vtable_slice))),
            //    //(RegSpec::rsi(), Value::Fixed(vtable)),
            //    (RegSpec::rsi(), Value::Known(Type {
            //        layout: Layout(layout::<F>(0)),
            //        offset: 0
            //    })),
            //].drain(..).collect(),
            //context: VarnodeContext::new(),
        let add = witch.jit(f);

        return add
    }

    fn pretty(&self, var: &VarnodeData) -> String {
        self.decompiler.format_varnode(var)
    }

    fn prettySpell(&self, spell: &Spell) -> String {
        format!("{:?} [{}] {}",
            spell.pcode.opcode,
            spell.pcode.vars.iter().map(|var| self.pretty(var)).collect::<Vec<_>>().join(", "),
            spell.pcode.outvar.as_ref().map(|var| self.pretty(&var)).unwrap_or_else(|| "".into()))
    }

    fn new() -> Self {
        let mut decompiler = Decompiler::builder().x86(X86Mode::Mode64).build();
        Self {
            decompiler,
            peephole: lru::LruCache::new(std::num::NonZeroUsize::new(32).unwrap()),
            materialize: Vec::new(),
            emitted_code: Default::default(),
            context: Default::default(),
        }
    }
}

fn main() {
    //let one: Rc<Function> = Rc::new(One::Ty());
    //let inc = Box::new(Inc::Ty(one.clone()));

    //let mut env = Env {
    //    values: vec![("a".into(), Item::Num(5))].drain(..).collect()
    //};
    //let inc = Witchcraft::speedup(inc).unwrap();

    //println!("add: {:?}", inc.call(&mut env));
    let testcase = &[
        72, 199, 193, 0, 0, 0, 0, // XOR RCX, RCX
        72, 131, 193, 2, // add rcx, 2
        72, 131, 195, 2, // add rbx, 2
        72, 131, 193, 2, // add rcx, 2
    ];
    let mut witchcraft = Witchcraft::new();
    witchcraft.analyze(testcase, 0);
    // force materialization of the output registers we want
    //witchcraft.add_external(VarnodeData { space: );
    println!("{:?}", witchcraft.peephole);
    for (varnode, spell) in &witchcraft.context {
        println!("{} - {}", witchcraft.pretty(&varnode), witchcraft.prettySpell(&spell.2));
    }
    for inst in &witchcraft.materialize {
        println!("mat {}", witchcraft.prettySpell(inst));
    }
    println!("materialized {} pcode ops", witchcraft.materialize.len());
}
