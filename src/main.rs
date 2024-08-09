#![feature(fn_traits, associated_type_defaults, const_type_id, ptr_metadata, entry_insert)]
#![allow(unused_imports, dead_code, non_upper_case_globals)]

mod tracer;
mod repr;
mod pretty;

use repr::{layout, Layout};
use tracer::{Tracer, TracerError};
use std::rc::Rc;
use std::collections::HashMap;
use thiserror::Error;

use rangemap::RangeMap;
use yaxpeax_x86::long_mode::{Instruction, Opcode, Operand, MemoryAccessSize, RegSpec};
use sleigh::VarnodeData;

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

struct Type { layout: repr::Layout, offset: u8 }

use std::borrow::Cow;
enum Value {
    Unknown,
    Known(Type),
    Fixed(Type, Cow<'static, [u8]>),
}

struct Space {
    mem: RangeMap<usize, Value>,
}

struct VarnodeContext {
    spaces: HashMap<String, Space>,
    unique: HashMap<usize, VarnodeData>,
}

impl VarnodeContext {
    fn new() -> Self {
        Self {
            spaces: Default::default(),
            unique: Default::default(),
        }
    }

    fn copy(&mut self, output: &VarnodeData, input: &VarnodeData) {
        if output.space.name == "unique" {
            self.unique.entry((output.offset as usize)).insert_entry(input.clone());
        }
    }
}

struct Witchcraft {
    values: HashMap<RegSpec, Value>,
    context: VarnodeContext,
}


impl Witchcraft {
    fn eval(&mut self, op: Operand, ms: Option<MemoryAccessSize>) {
        println!("{}", op);
    }

    fn jit<F>(&mut self, f: Box<F>) -> Result<Box<F>, Box<dyn std::error::Error>> where
        F: Closure<Env, Item> + ?Sized
    {
        let call = F::call;
        let mut tracer = Tracer::new().unwrap();
        let sym = tracer.get_symbol_from_address(call as *const ())?;
        let instructions = tracer.disassemble(call as *const (), sym.st_size as usize)?;
        tracer.format(&instructions)?;

        let tokens = unsafe { std::slice::from_raw_parts(call as *const u8, sym.st_size as usize) };
        println!("{:x}", call as u64);

        use sleigh::{Decompiler, X86Mode};
        let mut decompiler = Decompiler::builder().x86(X86Mode::Mode64).build();
        let (len, sleigh) = decompiler.translate(tokens, call as u64);

        println!("{:?}", sleigh);

        for inst in sleigh.iter() {
            dbg!(inst);
            match (&inst.opcode, &inst.vars[..], inst.outvar.as_slice()) {
                (sleigh::Opcode::Copy, [input, ..], [output, ..]) => {
                    println!("copy {} -> {}", input, output);
                    self.context.copy(output, input);
                },
                (sleigh::Opcode::IntSub, [left, right], [out]) => {
                    println!("intsub {} {} -> {}", left, right, out);
                },
                (sleigh::Opcode::Store, [spaceid, ptr, val], []) => {
                    //let addrspace = decompiler.getAddressSpace(spaceid.offset as i32);
                    let addrspace: *const sleigh_sys::ffi::AddrSpace = spaceid.offset as _;
                    let addrspace_name = unsafe { addrspace.as_ref().unwrap().getName() };
                    println!("store {}:[{}] <- {}", addrspace_name, ptr, val);
                }
                _ => unreachable!()
            }
        }

        Ok(f)
    }

    fn speedup<F>(f: Box<F>) -> Result<Box<F>, Box<dyn std::error::Error>> where
        F: Closure<Env, Item> + ?Sized
    {
        let [obj, vtable] = unsafe { *(f.as_ref() as *const _ as *const [*const (); 2]) };
        dbg!(obj, vtable);
        // TODO: get the size of the vtable from symbols/dwarf? hardcode it to 0x20 instead? idk
        let vtable_slice = unsafe { std::slice::from_raw_parts(vtable as *const u8, 0x1000) };


        let mut witch = Witchcraft {
            values: vec![
                // stupid vtable hack
                (RegSpec::rdi(), Value::Fixed(Type {
                    layout: Layout(layout::<*const ()>(0)),
                    offset:0
                }, Cow::Borrowed(vtable_slice))),
                //(RegSpec::rsi(), Value::Fixed(vtable)),
                (RegSpec::rsi(), Value::Known(Type {
                    layout: Layout(layout::<Add::Ty>(0)),
                    offset: 0
                })),
            ].drain(..).collect(),
            context: VarnodeContext::new(),
        };
        let add = witch.jit(f);

        return add
    }
}

fn main() {
    let one: Rc<Function> = Rc::new(One::Ty());
    let inc = Box::new(Inc::Ty(one));

    let mut env = Env {
        values: vec![("a".into(), Item::Num(5))].drain(..).collect()
    };
    let inc = Witchcraft::speedup(inc).unwrap();

    println!("add: {:?}", inc.call(&mut env));
}
