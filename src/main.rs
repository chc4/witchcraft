#![feature(fn_traits, associated_type_defaults, specialization, const_type_id)]

mod tracer;
mod repr;
use tracer::{Tracer, TracerError};
use std::rc::Rc;
use std::collections::HashMap;
use thiserror::Error;

use yaxpeax_x86::long_mode::{Instruction, Opcode, Operand, MemoryAccessSize, RegSpec};


struct Env {
    values: HashMap<String, Item>,
}
type Function = dyn Closure<Env, Item>;

trait Closure<IN, OUT> {
    fn call(&self, _: &mut IN) -> OUT;
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub struct Item(u32);

pub struct One;
impl Closure<Env, Item> for One {
    fn call(&self, _: &mut Env) -> Item {
        return Item(1);
    }
}

pub struct Get(String);
impl Closure<Env, Item> for Get {
    fn call(&self, env: &mut Env) -> Item {
        return env.values.get(&self.0).unwrap().clone()
    }
}

pub struct Add { left: Rc<Function>, right: Rc<Function> }
impl Closure<Env, Item> for Add {
    fn call(&self, env: &mut Env) -> Item {
        return Item(self.left.call(env).0 + self.right.call(env).0)
    }
}

enum Value {
    Unknown,
    Known(repr::Layout)
}

struct Witchcraft {
    values: HashMap<RegSpec, Value>,
}


impl Witchcraft {
    // TODO: this needs a size
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

        for inst in instructions.iter() {
            match inst.opcode() {
                Opcode::MOV => {
                    let right = self.eval(inst.operand(1), inst.mem_size());
                },
                _ => unreachable!()
            }
        }

        Ok(f)
    }
}

fn main() {
    let one: Rc<Function> = Rc::new(One);
    let get: Rc<Function> = Rc::new(Get("a".into()));
    let add: Box<Function> = Box::new(Add { left: one, right: get });

    let mut env = Env {
        values: vec![("a".into(), Item(5))].drain(..).collect()
    };


    let mut witch = Witchcraft {
        values: Default::default()
    };
    let add = witch.jit(add).unwrap();

    println!("add: {:?}", add.call(&mut env));
}
