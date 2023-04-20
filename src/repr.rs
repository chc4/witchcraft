use core::any::TypeId;
use lazy_static::lazy_static;
use memoffset::{offset_of, offset_of_tuple};

use linkme::distributed_slice;

#[distributed_slice]
pub static KNOWN: [(TypeId, LayoutFn)] = [..];

#[derive(Clone, Debug)]
pub struct Member {
    offset: usize,
    ty: TypeId,
}

#[derive(Debug)]
pub struct Layout(Vec<Member>);

pub trait Repr: 'static {
    fn known(&self) -> bool { false }
    const LAYOUT: (TypeId, fn(usize)->Vec<Member>);
}

type LayoutFn = fn(usize) -> Vec<Member>;

use std::sync::Mutex;
lazy_static! {
    static ref LAYOUTS: std::sync::Mutex<std::collections::HashMap<TypeId, LayoutFn>> = {
        Mutex::new(KNOWN.to_vec().into_iter().collect())
    };
}

pub fn layout<A>(offset: usize) -> Vec<Member> where A: 'static {
    println!("layout of {}", core::any::type_name::<A>());
    let f = if let Some(f) = LAYOUTS.lock().unwrap().get(&TypeId::of::<A>()) {
        f.clone()
    } else {
        return vec![Member { offset, ty: TypeId::of::<A>() }]
    };
    f(offset)
}

#[macro_export]
macro_rules! inspect {
    ($ty:ident, [$($mem:ty),*]) => {
        mod $ty {
            use linkme::distributed_slice;
            use crate::repr::{KNOWN, Member, layout};
            use core::any::TypeId;
            #[repr(C)]
            pub struct Ty($(pub $mem),*);

            #[distributed_slice(KNOWN)]
            static Lay: (TypeId, fn(offset: usize)->Vec<Member>) = (TypeId::of::<Ty>(), |offset|{
                println!("asf {}", offset);
                let fields: &[core::alloc::Layout] = &[$(core::alloc::Layout::new::<$mem>()),*];
                let mut curr = core::alloc::Layout::from_size_align(0, 1).unwrap();
                let mut memb_off = 0;
                let mut i = 0;
                (vec![
                    $(
                        {
                            let (new_layout, start_offset) = curr.extend(fields[i]).unwrap();
                            memb_off += start_offset;
                            let m = layout::<$mem>(offset+memb_off);
                            i += 1;
                            curr = new_layout;
                            m
                        }
                    ),*
                ] as Vec<Vec<Member>>).concat()
            });
        }
    }
}

inspect![Bar, [u8, u16]];
inspect![Foo, [usize, crate::repr::Bar::Ty]];

mod test {
    use crate::repr::{Repr, layout, Foo};

    #[test]
    fn simple_layout() {
        dbg!(layout::<Foo::Ty>(0));
    }
}
