#define __libmd_src_file__
#ifndef libmd_h
#include "../libmd.h"
#endif

template<ui dim> particle<dim>::particle(ldf mass,ui ptype,bool fixed,bool pbconded)
{
    //!
    //! Constructor for particle<dim> structure.
    //! Only sets particle mass, particle type, and whether or not
    //! the particle position is fixed.
    //!
    m=mass;
    type=ptype;
    fix=fixed;
    usepbcond=pbconded;
}
