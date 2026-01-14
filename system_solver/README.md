# Param Solver Crate

## Top level game design objectives
We're building a physiics-based "immersive sim" type game, where the physical dynamics of objects in the game world have a big impact on the "game feel" and player experience. Players will be able to interact with physics objects in many ways, and it is important that the interactions feel consistent so that players can develop intuition about how objects will behave and can attain mastery over the game world.

A common pitfall in physics-based game development is that the physical parameters of objects (mass, inertia, drag, joint stiffness, etc) are set in an ad-hoc way by game designers and developers, often through trial-and-error. This can lead to inconsistent and unintuitive behavior, as well as introducing bugs and edge cases that are hard to predict and control. For example, in a 2d game, motion along x and y axes are ofter set independently, which works very well and is extremly tunable for game where physics interactions are limited. However, when interactions become more complex (e.g. objects can rotate, collide, stack, be constrained by joints, etc), the independent tuning of the player controller's x and y dynamics can lead to gameplay that feels weird and inconsistent, as the interactions between the two axes are not properly accounted for.

We also require physical consistency in order to enable scalable interactions between the different "affixes" that might be applied to an object. For example, if an object has a "bouncy" affix that makes it bounce when it collides with the ground, and a "slippery" affix that makes it slide when on a surface, we want to ensure that the combined effect of these affixes is consistent with the underlying physics model. If the physical parameters are set in an ad-hoc way, it can lead to situations where the object behaves in unexpected ways when both affixes are applied, leading to a poor player experience, and making it hard for game designers to reason about how affixes will interact.

## High level objectives for param_solver crate
- game designer set a dynamics model that encodes the general kinds of physical dynamics and interactions desired for the game
- game designers set "DynamicsGivenParams" to constrain the dynamic. These must be frames in terms in easy to visualize+intrepet quatities with interpretable units that designers can understand and set directly, and that make sense from day-to-day life.
  - Examples:
    - "at time t=0.5s, the object should be at position (x,y,z) with velocity (vx,vy,vz)"
    - "the object should bounce off the ground with a coefficient of restitution of 0.8"
    - "the object should come to rest within 2s when sliding on a surface with friction coefficient 0.5"
  - Things that are generally understandable from day-to-day life, and that have direct impact on game feel
    - positions/distances
    - velocities
    - times
  
- Using the `DynamicsGivenParams` and the dynamics model, we want to *discover* the values of `DynamicsDerivedParams` variables needed to achieve consistent dynamics that. The `DynamicsDerivedParams` are typically low-level physics engine parameters that are hard to set directly, but whose values must be set correctly in order to achieve the desired dynamics. These unknowns will typically be values that are not directly intuitive to game designers.
  - Examples:
    - forces/torques
    - impulses
    - energy
    - momentum
    - friction coefficients
    - restitution coefficients

## second level objectives:
- code should use a declarative style so that it's harder to make errors while setting up objectives, and so that objectives can be scanned over and assessed quickly. (this would be in contrast to e.g. writing out bespoke functions for each objective)
- overall loss fn should be split into multiple components (and sub-components in some cases) so that it's possible to assess the quaility of the final params WRT each target.
- discovery process should be *identifiable* and *robust*, and should provide diagnostics and hints to the user about which parameters are not identifiable and why
- discovery process should be efficient enough to run in a game engine editor context (i.e. within a few seconds)


## Some details:
- we explicitly allow coupling of dynamics across different axes and DOFs.
  - However, to make the benchmark scenarios easier to think about, we will often (but not always) favor scenarios where the dynamics are decoupled across different axes and DOFs.
- This tool will only be run in an offline-context in the Godot game engine which is robust to rust panics. It is not intended to run in real-time during gameplay. Panics are desirable in order to catch errors early.