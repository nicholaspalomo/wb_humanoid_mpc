# Basis-Vector Contact Inputs

With `contact_input_parameterization: "basis_vectors"` in a robot's `config/mpc/task.textproto`, the centroidal NMPC no longer
optimizes each foot's contact wrench $W = (F, \tau) \in \mathbb{R}^6$ directly. It optimizes non-negative scalings
$\lambda$ of a fixed set of wrench generators instead,

$$W_{\text{local}} = B\,\lambda, \qquad \lambda \ge 0,$$

where the columns of $B \in \mathbb{R}^{6 \times n_b}$ are wrenches inside the foot's contact wrench cone, written in
the foot's local contact frame. The cone is convex, so every $\lambda \ge 0$ gives a wrench inside it, and the explicit
`contact_wrench_cone` soft constraint is replaced by the bound $\lambda \ge 0$ - a barrier the parameterization builds
for every contact, whatever the soft-constraint lists say (section 7).

This document covers the three choices that formulation involves, each made by name in the task file, and one fix to
how a wrench is written into the input:

| Choice | Task-file field | Names | Default |
| --- | --- | --- | --- |
| How the contact inputs are parameterized | `contact_input_parameterization` | `wrench`, `basis_vectors` | `wrench` |
| Which generators make up $B$ | `contacts.basis_generator_set` | `conservative_inner_approximation`, `exact_wrench_cone` | `conservative_inner_approximation` |
| How the singular input cost is regularized | `contacts.basis_regularization` | `full_diagonal`, `null_space` | `full_diagonal` |

The DRC Atlas and the EngineAI SA01 ship `basis_vectors`, the Unitree G1 and R1 `wrench`. **The generator set and the
regularization every robot ships are the defaults, which is what has always run.** Their alternatives change the closed
loop and have not been validated in simulation yet, so nothing switches to them on its own; section 7 says how to
select them.

---

## 1. Block diagram

```
 task.textproto                                     humanoid_common_mpc / humanoid_centroidal_mpc
 --------------                                     ---------------------------------------------

 contact_input_parameterization: "basis_vectors" -> contactInputParameterizationFromConfig (registry: wrench | basis_vectors)
                                                          | CentroidalMpcInterface::setupContactInputParameterization
                                                          v
 contacts.contact_wrench_cone_soft_constraint --+  contactWrenchConeBasesFromConfig(contacts, modelSettings)
   friction_coefficient (mu)                    |    contactWrenchConeConfigFromConfig (every field required)
   torsional_friction_coefficient (mu_t)        +-> ContactWrenchConeBasisMatrix::Create(config, footprint, set)
   num_basis_vectors (N)                        |    registry: getBasisGeneratorSetBuilder(set)
 contacts.contact_rectangle                   --+      conservative_inner_approximation  -> N + 7 columns
 contacts.basis_generator_set ------------------+      exact_wrench_cone                 -> 8N columns
                                                     checks every column against buildLocalWrenchConeRows
                                                          |
                                                          v
                                       B_i, B_i^+, P_i = I - B_i^+ B_i, basisInputsLibraryKey(B_0, B_1)
                                                          |
         +------------------------------+-----------------+-------------------+-----------------------------+
         v                              v                                     v                             v
 BasisInputsModelDecorator     CentroidalDynamicsBasisInputsAD     BasisInputsCostTransform       BasisScalingNonNegativity-
 u = [lambda_0, lambda_1, qd]  W_i,world = R_i(q) B_i lambda_i     R_basis = M^T R M               Constraint
 get: W_local = B lambda       -> centroidal flow map (CppAD,               + reg blkdiag(S, 0)    lambda >= 0
 set: lambda = NNLS(B, W)         library named by the key)        S by name: full_diagonal |     (barrier, or hinge when
                                                                              null_space           un-gated)
```

`M = blkdiag(B_0, B_1, I_joints)` is the constant map from the basis-vector input to the wrench-space input with
every wrench in its local contact frame (`BasisInputsModelDecorator::getLocalBasisToWrenchMap`). The dynamics rotate
each $B_i \lambda_i$ into the world frame inside the CppAD tape, so they do not use $M$.

---

## 2. The cone wrench mode enforces (H-representation)

`buildLocalWrenchConeRows` builds the rows `ContactWrenchConeConstraint` enforces in wrench mode, and it is the single
definition both formulations are checked against. Dropping the affine offsets (`min_normal_force`, `gripper_force`), a
local wrench $W = (F_x, F_y, F_z, \tau_x, \tau_y, \tau_z)$ is admissible when

$$
\begin{aligned}
&\mu F_z - (\cos\theta_k F_x + \sin\theta_k F_y) \ge 0, \quad \theta_k = 2\pi k / N, \; k = 0 \dots N-1 && \text{(friction facets)}\\
&F_z \ge 0 \\
&x_{\min} F_z \le -\tau_y \le x_{\max} F_z, \qquad y_{\min} F_z \le \tau_x \le y_{\max} F_z && \text{(center of pressure)}\\
&\left| \tau_z + p_y F_x - p_x F_y \right| \le \mu_t F_z && \text{(torsion about the patch point } p\text{)}
\end{aligned}
$$

The friction rows give $F_z \ge 0$ by themselves, and a wrench with $F_z = 0$ must be zero, so the cone is pointed and
is the cone over its slice at $F_z = 1$. On that slice the rows bound three things **independently**:

* the tangential force $f = (F_x, F_y)$ lies in the friction polygon $\mathcal{P}$ (the $N$ facets);
* the center of pressure $c = (-\tau_y, \tau_x)$ lies in the footprint $\mathcal{R}$;
* the torsion about the patch point $t = \tau_z + p_y F_x - p_x F_y$ lies in $\mathcal{T} = [-\mu_t, \mu_t]$.

The map $(f, c, t) \mapsto W = (f, 1, c_y, -c_x, t - p_y f_x + p_x f_y)$ is an affine bijection from
$\mathcal{P} \times \mathcal{R} \times \mathcal{T}$ onto the slice.

## 3. The exact generator set (V-representation)

The vertices of a product of polytopes are the products of their vertices, and an affine bijection maps vertices to
vertices. The slice therefore has exactly $N \cdot 4 \cdot 2 = 8N$ vertices, and the cone exactly $8N$ extreme rays:
every friction-pyramid edge

$$f_k = \frac{\mu}{\cos(\pi/N)} \left(\cos\tfrac{(2k+1)\pi}{N},\, \sin\tfrac{(2k+1)\pi}{N}\right)$$

at every footprint corner $c$, with the torsional limit of either sign $s = \pm 1$:

$$b_{k,c,s} = \left(f_{k,x},\; f_{k,y},\; 1,\; c_y,\; -c_x,\; s\,\mu_t + p_x f_{k,y} - p_y f_{k,x}\right).$$

That is `exact_wrench_cone`, column $(4k + c)\cdot 2 + s$. Its conic hull **is** the cone of section 2 - neither
larger nor smaller. With $\mu_t > 0$ and a footprint of positive area no smaller set can achieve that, because every
extreme ray has to be a generator (with $\mu_t = 0$ the two torsion signs give the same column, which is harmless). It
costs input dimension: 32 scalings per foot for $N = 4$ instead of 11, which grows the QP and the CppAD dynamics.

## 4. Why the shipped set is only an inner approximation

`conservative_inner_approximation` has $N + 7$ columns: the $N$ friction edges applied at the patch point, one pure
normal force at the patch point, a normal force at each of the four footprint corners, and a normal force at the patch
point with torsion $\pm\mu_t$. Every column has $F_z = 1$, so at $F_z = 1$ the scalings are convex weights, and each
column spends its weight on **one** of the three limits: the friction edges carry all the tangential force, the corner
rays all the center-of-pressure excursion from $p$, the torsion rays all the torsion. The three therefore share one
budget,

$$\rho_{\mathcal{P}}(f) + \rho_{\mathcal{R}}(c - p) + \rho_{\mathcal{T}}(t) \;\lesssim\; 1,$$

where each $\rho$ is the fraction of its own limit in use, while the rows of section 2 allow each of them up to 1 at
the same time. The set is **sound** - every column is inside the cone - but strongly conservative:

* its projection onto the force alone is exact: any force of the friction pyramid, applied at the patch point with no
  torsion, is representable;
* it cannot combine a tangential force with a center of pressure away from the patch point: at the toe
  ($c_x = x_{\max}$) no tangential force at all, at $c_x = 0.8\,x_{\max}$ at most a fifth of the friction limit;
* it cannot combine torsion with either: at the torsional limit, no tangential force and no center-of-pressure
  excursion.

Sampling the cone on the DRC Atlas and EngineAI SA01 geometries, it reproduces only a few percent of the admissible
wrenches (`testContactWrenchConeBasisMatrix` logs the figure for every geometry it checks). Braking, push-off and push
recovery all want tangential force with the center of pressure near an edge of the foot, which is exactly what this
set cannot express - so basis mode with this set plans with a much smaller contact-wrench set than wrench mode does.

**What no generator set can represent.** `min_normal_force` and `gripper_force` are affine offsets of the cone, and a
conic combination is homogeneous ($\lambda = 0$ always gives $W = 0$). Neither is enforced in basis mode, and
`CentroidalMpcInterface` warns when either is configured.

<!-- LINT.IfChange(generator_set_names) -->
| Generator set | Columns per foot | Spans |
| --- | --- | --- |
| `conservative_inner_approximation` (default) | $N + 7$ | a strict inner approximation of the wrench-mode cone |
| `exact_wrench_cone` | $8N$ | exactly the wrench-mode cone |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h:basis_generator_set_names) -->

## 5. The input cost and its regularization

The task file writes the input weight $R$ in wrench space. In basis mode it becomes

$$R_{\text{basis}} = M^\top R\, M + \text{reg} \cdot \operatorname{blkdiag}(S, 0_{\text{joints}}),$$

with `reg` = `contacts.basis_scaling_regularization`. Each $B_i$ has more columns than rows, so $M^\top R M$ is singular
on $\operatorname{null}(B_i)$: the scalings that produce no wrench. $S$ removes that singularity, and how it does so
decides what the optimizer actually sees on the wrench. Minimizing $u^\top R_{\text{basis}} u$ over the inputs that
produce a given wrench leaves $W^\top G\, W$ with

$$G = \left(M\, R_{\text{basis}}^{-1} M^\top\right)^{-1}.$$

**`full_diagonal`** ($S = I$, the default). Then, per foot, $G = R + \text{reg}\,(B B^\top)^{-1}$. The second term
is not small: with the DRC Atlas values at the time of writing ($\text{reg} = 10^{-4}$ against contact weights of
$10^{-5}$ to $2 \cdot 10^{-4}$) it is about 10x and 5x the tangential force weights and 100x to 400x the moment
weights, so in basis mode the moment and tangential entries of $R$ barely matter. Lowering `reg` shrinks the
distortion but not its shape.

**`null_space`** ($S = \operatorname{blkdiag}(I - B_i^+ B_i)$, the orthogonal projector onto $\operatorname{null}(B_i)$).
Split any $\lambda$ into $\lambda_r \in \operatorname{range}(B^\top)$ and $\lambda_n \in \operatorname{null}(B)$. Then

$$\lambda^\top R_{\text{basis}} \lambda = (B\lambda)^\top R\, (B\lambda) + \text{reg}\, \lVert \lambda_n \rVert^2,$$

the regularization vanishes on $\operatorname{range}(B^\top)$, the minimum over the scalings producing $W$ is reached
at $\lambda_n = 0$, and $G = R$ **exactly**. $R_{\text{basis}}$ is still positive definite on the scalings as long as
$\text{reg} > 0$ and $R$ gives every contact force and moment a positive weight: $B^\top R B$ is positive definite on
$\operatorname{range}(B^\top)$ and the projector on $\operatorname{null}(B)$. `checkLambdaBlockPositiveDefinite` reports
the case where it is not, naming the fields to change.

<!-- LINT.IfChange(regularization_names) -->
| Regularization | $S$ per foot | Metric on the wrench |
| --- | --- | --- |
| `full_diagonal` (default) | $I$ | $R + \text{reg}\,(B B^\top)^{-1}$ |
| `null_space` | $I - B^+ B$ | $R$ |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/common/BasisInputsCostTransform.h:basis_regularization_names) -->

Both are built by `transformWrenchInputCostToBasisSpace(R, config)` from one `BasisInputsCostTransformConfig`
(`CentroidalMpcInterface::getBasisInputsCostTransformConfig`), whose `regularization` names $S$ and which the OCP
factory (`HumanoidCostConstraintFactory::setBasisInputsCostTransform`) and the online parameter updater (its centroidal
`BasisInputsCostApplier`, `humanoid_centroidal_mpc/parameter_update/`) share, so a hot
reload applies the same regularization as the start-up. Start-up refuses a configuration whose $\lambda$ block is not
positive definite (a zero `reg`, or `null_space` with a contact wrench direction $R$ does not weigh), naming the field;
the updater refuses such a reload and keeps the running $R$.

## 6. Writing a wrench into the input

`BasisInputsModelDecorator::setContactWrench` (and the world-frame setters built on it) has to find $\lambda \ge 0$ with
$B\lambda = W$. The minimum-norm $B^+ W$ spreads the load over every generator and is used when it is already
non-negative. When it is not, clamping it at zero - what the setter used to do - changes the wrench even though an
exact non-negative solution exists: the weight of the robot seen from a stance foot pitched by 15 degrees came back
7 % too large with a 6 % sideways force on Atlas. The setter now solves the non-negative least-squares problem
(Lawson-Hanson, `solveNonNegativeLeastSquares`) instead, which reproduces every wrench inside the cone of $B$ exactly
and returns the closest one inside for a wrench outside it. That matters for the weight-compensation warm start of
every new horizon node and for the MRT's weight-compensation fallback torques. The AD instantiation of the decorator
cannot tape an active-set solve and keeps the clamp; no taped path calls a setter.

## 7. Selecting each choice

The registries are the single place a parameterization, a generator set or a regularization is added, and each rejects
an unknown name with a message that names its field and lists the valid names:

* `contactInputParameterizationNames()` / `contactInputParameterizationFromName(name)` in
  `common/ContactInputParameterization.h`, read by `contactInputParameterizationFromConfig(task)`;
* `getBasisGeneratorSetBuilder(name)` / `basisGeneratorSetNames()` in `ContactWrenchConeBasisMatrix.h`, used by
  `ContactWrenchConeBasisMatrix::Create(config, footprint, name)` and read by `contactWrenchConeBasesFromConfig`;
* `getBasisRegularizationBuilder(name)` / `basisRegularizationNames()` in `BasisInputsCostTransform.h`, used through
  `BasisInputsCostTransformConfig::regularization`.

<!-- LINT.IfChange(contact_input_parameterization_names) -->
| Field | Where | Read | Values |
| --- | --- | --- | --- |
| `contact_input_parameterization` | top level of `task.textproto` | at start-up | `wrench` (default when absent), `basis_vectors` |
| `contacts.basis_generator_set` | `contacts` | at start-up | section 4 |
| `contacts.basis_regularization` | `contacts` | at start-up and on every hot reload | section 5 |
| `contacts.basis_scaling_regularization` | `contacts` | at start-up and on every hot reload | `reg` of section 5, $\ge 0$ |
| `contacts.basis_non_negativity_barrier.mu`, `.delta` | `contacts` | at start-up and on every hot reload | the barrier below |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/common/ContactInputParameterization.h:contact_input_parameterization_names) -->

The parameterization and the generator set fix the input dimension, so they take effect at the next start; a hot
reload that changes the parameterization is reported by the parameter updater rather than applied. The tuning GUI
labels the `contacts` sliders to match: the geometry of the cone and of the footprint `(restart)`, and the barrier of a
block the selected parameterization does not read - the soft wrench cone's under `basis_vectors`, the basis blocks
under `wrench` - `(not applicable: ...)`. The `use_contact_basis_vector_inputs` boolean that used to select the
parameterization is a retired field of the task file: a file that carries it is refused when it is read, whatever its
value, with a message naming its replacement. The
whole-body MPC implements the wrench parameterization only and refuses `basis_vectors`. A value of any of these fields
that is out of range or an unknown name is refused by its field, at start-up before any CppAD model is built, and on a
hot reload by keeping the running value.

**The barrier.** Under `basis_vectors` the bound $\lambda \ge 0$ is the contact wrench cone: nothing else in the
problem bounds the individual scalings (`friction_force_cone` bounds only the assembled force). So
`CentroidalMpcInterface` builds `BasisScalingNonNegativityConstraint` for every contact, gated on the mode schedule
exactly when `zero_wrench` is, whatever the soft-constraint lists say; a listed `contact_wrench_cone` builds nothing
more. It used to be built only when `contact_wrench_cone` was listed, so dropping that entry silently removed the cone.

**The ground.** `contacts.contact_wrench_cone_soft_constraint` is read by one conversion,
`contactWrenchConeConfigFromConfig`, for the wrench cone, for these generators and for the online contact
planner's friction and torsion bounds. Every one of its five geometry fields is required: a missing field used to leave
the library default of the config (mu 0.7, 5 N) in place without a word.

## 8. CppAD libraries

Every CppAD tape built on the basis-vector input depends on the parameterization: its input dimension is
$n_b N_{\text{contacts}} + n_j$, and some tapes (the external-torque cost, the contact-moment constraint) embed $B$
itself. `basisInputsLibraryKey(B_0, B_1)` returns `basis<n_b>_<16 hex digits>`, a content hash of the bases that is
stable across processes and platforms, and `kWrenchInputsLibraryKey` is its wrench-mode counterpart. The dynamics
library carries the key in its name (`dynamics_basis11_...`), and `CentroidalMpcInterface` appends the key of the
parameterization to the CppAD model folder before any term is built -
`cppad_code_gen/cppad_centroidal_mpc_<robot>/wrench_inputs` or `.../basis11_<hash>` - so every taped term is keyed
by it. OCS2 loads a cached library by name without checking its domain, and every robot ships
`model_settings.recompile_libraries_cpp_ad: false`, so without that key switching the parameterization or the basis
loaded a library built for another input dimension and the process exited on its first evaluation. The first start of
a centroidal robot with a parameterization or a basis it has not run before compiles every library once, into that
key's folder.

## 9. Tests

| Target | What it shows |
| --- | --- |
| `//humanoid_nmpc/humanoid_common_mpc:testContactWrenchConeBasisMatrix` | every column of every set lies inside the rows; every sampled wrench of the cone is a non-negative combination of the exact set, whose columns are exactly the cone's vertices and none redundant; the conservative set provably misses the audit's counterexamples and its coverage is logged; the registry, the key-naming errors, the NNLS and the library key |
| `//humanoid_nmpc/humanoid_common_mpc:testBasisInputsCostTransform` | `null_space` leaves the wrench metric exactly $R$ and the regularization vanishes on $\operatorname{range}(M^\top)$; `full_diagonal` is the old transform bit for bit and distorts the metric by $\text{reg}\,(BB^\top)^{-1}$; positive definiteness and when it fails |
| `//humanoid_nmpc/humanoid_centroidal_mpc:testBasisInputsModelDecorator` | wrenches whose minimum-norm scalings are negative, and the weight seen from a pitched foot, round-trip exactly; the exact set decorates the same model |
| `//humanoid_nmpc/humanoid_centroidal_mpc_test:test_basis_inputs_formulation` | the full Atlas interface in both modes: dimensions, dynamics, the input cost, and that the shipped defaults are unchanged; the CppAD folder keyed by the parameterization and the basis; the barrier built without `contact_wrench_cone`; `null_space` reaching the start-up input cost |
| `//humanoid_nmpc/humanoid_common_mpc:testContactInputParameterization` | the parameterization registry, its conversion and the refusal of the retired boolean; the shipped robots select what they ran before; every field of the wrench-cone block required by name; the bases a task file's `basis_generator_set` builds |
| `//humanoid_nmpc/humanoid_centroidal_mpc:testContactInputParameterizationWiring` | `CentroidalMpcInterface::Create` refuses, by field and before any CppAD model is built, the retired boolean, an unknown parameterization, generator set or regularization, a negative, zero or unparsable `basis_scaling_regularization`, an unparsable barrier `mu` or `delta`, and a wrench-cone block missing a field (for the basis and for the contact planner) |
| `//humanoid_nmpc/humanoid_centroidal_mpc:testMpcParameterUpdaterModule` | on the shipped Atlas: a hot reload reproduces the factory's basis-space $R$, applies a named `basis_regularization`, refuses an unknown shape, an unparsable weight or an indefinite $\lambda$ block and keeps the running $R$, keeps the running barrier when its `mu` is not a number, reports a changed parameterization as structural, and logs no failure for cone terms the problem does not carry |
| `//humanoid_nmpc/humanoid_wb_mpc:testWBMpcFormulation` | the whole-body MPC refuses `basis_vectors`, the retired boolean and the contact planner by name |
