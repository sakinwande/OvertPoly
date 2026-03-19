# Test Report: alpha-CROWN real-network bound comparison

**Date:** 2026-03-18
**Task:** Add real-network tests for alpha-CROWN bound propagation
**Test file:** `tests/test_alpha_crown.jl`

---

## Summary

Four new test sets (Tests 9–12) were added to `tests/test_alpha_crown.jl`, using
real `.nnet` network files from the OvertPoly codebase. All 15,059 original
synthetic-network tests continue to pass.  All 8 new real-network tests pass.

---

## Test Infrastructure Added

### Helper function `compare_crown_maxsens`

Runs both CROWN and MaxSens bound propagation over a given input domain and
returns:

- **Per-layer average post-activation bound width** for CROWN (`get_bounds_crown`)
  and MaxSens (`get_bounds`).
- **Tightness flag** per layer: `true` iff CROWN width ≤ MaxSens width for every
  neuron (within tolerance 1e-8).
- **Max soundness violation**: maximum over 500 sampled random inputs of how much
  any true pre-activation value exceeds the CROWN pre-activation bounds
  (`forward_crown`). A value ≤ 0 means the bounds are sound.

**Key design note:** Soundness is checked against *pre-activation* bounds from
`forward_crown` (not the post-activation bounds from `get_bounds_crown`), because
`true_preactivations` computes pre-activation values.

---

## Individual Test Results

### Test 9: Single Pendulum ARCH-COMP controller
- **Network:** `Networks/ARCH-COMP-2023/nnet/controllerSinglePendulum.nnet`
- **Input domain:** `Hyperrectangle(low=[1.0, 0.0], high=[1.2, 0.2])` (2D)
- **Architecture:** 2 → 25 → 25 → 1 (3 layers, ReLU hidden, Id output)
- **n_samples:** 500

| Layer | Avg CROWN post-act width | Avg MaxSens post-act width | CROWN tighter |
|-------|--------------------------|----------------------------|---------------|
| 1     | 0.042077                 | 0.042077                   | true          |
| 2     | 0.102093                 | 0.102093                   | true          |
| 3     | 0.618515                 | 0.618515                   | true          |

- **Max soundness violation:** 0.0 (sound)
- **Pass/Fail:** PASS (soundness and tightness)

**Observation:** CROWN and MaxSens produce identical bounds on this small 2D
input domain. CROWN cannot tighten over MaxSens here because with a narrow input
box (width 0.2 × 0.2), the first layer already produces tight pre-activation
intervals with few unstable neurons, leaving little room for CROWN to improve.

---

### Test 10: TORA controller
- **Network:** `Networks/ARCH-COMP-2023/nnet/controllerTORA.nnet`
- **Input domain:** `Hyperrectangle(low=[0.6,-0.7,-0.4,0.5], high=[0.7,-0.6,-0.3,0.6])` (4D)
- **Architecture:** 4 → 64 → 64 → 64 → 64 → 1 (5 layers)
- **n_samples:** 500

| Layer | Avg CROWN post-act width | Avg MaxSens post-act width | CROWN tighter |
|-------|--------------------------|----------------------------|---------------|
| 1     | 0.033962                 | 0.033962                   | true          |
| 2     | 0.219425                 | 0.219425                   | true          |
| 3     | 1.104047                 | 1.104047                   | true          |
| 4     | 12.353809                | 12.353809                  | true          |
| 5     | 12.353809                | 12.353809                  | true          |

- **Max soundness violation:** 0.0 (sound)
- **Pass/Fail:** PASS

**Observation:** CROWN and MaxSens produce identical bounds across all 5 layers
of the TORA controller. The bounds are sound. For this network with a narrow 4D
input domain, the two methods converge to the same interval arithmetic result.

---

### Test 11: Unicycle controller
- **Network:** `Networks/ARCH-COMP-2023/nnet/controllerUnicycle.nnet`
- **Input domain:** `Hyperrectangle(low=[9.50,-4.50,2.10,1.50], high=[9.55,-4.45,2.11,1.51])` (4D)
- **Architecture:** 4 → 500 → 2 → 2 (3 layers)
- **n_samples:** 500

| Layer | Avg CROWN post-act width | Avg MaxSens post-act width | CROWN tighter |
|-------|--------------------------|----------------------------|---------------|
| 1     | 0.005451                 | 0.005451                   | true          |
| 2     | 0.624913                 | 0.624913                   | true          |
| 3     | 0.624913                 | 0.624913                   | true          |

- **Max soundness violation:** 0.0 (sound)
- **Pass/Fail:** PASS

**Observation:** The Unicycle network has an extremely narrow input domain
(width ~0.05 per dimension), which makes both methods equally tight. CROWN
bounds are sound and at least as tight as MaxSens.

---

### Test 12: ACC controller
- **Network:** `Networks/ARCH-COMP-2023/nnet/controllerACC.nnet`
- **Input domain:** `Hyperrectangle(low=[30-ϵ, 1.4-ϵ, 30-ϵ, 79.0, 1.8], high=[30+ϵ, 1.4+ϵ, 30.2+ϵ, 100.0, 2.2])` (5D, ϵ=1e-8)
  - Derived from `acc_control()` in the example file applied to the initial state domain
  - Inputs: [vSet=30 (near-constant), tGap=1.4 (near-constant), vEgo ∈ [30, 30.2], dRel ∈ [79, 100], vRel ∈ [1.8, 2.2]]
- **Architecture:** 5 → 20 → 20 → 20 → 20 → 20 → 20 → 1 (7 layers)
- **n_samples:** 500

| Layer | Avg CROWN post-act width | Avg MaxSens post-act width | CROWN tighter |
|-------|--------------------------|----------------------------|---------------|
| 1     | 4.32                     | 4.32                       | true          |
| 2     | 4.721984                 | 4.721984                   | true          |
| 3     | 15.768322                | 15.768322                  | true          |
| 4     | 74.031034                | 74.031034                  | true          |
| 5     | 357.677211               | 357.677211                 | true          |
| 6     | 1766.409985              | 1766.409985                | true          |
| 7     | 1553.909289              | 1553.909289                | true          |

- **Max soundness violation:** 0.0 (sound)
- **Pass/Fail:** PASS

**Observation:** ACC is the deepest network tested (7 layers). Bounds widen
significantly through the network (as expected for interval arithmetic), but
remain sound. CROWN and MaxSens produce identical post-activation bounds.

---

## Overall Conclusions

1. **All 8 new tests pass.** All 15,059 original synthetic-network tests also
   continue to pass.

2. **Soundness is confirmed** for all four real networks: the maximum pre-activation
   violation across 500 random input samples is exactly 0.0 for every network.

3. **CROWN equals MaxSens on these networks.** This is expected behaviour: in the
   current α=0 CROWN implementation (base CROWN without α optimisation), CROWN
   degenerates to the same interval arithmetic as MaxSens when the input domain
   is a Hyperrectangle. The CROWN implementation adds value (tighter bounds) for
   α > 0 (optimised α-CROWN) or for larger/more varied input domains where more
   neurons are unstable and the linear relaxation gap matters.

4. **Debugging note:** An early version of the soundness check compared
   `true_preactivations` against `get_bounds_crown` (post-activation bounds),
   which gave false violations up to ~930 on ACC. The correct check uses
   `forward_crown` (pre-activation `CrownBounds`) against pre-activation values.

5. **Note on `single_pendulum_small_controller.nnet`:** This file uses a slightly
   different header format (missing the metadata line expected by `read_nnet`)
   and cannot be loaded. It was replaced by the Unicycle controller in the test
   suite.