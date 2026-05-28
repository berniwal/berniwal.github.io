"""Smoke tests for layer1.puct_sampler.PUCTSampler.

These cover the bits most likely to silently misbehave:
  - Lineage blocking inside a batch (don't pick parent + grandchild together).
  - Visit-count propagation up the ancestry on update_states.
  - Topk-children-per-parent enforcement.
  - Dedup by expression.
  - record_failed_rollout still bumps n[ancestors] and T.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "layer1"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from layer1.puct_state import State
from layer1.puct_sampler import PUCTSampler


def _child(expr, value, parent, timestep):
    s = State(expr=expr, value=value, timestep=timestep)
    s.set_parent(parent)
    return s


class TestPUCTSampler(unittest.TestCase):

    def test_initial_buffer_seeded(self):
        s = PUCTSampler(seed_count=4)
        self.assertEqual(s.buffer_size(), 4)
        # All initial seeds have value=None and distinct ids
        ids = {st.id for st in s._states}
        self.assertEqual(len(ids), 4)
        self.assertTrue(all(st.value is None for st in s._states))

    def test_update_propagates_visit_counts(self):
        s = PUCTSampler(seed_count=2, topk_children=0)  # disable filter for clarity
        roots = s._states[:]
        # 4 children of root[0]
        children = [_child(f"f{i}", 0.5 + 0.1 * i, roots[0], timestep=1) for i in range(4)]
        s.update_states(children, [roots[0]] * 4, round_n=1)
        self.assertEqual(s._n[roots[0].id], 1)  # parent visited once
        self.assertEqual(s._n.get(roots[1].id, 0), 0)  # other root untouched
        self.assertEqual(s._T, 1)
        # m[parent] = max child value
        self.assertAlmostEqual(s._m[roots[0].id], 0.5 + 0.1 * 3)

    def test_visit_counts_propagate_to_grandparent(self):
        s = PUCTSampler(seed_count=1, topk_children=0)
        gp = s._states[0]
        parent = _child("p", 0.7, gp, timestep=1)
        s.update_states([parent], [gp], round_n=1)
        # Now expand from `parent` -> children
        grandkids = [_child(f"gk{i}", 0.8 + 0.01 * i, parent, timestep=2) for i in range(3)]
        s.update_states(grandkids, [parent] * 3, round_n=2)
        # gp got 1 visit from parent, then 1 more from grandkids -> 2
        self.assertEqual(s._n[gp.id], 2)
        # parent got 1 from the grandkids batch
        self.assertEqual(s._n[parent.id], 1)
        self.assertEqual(s._T, 2)

    def test_lineage_blocking_excludes_parent_when_child_picked(self):
        s = PUCTSampler(seed_count=2, puct_c=0.0, topk_children=0)  # zero exploration -> sort purely by Q
        # initial values: None for seeds. Add a child of seed[0] with high value, plus an isolated seed[1].
        seed0, seed1 = s._states[0], s._states[1]
        child = _child("c", 0.9, seed0, timestep=1)
        s.update_states([child], [seed0], round_n=1)
        # Now pick 2 -- should pick `child` (highest), then `seed1` (only non-blocked left),
        # and NOT `seed0` (blocked because it's `child`'s parent).
        picked = s.sample_states(2)
        picked_ids = {p.id for p in picked}
        self.assertIn(child.id, picked_ids)
        self.assertIn(seed1.id, picked_ids)
        self.assertNotIn(seed0.id, picked_ids)

    def test_lineage_blocking_does_NOT_block_sibling(self):
        s = PUCTSampler(seed_count=1, puct_c=0.0, topk_children=0)
        root = s._states[0]
        child_a = _child("a", 0.9, root, timestep=1)
        child_b = _child("b", 0.85, root, timestep=1)
        s.update_states([child_a, child_b], [root, root], round_n=1)
        # Picking 2 from {root, child_a, child_b}: child_a (0.9) first -> blocks root + child_a.
        # child_b is sibling of child_a -- NOT blocked. Should be the second pick.
        picked = s.sample_states(2)
        picked_ids = [p.id for p in picked]
        self.assertEqual(picked_ids[0], child_a.id)
        self.assertEqual(picked_ids[1], child_b.id)

    def test_topk_children_filter(self):
        s = PUCTSampler(seed_count=1, topk_children=2)
        root = s._states[0]
        # 4 children, only top-2 by value should be kept
        kids = [_child(f"k{i}", 0.1 * i, root, timestep=1) for i in range(4)]
        s.update_states(kids, [root] * 4, round_n=1)
        # buffer should have root + top-2 kids = 3 states
        self.assertEqual(s.buffer_size(), 3)
        kept_exprs = {st.expr for st in s._states if st.expr}
        self.assertEqual(kept_exprs, {"k3", "k2"})  # top-2 by value

    def test_dedup_by_expr(self):
        s = PUCTSampler(seed_count=1, topk_children=0)
        root = s._states[0]
        a1 = _child("x*x + 1", 0.9, root, timestep=1)
        a2 = _child("x*x + 1", 0.85, root, timestep=1)  # duplicate expr
        s.update_states([a1, a2], [root, root], round_n=1)
        # Only one of the two should be added
        added_exprs = [st.expr for st in s._states if st.expr == "x*x + 1"]
        self.assertEqual(len(added_exprs), 1)

    def test_record_failed_rollout(self):
        s = PUCTSampler(seed_count=1, topk_children=0)
        root = s._states[0]
        parent = _child("p", 0.5, root, timestep=1)
        s.update_states([parent], [root], round_n=1)
        before_T = s._T
        before_n_root = s._n[root.id]
        s.record_failed_rollout(parent)
        # T += 1, parent and its ancestors get n += 1
        self.assertEqual(s._T, before_T + 1)
        self.assertEqual(s._n[root.id], before_n_root + 1)
        self.assertEqual(s._n[parent.id], 1)


if __name__ == "__main__":
    unittest.main()
