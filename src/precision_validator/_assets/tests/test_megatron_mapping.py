import copy
import itertools
import unittest
import numpy as np

from precision_validator.megatron_mapping import contiguous_runs, elect_owners, Run

try:
    import torch
    from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
    from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
    from precision_validator.megatron_mapping import MegatronParameterMap, fragment_metadata
except ImportError:
    torch = None


class GeometryTests(unittest.TestCase):
    def test_multiaxis_shards_match_independent_numpy_slices(self):
        full = np.arange(4 * 6 * 8).reshape(4, 6, 8)
        assembled = np.full(full.size, -1)
        for a, b in itertools.product(range(2), repeat=2):
            local = full[a * 2:(a + 1) * 2, :, b * 4:(b + 1) * 4].copy()
            runs = contiguous_runs('x', full.shape, (a * 2, 0, b * 4), local.shape,
                                   tuple(s // 8 for s in local.strides), 0)
            for r in runs:
                assembled[r.start:r.end] = local.reshape(-1)[r.local_start:r.local_end]
        np.testing.assert_array_equal(assembled, full.reshape(-1))

    def test_transpose_and_nonzero_storage_offset(self):
        storage = np.arange(35)
        view = storage[7:31].reshape(4, 6).T
        runs = contiguous_runs('x', view.shape, (0, 0), view.shape, (1, 6), 7)
        actual = np.full(view.size, -1)
        for r in runs:
            actual[r.start:r.end] = storage[r.local_start:r.local_end]
        np.testing.assert_array_equal(actual, view.flatten())

    def test_prepended_layer_expert_split_and_scalar(self):
        r = contiguous_runs('layers.experts.weight', (4, 8, 2, 3), (2, 5, 0, 0), (2, 3), (3, 1), 0, 2)
        self.assertEqual(r, [Run('layers.experts.weight::index=2,5', (2, 3), 0, 6, 0)])
        self.assertEqual(contiguous_runs('scalar', (), (), (), (), 0), [Run('scalar', (), 0, 1, 0)])

    def test_flat_dp_cut_can_start_and_end_inside_rows(self):
        runs = contiguous_runs('row', (3, 8), (0, 4), (3, 4), (4, 1), 0)
        actual = [r.intersect(2, 10) for r in runs]
        self.assertEqual(actual, [Run('row', (3, 8), 6, 8, 2), Run('row', (3, 8), 12, 16, 4), Run('row', (3, 8), 20, 22, 8)])

    def test_replica_divergence_overlap_and_missing_rejected(self):
        def row(rank, start, end, digest='a' * 64):
            return dict(rank=rank, fragment=0, name='x', shape=[4], start=start, end=end,
                        elements=end-start, offset_bytes=0, dtype='torch.float32', sha256=digest)
        catalog = {'x': {'shape': [4]}}
        original = [row(0,0,2), row(1,2,4), row(2,0,2)]
        self.assertEqual(len(elect_owners(original, catalog)), 2)
        bad = [original[:1], original + [row(3,1,3)], original + [row(2,0,2)],
               [row(0,0,2), row(1,2,4), row(2,0,2,'b'*64)]]
        for records in bad:
            with self.subTest(records=records), self.assertRaises(RuntimeError):
                elect_owners(records, catalog)


@unittest.skipIf(torch is None, 'native Megatron/PyTorch runtime is unavailable')
class NativeFactoryTests(unittest.TestCase):
    def make_model(self, rank=0, copying=False, missing=False):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(8,3))
                if missing:
                    self.unmapped = torch.nn.Parameter(torch.ones(2))
            def sharded_state_dict(self, metadata):
                native = ShardedTensor.from_rank_offsets('layer.weight', self.weight, (0,rank,2))
                factory = apply_swiglu_sharded_factory(native, (), singleton_local_shards=True)
                if copying:
                    original = factory.build_fn
                    factory.build_fn = lambda k,t,r,f: original(k,t.clone(),r,f)
                return {'weight': factory}
        return Model()

    def test_native_swiglu_factory_and_dp_owned_ranges(self):
        all_records, actual = [], {}
        for rank in (0,1):
            model = self.make_model(rank)
            mapping = MegatronParameterMap(model)
            self.assertEqual(mapping.factories,1)
            # Three uneven DP slices split one GLU half internally.
            for dp, (a,b) in enumerate(((0,7),(7,19),(19,24))):
                fragments = mapping.fragments({'weight': model.weight.detach().flatten()[a:b]}, {'weight': (a,b)})
                all_records.extend(fragment_metadata(fragments,rank*3+dp))
                for run,value in fragments:
                    dst = actual.setdefault(run.name,np.full(24,-1.0))
                    dst[run.start:run.end] = value.numpy()
        owners = elect_owners(all_records,mapping.catalog())
        self.assertEqual(sum(r['elements'] for r in owners),48)
        for suffix,start in (('_w',0),('_v',12)):
            expected = np.tile(np.arange(start,start+12),2)
            np.testing.assert_array_equal(actual['layer.weight'+suffix],expected)

    def test_copying_and_missing_native_metadata_rejected(self):
        for kwargs in ({'copying':True},{'missing':True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                MegatronParameterMap(self.make_model(**kwargs))

    def test_native_replica_difference_is_never_dropped(self):
        model = self.make_model(0)
        mapping = MegatronParameterMap(model)
        fragments = mapping.fragments({'weight':model.weight})
        records = fragment_metadata(fragments,0) + fragment_metadata(fragments,1)
        records[-1]['sha256'] = '0'*64
        with self.assertRaisesRegex(RuntimeError,'replicas disagree'):
            elect_owners(records,mapping.catalog())


if __name__ == '__main__':
    unittest.main()
