"""CPU mocks/PNG fixtures only. No native Runtime is loaded or executed."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import demo_driver as d


def receipt():
    return {'condition_hashes': {'au_embedding': 'fixed-au', 'image_embedding': 'fixed-ip'},
        'source_latent_sha256': 'fixed-source', 'noise_sha256': 'fixed-noise',
        'projection_trace': [{'t': t, 'latent_sha256': str(t)} for t in range(15)],
        'deis_unet_calls': 15, 'terminal_unet_calls': 0, 'total_unet_calls': 15}


class FakeRuntime:
    def __init__(self):
        self.prepares, self.samples = [], []
    def prepare(self, *args):
        self.prepares.append(args)
        return {'unchanged_native_prepared': True}
    def sample(self, prepared, seed):
        self.samples.append((prepared, seed))
        return Image.new('RGB', (512, 512), (43, 78, 95)), receipt()


class DemoTests(unittest.TestCase):
    def test_fixed_plan_has_one_unchanged_request_and_no_statistical_addition(self):
        plan = d.make_plan('demo_synthetic_plan_only')
        self.assertEqual((plan['case_id'], plan['variant'], plan['seed']), ('postfreeze_SN003_s00869_t01203', 'bidirectional', 101))
        self.assertFalse(plan['included_in_statistics'])
        self.assertEqual(plan['logical_outputs_added_to_main'], 0)
        self.assertEqual(plan['full_protocol_logical_outputs_unchanged'], 768)
        self.assertEqual(plan['full_protocol_physical_jobs_unchanged'], 704)
        self.assertFalse(Path(plan['output_directory']).exists())

    def test_default_dryrun_never_loads_native_or_creates_output(self):
        with patch.object(sys, 'argv', ['demo_driver.py', '--run-name', 'demo_synthetic_dryrun']), patch.object(d, 'load_native') as load, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(d.main(), 0)
        load.assert_not_called()
        self.assertFalse((d.OUTPUT_ROOT / 'demo_synthetic_dryrun').exists())

    def test_arbitrary_case_option_is_not_an_api(self):
        with patch.object(sys, 'argv', ['demo_driver.py', '--case-id', 'another']), patch.object(d, 'load_native') as load, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                d.main()
        self.assertEqual(error.exception.code, 2)
        load.assert_not_called()

    def test_pending_terminal_guard_prevents_native_load(self):
        with patch.object(d.reproduce, 'execution_guard', return_value={'status': 'PENDING', 'exit_code': 3}), patch.object(d, 'load_native') as load:
            result, code = d.execute({})
        self.assertEqual(code, 3)
        self.assertEqual(result['status'], 'PENDING')
        load.assert_not_called()

    def test_native_routing_calls_exactly_one_prepare_and_sample(self):
        with tempfile.TemporaryDirectory(dir=d.HERE) as directory:
            output = Path(directory)
            original = output / 'original.png'
            Image.new('RGB', (512, 512), (43, 78, 95)).save(original)
            expected = {**receipt(), 'output': str(original), 'sha256': d.sha(original)}
            runtime = FakeRuntime()
            result = d.produce_one(runtime, {'case_id': d.CASE_ID}, {'source_id': 'source-only'}, ('edit', 'cleanup', 'receipt'), output, expected)
            self.assertEqual(len(runtime.prepares), 1)
            self.assertEqual(runtime.prepares[0][0], 'bidirectional')
            self.assertEqual(runtime.samples, [({'unchanged_native_prepared': True}, 101)])
            self.assertTrue(result['matched'])
            self.assertEqual(result['sampler_calls'], 1)
            self.assertFalse(result['included_in_statistics'])
            # Existing image is rejected before even a mocked sampler call.
            second = FakeRuntime()
            with self.assertRaises(ValueError):
                d.produce_one(second, {}, {}, (), output, expected)
            self.assertEqual(second.samples, [])

    def test_png_encoding_fallback_requires_exact_pixels(self):
        with tempfile.TemporaryDirectory(dir=d.HERE) as directory:
            output = Path(directory)
            original, differently_encoded, changed = [output / name for name in ('a.png', 'b.png', 'changed.png')]
            image = Image.new('RGB', (512, 512), (43, 78, 95))
            image.save(original, compress_level=0)
            image.save(differently_encoded, compress_level=9)
            expected = {'output': str(original), 'sha256': d.sha(original)}
            result = d.compare_output(differently_encoded, expected)
            self.assertFalse(result['png_sha_equal'])
            self.assertTrue(result['decoded_pixels_equal'])
            image.putpixel((0, 0), (44, 78, 95))
            image.save(changed)
            self.assertFalse(d.compare_output(changed, expected)['matched'])

    def test_existing_demo_directory_is_not_reused(self):
        with tempfile.TemporaryDirectory(dir=d.HERE) as directory, patch.object(d, 'OUTPUT_ROOT', Path(directory)):
            (Path(directory) / 'demo_existing').mkdir()
            with self.assertRaises(ValueError):
                d.choose_output('demo_existing')

    def test_mock_native_failure_preserves_demo_receipt_without_statistics(self):
        with tempfile.TemporaryDirectory(dir=d.HERE) as directory:
            root = Path(directory)
            output = root / 'demo_failure'
            plan = {'kind': 'fixed_v11_single_image_demo', 'output_directory': str(output)}
            expected = {'output': 'old.png', 'sha256': 'oldsha'}
            with patch.object(d, 'OUTPUT_ROOT', root), patch.object(d.reproduce, 'execution_guard', return_value={'status': 'READY', 'exit_code': 0}), patch.object(d.reproduce, 'exclusive_lock', return_value=contextlib.nullcontext()), patch.object(d.reproduce, 'frozen_inputs'), patch.object(d, 'original_reference', return_value=(expected, {'path': 'original.json', 'sha256': 'originalsha'})), patch.object(d, 'native_demo', side_effect=RuntimeError('mock failure; no ML')):
                result, code = d.execute(plan)
            self.assertEqual(code, 1)
            self.assertEqual(result['status'], 'failed')
            self.assertFalse(result['included_in_statistics'])
            self.assertFalse(result['main_704_768_records_modified'])
            self.assertTrue((output / 'demo_result.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
