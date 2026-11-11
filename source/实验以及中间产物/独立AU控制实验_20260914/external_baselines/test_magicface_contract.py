import unittest
from magicface_contract import AUS, PROMPT, validate_case


def case():
    return {'split': 'dev', 'input_group': 'source_plus_manual_au', 'au_order': AUS,
            'source_au_vector_0_5': [2] + [0] * 11, 'target_au_vector_0_5': [0] * 12,
            'au_delta_native': [-2] + [0] * 11, 'native_inference_steps': 50, 'prompt': PROMPT}


class ContractTests(unittest.TestCase):
    def test_negative_change_keeps_native_units(self):
        self.assertEqual(validate_case(case())[0], -2)

    def test_reject_dividing_by_five(self):
        row = case()
        row['au_delta_native'][0] = -.4
        with self.assertRaises(ValueError):
            validate_case(row)

    def test_reject_driver_field(self):
        row = case()
        row['target'] = 'a/real/target.png'
        with self.assertRaises(ValueError):
            validate_case(row)

    def test_noedit_is_full_zero_vector(self):
        row = case()
        row['target_au_vector_0_5'] = row['source_au_vector_0_5'][:]
        row['au_delta_native'] = [0] * 12
        self.assertEqual(validate_case(row), [0] * 12)

    def test_reject_test_contract(self):
        row = case()
        row['split'] = 'test'
        with self.assertRaises(ValueError):
            validate_case(row)


if __name__ == '__main__':
    unittest.main()
