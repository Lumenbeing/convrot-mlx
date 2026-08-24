#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Comfy Org. All rights reserved.
# SPDX-FileCopyrightText: 2026 Lumenbeing contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np

from convrot_mlx import (
    ConvRotLinear,
    compiled_prepare_convrot_activation,
    expanded_fp32_w4a4_matmul,
    metal_fused_prepare_convrot_activation,
    metal_fused_prepare_convrot_activation_i8,
    metal_fused_rotate_convrot_activation,
    metal_unpack_signed_int4,
    mpp_w4a4_matmul,
    mpp_w4a16_matmul,
    pack_signed_int4,
    packed_w4a4_matmul,
    packed_w4a4_matmul_k32,
    packed_w4a4_matmul_k64,
    packed_w4a4_matmul_k128,
    packed_w4a4_matmul_k256,
    packed_w4a4_matmul_m16n64_k64,
    packed_w4a4_matmul_m32n32_k64,
    prepare_convrot_activation,
    quantize_convrot_weight,
    quantize_signed_int4_rowwise,
    reference_w4a4_matmul,
    regular_hadamard,
    repack_mpp_weight_to_nk,
    repack_nk_weight_for_mpp,
    unpack_signed_int4,
)


def regular_hadamard_matrix(size: int) -> np.ndarray:
    h4 = np.array(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=np.float32,
    )
    matrix = h4
    while matrix.shape[0] < size:
        matrix = np.kron(matrix, h4)
    return matrix / np.sqrt(size)


class ConvRotTests(unittest.TestCase):
    def test_compiled_activation_preparation_matches_eager_contract(self):
        rng = np.random.default_rng(17)
        values = mx.array(rng.normal(size=(19, 512)).astype(np.float32)).astype(
            mx.bfloat16
        )
        expected = prepare_convrot_activation(values)
        actual = compiled_prepare_convrot_activation(values, 256)
        mx.eval(expected, actual)
        self.assertTrue(bool(mx.all(expected[0] == actual[0]).item()))
        self.assertTrue(bool(mx.all(expected[1] == actual[1]).item()))

    def test_fused_metal_activation_preparation_matches_eager_contract(self):
        rng = np.random.default_rng(18)
        values = mx.array(rng.normal(size=(19, 512)).astype(np.float32)).astype(
            mx.bfloat16
        )
        expected = prepare_convrot_activation(values)
        actual = metal_fused_prepare_convrot_activation(values)
        mx.eval(expected, actual)
        self.assertTrue(bool(mx.all(expected[0] == actual[0]).item()))
        self.assertTrue(bool(mx.all(expected[1] == actual[1]).item()))

    def test_fused_metal_i8_activation_preparation_matches_packed_contract(self):
        rng = np.random.default_rng(1801)
        values = mx.array(rng.normal(size=(19, 512)).astype(np.float32)).astype(
            mx.bfloat16
        )
        expected_qact, expected_scales = prepare_convrot_activation(values)
        actual_i8, actual_scales = metal_fused_prepare_convrot_activation_i8(values)
        expected_i8 = unpack_signed_int4(expected_qact)
        mx.eval(expected_i8, expected_scales, actual_i8, actual_scales)
        self.assertTrue(bool(mx.all(expected_i8 == actual_i8).item()))
        self.assertTrue(bool(mx.all(expected_scales == actual_scales).item()))

    def test_fused_metal_rotation_matches_eager_contract(self):
        rng = np.random.default_rng(1802)
        values = mx.array(rng.normal(size=(19, 512)).astype(np.float32)).astype(
            mx.bfloat16
        )
        expected = regular_hadamard(values)
        actual = metal_fused_rotate_convrot_activation(values)
        mx.eval(expected, actual)
        self.assertTrue(bool(mx.all(expected == actual).item()))

    def test_group64_fused_preparation_matches_eager_at_h3_k2688(self):
        rng = np.random.default_rng(2026082305)
        values = mx.array(rng.normal(size=(17, 2688)).astype(np.float32)).astype(
            mx.bfloat16
        )
        expected_qact, expected_scales = prepare_convrot_activation(values, 64)
        actual_qact, actual_scales = metal_fused_prepare_convrot_activation(values, 64)
        actual_i8, actual_i8_scales = metal_fused_prepare_convrot_activation_i8(
            values, 64
        )
        expected_i8 = unpack_signed_int4(expected_qact)
        expected_rotation = regular_hadamard(values, 64)
        actual_rotation = metal_fused_rotate_convrot_activation(values, 64)
        mx.eval(
            expected_qact,
            expected_scales,
            actual_qact,
            actual_scales,
            expected_i8,
            actual_i8,
            actual_i8_scales,
            expected_rotation,
            actual_rotation,
        )
        self.assertTrue(bool(mx.all(expected_qact == actual_qact).item()))
        self.assertTrue(bool(mx.all(expected_scales == actual_scales).item()))
        self.assertTrue(bool(mx.all(expected_i8 == actual_i8).item()))
        self.assertTrue(bool(mx.all(expected_scales == actual_i8_scales).item()))
        self.assertTrue(bool(mx.all(expected_rotation == actual_rotation).item()))

    def test_fused_metal_activation_preparation_is_exact_at_route_shape(self):
        rng = np.random.default_rng(20260823)
        values = mx.array(rng.normal(size=(1536, 3072)).astype(np.float32)).astype(
            mx.bfloat16
        )
        expected = prepare_convrot_activation(values)
        actual = metal_fused_prepare_convrot_activation(values)
        mx.eval(expected, actual)
        self.assertTrue(bool(mx.all(expected[0] == actual[0]).item()))
        self.assertTrue(bool(mx.all(expected[1] == actual[1]).item()))

    def test_staged_tile_variants_match_the_eager_integer_oracle(self):
        rng = np.random.default_rng(29)
        qact = pack_signed_int4(
            mx.array(rng.integers(-7, 8, size=(17, 512), dtype=np.int8))
        )
        qweight = pack_signed_int4(
            mx.array(rng.integers(-7, 8, size=(39, 512), dtype=np.int8))
        )
        xscales = mx.array(rng.uniform(0.01, 0.2, size=(17,)).astype(np.float32))
        wscales = mx.array(rng.uniform(0.01, 0.2, size=(39,)).astype(np.float32))
        bias = mx.array(rng.normal(size=(39,)).astype(np.float32))
        expected = reference_w4a4_matmul(
            qact,
            qweight,
            xscales,
            wscales,
            bias,
            output_dtype=mx.bfloat16,
        )
        for backend in (
            packed_w4a4_matmul_k32,
            packed_w4a4_matmul_k64,
            packed_w4a4_matmul_k128,
            packed_w4a4_matmul_k256,
            packed_w4a4_matmul_m32n32_k64,
            packed_w4a4_matmul_m16n64_k64,
        ):
            actual = backend(
                qact,
                qweight,
                xscales,
                wscales,
                bias,
                output_dtype=mx.bfloat16,
            )
            mx.eval(actual)
            self.assertTrue(bool(mx.all(expected == actual).item()), backend.__name__)

    def test_radix_transform_matches_regular_hadamard(self):
        rng = np.random.default_rng(17)
        values = rng.normal(size=(3, 512)).astype(np.float32)
        result = regular_hadamard(mx.array(values))
        mx.eval(result)
        h = regular_hadamard_matrix(256)
        expected = np.concatenate([values[:, :256] @ h, values[:, 256:] @ h], axis=1)
        np.testing.assert_allclose(np.array(result), expected, rtol=2e-6, atol=2e-6)

    def test_pack_layout_matches_signed_low_nibble_contract(self):
        values = mx.array(np.array([[-8, -7, -1, 0, 1, 7, 3, -4]], dtype=np.int8))
        packed = pack_signed_int4(values)
        unpacked = unpack_signed_int4(packed)
        mx.eval(packed, unpacked)
        self.assertEqual(np.array(packed).tolist(), [[0x98, 0x0F, 0x71, 0xC3]])
        self.assertEqual(np.array(unpacked).tolist(), np.array(values).tolist())

    def test_direct_metal_unpack_matches_eager_contract(self):
        rng = np.random.default_rng(2026082300)
        values = mx.array(rng.integers(-8, 8, size=(37, 64), dtype=np.int8))
        packed = pack_signed_int4(values)
        actual = metal_unpack_signed_int4(packed)
        expected = unpack_signed_int4(packed)
        mx.eval(actual, expected)
        self.assertTrue(bool(mx.all(actual == expected).item()))

    def test_mpp_weight_layout_repack_is_bit_exact_and_reversible(self):
        rng = np.random.default_rng(2026082301)
        values = mx.array(rng.integers(-7, 8, size=(46, 64), dtype=np.int8))
        packed_nk = pack_signed_int4(values)
        packed_kn = repack_nk_weight_for_mpp(packed_nk)
        round_trip = repack_mpp_weight_to_nk(packed_kn)
        mx.eval(packed_nk, packed_kn, round_trip)
        self.assertEqual(packed_kn.shape, (64, 23))
        self.assertTrue(bool(mx.all(packed_nk == round_trip).item()))

    def test_mpp_w4a4_tensorop_matches_integer_oracle_at_edges(self):
        rng = np.random.default_rng(2026082302)
        activations = mx.array(rng.integers(-7, 8, size=(37, 64), dtype=np.int8))
        weights = mx.array(rng.integers(-7, 8, size=(46, 64), dtype=np.int8))
        qact = pack_signed_int4(activations)
        qweight_nk = pack_signed_int4(weights)
        qweight_mpp = repack_nk_weight_for_mpp(qweight_nk)
        xscales = mx.array(rng.uniform(0.01, 0.2, size=(37,)).astype(np.float32))
        wscales = mx.array(rng.uniform(0.01, 0.2, size=(46,)).astype(np.float32))
        bias = mx.array(rng.normal(size=(46,)).astype(np.float32))
        actual = mpp_w4a4_matmul(
            qact,
            qweight_mpp,
            xscales,
            wscales,
            bias,
            output_dtype=mx.float32,
        )
        expected = reference_w4a4_matmul(
            qact,
            qweight_nk,
            xscales,
            wscales,
            bias,
            output_dtype=mx.float32,
        )
        mx.eval(actual, expected)
        np.testing.assert_allclose(
            np.array(actual), np.array(expected), rtol=1e-6, atol=1e-6
        )

        unit_xscales = mx.ones((37,), dtype=mx.float32)
        unit_wscales = mx.ones((46,), dtype=mx.float32)
        zero_bias = mx.zeros((46,), dtype=mx.float32)
        actual_integer = mpp_w4a4_matmul(
            qact,
            qweight_mpp,
            unit_xscales,
            unit_wscales,
            zero_bias,
            output_dtype=mx.float32,
        )
        expected_integer = reference_w4a4_matmul(
            qact,
            qweight_nk,
            unit_xscales,
            unit_wscales,
            zero_bias,
            output_dtype=mx.float32,
        )
        mx.eval(actual_integer, expected_integer)
        self.assertTrue(bool(mx.all(actual_integer == expected_integer).item()))

        legacy_bfloat16 = packed_w4a4_matmul_m32n32_k64(
            qact,
            qweight_nk,
            xscales,
            wscales,
            bias,
            output_dtype=mx.bfloat16,
        )
        native_bfloat16 = mpp_w4a4_matmul(
            qact,
            qweight_mpp,
            xscales,
            wscales,
            bias,
            output_dtype=mx.bfloat16,
        )
        mx.eval(legacy_bfloat16, native_bfloat16)
        self.assertTrue(bool(mx.all(legacy_bfloat16 == native_bfloat16).item()))

    def test_mpp_w4a16_tensorop_matches_fp32_oracle_at_edges(self):
        rng = np.random.default_rng(2026082303)
        weights = mx.array(rng.integers(-7, 8, size=(46, 64), dtype=np.int8))
        qweight_mpp = repack_nk_weight_for_mpp(pack_signed_int4(weights))
        wscales = mx.array(rng.uniform(0.01, 0.2, size=(46,)).astype(np.float32))
        bias = mx.array(rng.normal(size=(46,)).astype(np.float32))
        for dtype in (mx.float16, mx.bfloat16):
            with self.subTest(dtype=str(dtype)):
                activations = mx.array(
                    rng.normal(size=(37, 64)).astype(np.float32)
                ).astype(dtype)
                actual = mpp_w4a16_matmul(
                    activations,
                    qweight_mpp,
                    wscales,
                    bias,
                    output_dtype=mx.float32,
                )
                expected = (
                    activations.astype(mx.float32) @ weights.astype(mx.float32).T
                ) * wscales[None, :] + bias[None, :]
                mx.eval(actual, expected)
                np.testing.assert_allclose(
                    np.array(actual), np.array(expected), rtol=1e-6, atol=1e-5
                )

    def test_packed_metal_kernel_matches_eager_integer_oracle(self):
        rng = np.random.default_rng(23)
        activations = mx.array(rng.normal(size=(19, 256)).astype(np.float16))
        weights = mx.array(rng.normal(size=(37, 256)).astype(np.float16))
        qact, xscales = quantize_signed_int4_rowwise(activations)
        qweight, wscales = quantize_signed_int4_rowwise(weights)
        bias = mx.array(rng.normal(size=(37,)).astype(np.float32))
        actual = packed_w4a4_matmul(
            qact, qweight, xscales, wscales, bias, output_dtype=mx.float32
        )
        expected = reference_w4a4_matmul(
            qact, qweight, xscales, wscales, bias, output_dtype=mx.float32
        )
        mx.eval(actual, expected)
        np.testing.assert_allclose(
            np.array(actual), np.array(expected), rtol=1e-6, atol=2e-5
        )

    def test_expanded_mlx_backend_is_bit_exact_at_bfloat16_output(self):
        rng = np.random.default_rng(27)
        activations = mx.array(rng.integers(-7, 8, size=(31, 1024), dtype=np.int8))
        weights = mx.array(rng.integers(-7, 8, size=(61, 1024), dtype=np.int8))
        qact = pack_signed_int4(activations)
        qweight = pack_signed_int4(weights)
        xscales = mx.array(rng.uniform(0.01, 0.2, size=(31,)).astype(np.float32))
        wscales = mx.array(rng.uniform(0.01, 0.2, size=(61,)).astype(np.float32))
        bias = mx.array(rng.normal(size=(61,)).astype(np.float32))
        actual = expanded_fp32_w4a4_matmul(
            qact,
            qweight,
            xscales,
            wscales,
            bias,
            output_dtype=mx.bfloat16,
        )
        expected = reference_w4a4_matmul(
            qact,
            qweight,
            xscales,
            wscales,
            bias,
            output_dtype=mx.bfloat16,
        )
        mx.eval(actual, expected)
        self.assertTrue(bool(mx.all(actual == expected).item()))

    def test_k32_packed_kernel_matches_eager_integer_oracle(self):
        rng = np.random.default_rng(28)
        activations = mx.array(rng.normal(size=(19, 1024)).astype(np.float16))
        weights = mx.array(rng.normal(size=(37, 1024)).astype(np.float16))
        qact, xscales = quantize_signed_int4_rowwise(activations)
        qweight, wscales = quantize_signed_int4_rowwise(weights)
        bias = mx.array(rng.normal(size=(37,)).astype(np.float32))
        actual = packed_w4a4_matmul_k32(
            qact, qweight, xscales, wscales, bias, output_dtype=mx.float32
        )
        expected = reference_w4a4_matmul(
            qact, qweight, xscales, wscales, bias, output_dtype=mx.float32
        )
        mx.eval(actual, expected)
        np.testing.assert_allclose(
            np.array(actual), np.array(expected), rtol=1e-6, atol=2e-5
        )

    def test_layer_runs_the_same_contract_at_bfloat16(self):
        rng = np.random.default_rng(29)
        dense_weight = mx.array(rng.normal(size=(33, 256)).astype(np.float32)).astype(
            mx.bfloat16
        )
        dense_bias = mx.array(rng.normal(size=(33,)).astype(np.float32))
        qweight, wscales = quantize_convrot_weight(dense_weight)
        layer = ConvRotLinear(256, 33, backend="metal")
        layer.qweight = qweight
        layer.scales = wscales
        layer.bias = dense_bias
        values = mx.array(rng.normal(size=(2, 7, 256)).astype(np.float32)).astype(
            mx.bfloat16
        )
        actual = layer(values)

        rotated = regular_hadamard(values.reshape(-1, 256))
        qact, xscales = quantize_signed_int4_rowwise(rotated)
        expected = reference_w4a4_matmul(
            qact,
            qweight,
            xscales,
            wscales,
            dense_bias,
            output_dtype=mx.bfloat16,
        ).reshape(2, 7, 33)
        mx.eval(actual, expected)
        self.assertTrue(bool(mx.all(actual == expected).item()))

    def test_mpp_layers_load_kn_layout_and_run_both_activation_modes(self):
        rng = np.random.default_rng(2026082304)
        dense_weight = mx.array(rng.normal(size=(34, 256)).astype(np.float32)).astype(
            mx.bfloat16
        )
        dense_bias = mx.array(rng.normal(size=(34,)).astype(np.float32))
        qweight_nk, wscales = quantize_convrot_weight(dense_weight)
        qweight_mpp = repack_nk_weight_for_mpp(qweight_nk)
        values = mx.array(rng.normal(size=(2, 7, 256)).astype(np.float32)).astype(
            mx.bfloat16
        )

        for backend in ("mpp-w4a4", "mpp-w4a16"):
            with self.subTest(backend=backend):
                layer = ConvRotLinear(256, 34, backend=backend)
                layer.qweight = qweight_mpp
                layer.scales = wscales
                layer.bias = dense_bias
                actual = layer(values)
                mx.eval(actual)
                self.assertEqual(actual.shape, (2, 7, 34))
                self.assertTrue(bool(mx.all(mx.isfinite(actual)).item()))

    def test_group64_mpp_layers_run_h3_k2688_contract(self):
        rng = np.random.default_rng(2026082306)
        dense_weight = mx.array(rng.normal(size=(34, 2688)).astype(np.float32)).astype(
            mx.bfloat16
        )
        dense_bias = mx.array(rng.normal(size=(34,)).astype(np.float32))
        qweight_nk, wscales = quantize_convrot_weight(dense_weight, 64)
        qweight_mpp = repack_nk_weight_for_mpp(qweight_nk)
        values = mx.array(rng.normal(size=(2, 7, 2688)).astype(np.float32)).astype(
            mx.bfloat16
        )

        for backend in ("mpp-w4a4", "mpp-w4a16"):
            with self.subTest(backend=backend):
                layer = ConvRotLinear(2688, 34, group_size=64, backend=backend)
                layer.qweight = qweight_mpp
                layer.scales = wscales
                layer.bias = dense_bias
                actual = layer(values)
                mx.eval(actual)
                self.assertEqual(actual.shape, (2, 7, 34))
                self.assertTrue(bool(mx.all(mx.isfinite(actual)).item()))


if __name__ == "__main__":
    unittest.main()
