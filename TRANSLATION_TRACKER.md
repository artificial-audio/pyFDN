# fdnToolbox → pyFDN translation tracker

This file tracks 1:1 translations of MATLAB files from `fdnToolbox` to

| Folder | Total | Translated | TODO |
|---|---:|---:|---:|
| `Generate` | 36 | 7 | 29 |
| `Translate` | 11 | 2 | 9 |
| `dsp` | 4 | 1 | 3 |
| `auxiliary` | 72 | 27 | 45 |

Python modules in the `pyFDN` package. Filenames are converted to Python
conventions (snake_case, `.py`). If no corresponding Python file is found the
status is `TODO`.

_Generated: 2025-11-13T15:04:12.427402Z_

## auxiliary (fdnToolbox/auxiliary)

| MATLAB file | Python candidate | pyFDN path | Status |
|---|---|---|---|
| `absorption2T60.m` | `absorption_to_t60.py` | `absorption_to_t60.py` | **Translated** |
| `absorptionFilters.m` | `absorption_filters.py` | `absorption_filters.py` | **Translated** |
| `adjPoly.m` | `adj_poly.py` | `` | **TODO** |
| `adjugate.m` | `adjugate.py` | `` | **TODO** |
| `blueWhiteRedColormap.m` | `blue_white_red_colormap.py` | `` | **TODO** |
| `circspace.m` | `circspace.py` | `` | **TODO** |
| `circulant.m` | `circulant.py` | `` | **TODO** |
| `clip.m` | `clip.py` | `` | **TODO** |
| `combnk.m` | `combnk.py` | `` | **TODO** |
| `convert2zFilter.m` | `convert2zfilter.py` | `convert2zfilter.py` | **Translated** |
| `decayFitNet2InitialLevel.m` | `decay_fit_net2_initial_level.py` | `` | **TODO** |
| `detPolynomial.m` | `det_polynomial.py` | `det_polynomial.py` | **Translated** |
| `EDC.m` | `edc.py` | `` | **TODO** |
| `firstOrderAbsorption.m` | `first_order_absorption.py` | `` | **TODO** |
| `generalCharPoly.m` | `general_char_poly.py` | `` | **TODO** |
| `generalCharPolySym.m` | `general_char_poly_sym.py` | `` | **TODO** |
| `hertz2unit.m` | `hertz2unit.py` | `hertz2unit.py` | **Translated** |
| `isBoundingCurve.m` | `is_bounding_curve.py` | `is_bounding_curve.py` | **Translated** |
| `loopTF.m` | `loop_tf.py` | `` | **TODO** |
| `lowRankApprox.m` | `low_rank_approx.py` | `` | **TODO** |
| `m2pm.m` | `m_to_pm.py` | `` | **TODO** |
| `mat.m` | `mat.py` | `` | **TODO** |
| `matrix_polyder.m` | `matrix_polyder.py` | `matrix_polyder.py` | **Translated** |
| `matrix_polyval.m` | `matrix_polyval.py` | `matrix_polyval.py` | **Translated** |
| `matrixConvolution.m` | `matrix_convolution.py` | `matrix_convolution.py` | **Translated** |
| `matrixDelayApproximation.m` | `matrix_delay_approximation.py` | `matrix_delay_approximation.py` | **Translated** |
| `maxCorr.m` | `max_corr.py` | `` | **TODO** |
| `mcircshift.m` | `mcircshift.py` | `` | **TODO** |
| `mgrpdelay.m` | `mgrpdelay.py` | `mgrpdelay.py` | **Translated** |
| `mpoly2sym.m` | `mpoly_to_sym.py` | `` | **TODO** |
| `mpolyDegree.m` | `mpoly_degree.py` | `` | **TODO** |
| `ms2smp.m` | `ms2smp.py` | `ms2smp.py` | **Translated** |
| `msym2poly.m` | `msym_to_poly.py` | `` | **TODO** |
| `negpolyder.m` | `negpolyder.py` | `negpolyder.py` | **Translated** |
| `onePoleAbsorption.m` | `one_pole_absorption.py` | `one_pole_absorption.py` | **Translated** |
| `outerSumApproximation.m` | `outer_sum_approximation.py` | `outer_sum_approximation.py` | **Translated** |
| `plotImpulseResponseMatrix.m` | `plot_impulse_response_matrix.py` | `` | **TODO** |
| `plotMatrix.m` | `plot_matrix.py` | `` | **TODO** |
| `plotSystemMatrix.m` | `plot_system_matrix.py` | `` | **TODO** |
| `poleBoundaries.m` | `pole_boundaries.py` | `pole_boundaries.py` | **Translated** |
| `poleQuality.m` | `pole_quality.py` | `` | **TODO** |
| `polyDegree.m` | `poly_degree.py` | `poly_degree.py` | **Translated** |
| `polydiag.m` | `polydiag.py` | `polydiag.py` | **Translated** |
| `printMatLatex.m` | `print_mat_latex.py` | `` | **TODO** |
| `printMatSyntax.m` | `print_mat_syntax.py` | `` | **TODO** |
| `processFDN.m` | `process_fdn.py` | `process_fdn.py` | **Translated** |
| `processTransposedFDN.m` | `process_transposed_fdn.py` | `` | **TODO** |
| `rad2hertz.m` | `rad_to_hertz.py` | `` | **TODO** |
| `randSgn.m` | `rand_sgn.py` | `` | **TODO** |
| `realLogOfNormalMatrix.m` | `real_log_of_normal_matrix.py` | `` | **TODO** |
| `reduceConjugatePairs.m` | `reduce_conjugate_pairs.py` | `` | **TODO** |
| `refinePolePositions.m` | `refine_pole_positions.py` | `` | **TODO** |
| `restoreConjugatePairs.m` | `restore_conjugate_pairs.py` | `` | **TODO** |
| `reverberationTime.m` | `reverberation_time.py` | `` | **TODO** |
| `RT602slope.m` | `rt60_to_slope.py` | `rt60_to_slope.py` | **Translated** |
| `slope2RT60.m` | `slope_to_rt60.py` | `slope_to_rt60.py` | **Translated** |
| `smp2ms.m` | `smp_to_ms.py` | `` | **TODO** |
| `snap.m` | `snap.py` | `` | **TODO** |
| `sortby.m` | `sortby.py` | `` | **TODO** |
| `tfMatrix.m` | `tf_matrix.py` | `tf_matrix.py` | **Translated** |
| `toDiagonalSimilarCanonicalForm.m` | `to_diagonal_similar_canonical_form.py` | `` | **TODO** |
| `transposeAllFields.m` | `transpose_all_fields.py` | `` | **TODO** |
| `unit2hertz.m` | `unit_to_hertz.py` | `` | **TODO** |
| `zDelay.m` | `z_delay.py` | `` | **TODO** |
| `zFDNloop.m` | `z_fdnloop.py` | `` | **TODO** |
| `zFDNloopSimple.m` | `z_fdnloop_simple.py` | `` | **TODO** |
| `zFilter.m` | `zfilter.py` | `zfilter.py` | **Translated** |
| `zFIR.m` | `zfir.py` | `zfir.py` | **Translated** |
| `zp2rpk.m` | `zp_to_rpk.py` | `` | **TODO** |
| `zScalar.m` | `zscalar.py` | `zscalar.py` | **Translated** |
| `zSOS.m` | `zsos.py` | `zsos.py` | **Translated** |
| `zTF.m` | `ztf.py` | `ztf.py` | **Translated** |

## dsp (fdnToolbox/dsp)

| MATLAB file | Python candidate | pyFDN path | Status |
|---|---|---|---|
| `complexOscillatorBank.m` | `complex_oscillator_bank.py` | `` | **TODO** |
| `dfiltMatrix.m` | `dfilt_matrix.py` | `` | **TODO** |
| `feedbackDelay.m` | `feedback_delay.py` | `feedback_delay.py` | **Translated** |
| `timeVaryingMatrix.m` | `time_varying_matrix.py` | `` | **TODO** |

## Translate (fdnToolbox/Translate)

| MATLAB file | Python candidate | pyFDN path | Status |
|---|---|---|---|
| `dss2impz.m` | `dss2impz.py` | `dss2impz.py` | **Translated** |
| `dss2impzTransposed.m` | `dss_to_impz_transposed.py` | `` | **TODO** |
| `dss2pr.m` | `dss_to_pr.py` | `` | **TODO** |
| `dss2pr_direct.m` | `dss_to_pr_direct.py` | `` | **TODO** |
| `dss2res.m` | `dss_to_res.py` | `` | **TODO** |
| `dss2ss.m` | `dss2ss.py` | `dss2ss.py` | **Translated** |
| `dss2tf.m` | `dss_to_tf.py` | `` | **TODO** |
| `dss2tfSym.m` | `dss_to_tf_sym.py` | `` | **TODO** |
| `impz2res.m` | `impz_to_res.py` | `` | **TODO** |
| `mtf2impz.m` | `mtf_to_impz.py` | `` | **TODO** |
| `pr2impz.m` | `pr_to_impz.py` | `` | **TODO** |

## Generate (fdnToolbox/Generate)

| MATLAB file | Python candidate | pyFDN path | Status |
|---|---|---|---|
| `allpassInFDN.m` | `allpass_in_fdn.py` | `` | **TODO** |
| `AndersonMatrix.m` | `anderson_matrix.py` | `` | **TODO** |
| `completeAllpassFDN.m` | `complete_allpass_fdn.py` | `` | **TODO** |
| `completeOrthogonal.m` | `complete_orthogonal.py` | `` | **TODO** |
| `constructCascadedParaunitaryMatrix.m` | `construct_cascaded_paraunitary_matrix.py` | `construct_cascaded_paraunitary_matrix.py` | **Translated** |
| `constructDelayFeedbackMatrix.m` | `construct_delay_feedback_matrix.py` | `` | **TODO** |
| `constructDelayMatrix.m` | `construct_delay_matrix.py` | `` | **TODO** |
| `constructParaunitaryFromElementals.m` | `construct_paraunitary_from_elementals.py` | `` | **TODO** |
| `constructVelvetFeedbackMatrix.m` | `construct_velvet_feedback_matrix.py` | `construct_velvet_feedback_matrix.py` | **Translated** |
| `degreeOneLossless.m` | `degree_one_lossless.py` | `` | **TODO** |
| `diagonallyEquivalent.m` | `diagonally_equivalent.py` | `` | **TODO** |
| `fdnMatrixGallery.m` | `fdn_matrix_gallery.py` | `` | **TODO** |
| `homogeneousAllpassFDN.m` | `homogeneous_allpass_fdn.py` | `` | **TODO** |
| `householderMatrix.m` | `householder_matrix.py` | `` | **TODO** |
| `interpolateOrthogonal.m` | `interpolate_orthogonal.py` | `` | **TODO** |
| `isAllpass.m` | `is_allpass.py` | `` | **TODO** |
| `isAlmostZero.m` | `is_almost_zero.py` | `is_almost_zero.py` | **Translated** |
| `isDiag.m` | `is_diag.py` | `` | **TODO** |
| `isDiagonallyEquivalentToOrthogonal.m` | `is_diagonally_equivalent_to_orthogonal.py` | `` | **TODO** |
| `isDiagonallySimilarToOrthogonal.m` | `is_diagonally_similar_to_orthogonal.py` | `` | **TODO** |
| `isParaunitary.m` | `is_paraunitary.py` | `` | **TODO** |
| `isUniallpass.m` | `is_uniallpass.py` | `` | **TODO** |
| `minRankChoiceBruteForce.m` | `min_rank_choice_brute_force.py` | `` | **TODO** |
| `mroots.m` | `mroots.py` | `` | **TODO** |
| `nearestOrthogonal.m` | `nearest_orthogonal.py` | `` | **TODO** |
| `nearestSignAgnosticOrthogonal.m` | `nearest_sign_agnostic_orthogonal.py` | `` | **TODO** |
| `nestedAllpass.m` | `nested_allpass.py` | `` | **TODO** |
| `polettiAllpass.m` | `poletti_allpass.py` | `` | **TODO** |
| `randAdmissibleHomogeneousAllpass.m` | `rand_admissible_homogeneous_allpass.py` | `` | **TODO** |
| `randomMatrixShift.m` | `random_matrix_shift.py` | `random_matrix_shift.py` | **Translated** |
| `randomOrthogonal.m` | `random_orthogonal.py` | `random_orthogonal.py` | **Translated** |
| `SchroederReverberator.m` | `schroeder_reverberator.py` | `` | **TODO** |
| `seriesAllpass.m` | `series_allpass.py` | `` | **TODO** |
| `shiftMatrix.m` | `shift_matrix.py` | `shift_matrix.py` | **Translated** |
| `shiftMatrixDistribute.m` | `shift_matrix_distribute.py` | `shift_matrix_distribute.py` | **Translated** |
| `tinyRotationMatrix.m` | `tiny_rotation_matrix.py` | `` | **TODO** |
