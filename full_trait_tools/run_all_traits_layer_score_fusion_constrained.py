#!/usr/bin/env python3
"""Constrained mild layer-28 residual fusion for all-traits scores."""

from full_trait_tools.run_all_traits_layer_score_fusion_quick import main
import full_trait_tools.run_all_traits_layer_score_fusion_quick as quick


quick.ALPHAS = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4]


if __name__ == "__main__":
    main()
