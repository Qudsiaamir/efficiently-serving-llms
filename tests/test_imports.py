def test_package_imports_without_ml_dependencies():
    import efficient_llm_serving

    assert efficient_llm_serving.BenchmarkConfig().num_samples > 0
