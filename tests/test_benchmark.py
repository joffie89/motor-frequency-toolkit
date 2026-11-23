from motorfreq import benchmark, data


def test_loop_and_vectorised_give_identical_tables(synthetic_raw):
    df = data.prepare(synthetic_raw)
    timings = benchmark.run(df, ["VehBrand", "Region"])
    assert timings["identical_results"].all()


def test_categorical_uses_less_memory(synthetic_raw):
    df = data.prepare(synthetic_raw)
    mem = benchmark.memory(df, ["VehBrand", "Region", "Area", "VehGas"])
    assert mem["categorical_mb"] < mem["strings_mb"]
