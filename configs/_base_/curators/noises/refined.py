noise_filter = dict(
    assay=dict(
        measurement_type=["EC50"],
        assay_value_units=["nM", "uM"],
        molecules_number=[32, 5000],
        confidence_score=3),
    sample=dict(
        filter_none=[],
        smile_exist=[],
        smile_legal=[],
        value_relation=["=", ">=", "<=", "~"])
)
