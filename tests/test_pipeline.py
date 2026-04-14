from src.data_loader import load_data, convert_to_binary_label

def test_data_loading():
    train_df, test_df = load_data()
    assert train_df is not None
    assert test_df is not None
    assert len(train_df) > 0
    assert len(test_df) > 0

def test_target_column_creation():
    train_df, _ = load_data()
    train_df = convert_to_binary_label(train_df)
    assert "target" in train_df.columns
    assert set(train_df["target"].unique()).issubset({0, 1})