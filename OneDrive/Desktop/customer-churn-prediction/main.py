from src.data_loader import DataLoader

loader = DataLoader()

df = loader.load_data()

loader.validate_columns(df)

print(df.head())

print(df.shape)