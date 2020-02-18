from examples import get_data_example

if __name__ == "__main__":
    data = get_data_example()
    data.iloc[:, :].to_csv("example.csv")




