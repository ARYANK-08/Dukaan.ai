import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

def generate_shelves(data_file):
    # Read the CSV file and replace 'nan' values with ''
    data = pd.read_csv(data_file, header=None).fillna('')
    transactions = data.applymap(str).values.tolist()

    # Encode the transactions
    te = TransactionEncoder()
    dataset = te.fit_transform(transactions)
    dataset = pd.DataFrame(dataset, columns=te.columns_)

    # Encode units
    def encode_units(x):
        return 1 if x else 0

    dataset = dataset.applymap(encode_units)

    # Generate frequent itemsets
    frequent_itemsets = apriori(dataset, min_support=0.01, use_colnames=True)

    # Add length column to frequent_itemsets
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)

    # Filter frequent itemsets with lengths 2, 3, and 4
    frequent_itemsets_filtered = frequent_itemsets[
        (frequent_itemsets['length'] >= 2) &
        (frequent_itemsets['length'] <= 4) &
        (~frequent_itemsets['itemsets'].apply(lambda x: '' in x))
    ]

    frequent_itemsets_filtered_dict = dict(zip(frequent_itemsets_filtered['itemsets'], frequent_itemsets_filtered['support']))
    # Compute the sum of occurrences for each product
    product_counts = dataset.sum()

    # Sort products based on popularity (descending order)
    popular_products = product_counts.sort_values(ascending=False)

    # Export top 10 selling products
    top_selling_products = popular_products.head(50)

    # Define the number of shelves
    num_shelves = 25

    # Calculate the number of products per shelf
    products_per_shelf = len(popular_products) // num_shelves

    # Initialize a set to keep track of assigned products
    assigned_products = set()

    # Assign products to shelves
    shelves = {}
    for i in range(num_shelves):
        start_idx = i * products_per_shelf
        end_idx = (i + 1) * products_per_shelf
        if i == num_shelves - 1:  # Last shelf may have fewer products
            end_idx = len(popular_products)
        shelf_products = []
        for product in popular_products[start_idx:end_idx].index:
            if product not in assigned_products:
                shelf_products.append(product)
                assigned_products.add(product)
        shelves[f"Shelf {i+1}"] = shelf_products

    return shelves, frequent_itemsets_filtered_dict, top_selling_products
