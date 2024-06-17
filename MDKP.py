import random

def remove_items_to_balance(sorted_items1, sorted_items2, weights1, weights2, W1, W2, values):
    current_weight1 = sum(weights1)
    current_weight2 = sum(weights2)
    selected_items = set(range(len(values)))

    while current_weight1 > W1 or current_weight2 > W2:
        overfill_ratio1 = (current_weight1 - W1) / W1 if current_weight1 > W1 else 0
        overfill_ratio2 = (current_weight2 - W2) / W2 if current_weight2 > W2 else 0

        worst_dimension = 1 if overfill_ratio1 > overfill_ratio2 else 2

        if worst_dimension == 1:
            item_to_remove = sorted_items1.pop()
        else:
            item_to_remove = sorted_items2.pop()

        if item_to_remove in selected_items:
            selected_items.remove(item_to_remove)
            current_weight1 -= weights1[item_to_remove]
            current_weight2 -= weights2[item_to_remove]

    return selected_items


def balanced_knapsack_2d(values, weights1, weights2, W1, W2):
    n = len(values)

    # Step 1: Create sorted lists based on value-to-weight ratios for each dimension
    sorted_by_weight1 = sorted(range(n), key=lambda i: values[i] / weights1[i], reverse=True)
    sorted_by_weight2 = sorted(range(n), key=lambda i: values[i] / weights2[i], reverse=True)

    # Step 2: Insert all items
    selected_items = set(range(n))

    # Step 3: Iteratively remove items until the solution becomes feasible
    selected_items = remove_items_to_balance(sorted_by_weight1, sorted_by_weight2, weights1, weights2, W1, W2, values)

    # Calculate the total value of the selected items
    total_value = sum(values[i] for i in selected_items)
    selected_items = list(selected_items)

    return total_value, selected_items


values = [random.randint(1, 10) for _ in range(1000)]
weights1 = [random.randint(1, 5) for _ in range(1000)]
weights2 = [random.randint(1, 7) for _ in range(1000)]
W1 = 2000
W2 = 1500

total_value, selected_items = balanced_knapsack_2d(values, weights1, weights2, W1, W2)
print(f"values: {values}")
print(f"weights1: {weights1}")
print(f"weights2: {weights2}")
print(f"Total Value: {total_value}")
print(f"Selected Items: {selected_items}")
