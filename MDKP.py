import random

def generate_random_list(size, lower, upper):
    return [random.randint(lower, upper) for _ in range(size)]

# This first (simple) method is faster, with an nlog(n) complexity, but is packs less efficiently.
def remove_items_to_balance_simple(weights1, weights2, W1, W2, values):
    n = len(values)

    # Step 1: Create sorted lists based on value-to-weight ratios for each dimension
    sorted_by_weight1 = sorted(range(n), key=lambda i: values[i] / weights1[i], reverse=True)
    sorted_by_weight2 = sorted(range(n), key=lambda i: values[i] / weights2[i], reverse=True)

    current_weight1 = sum(weights1)
    current_weight2 = sum(weights2)
    selected_items = set(range(len(values)))

    while current_weight1 > W1 or current_weight2 > W2:
        overfill_ratio1 = (current_weight1 - W1) / W1 if current_weight1 > W1 else 0
        overfill_ratio2 = (current_weight2 - W2) / W2 if current_weight2 > W2 else 0

        worst_dimension = 1 if overfill_ratio1 > overfill_ratio2 else 2

        if worst_dimension == 1:
            item_to_remove = sorted_by_weight1.pop()
        else:
            item_to_remove = sorted_by_weight2.pop()

        if item_to_remove in selected_items:
            selected_items.remove(item_to_remove)
            current_weight1 -= weights1[item_to_remove]
            current_weight2 -= weights2[item_to_remove]

    return selected_items

# This second (vector projection based) method packs better but is slower, with an n^2 complexity.
def remove_items_to_balance_vec(weights1, weights2, W1, W2, values):
    current_weight1 = sum(weights1)
    current_weight2 = sum(weights2)
    selected_items = set(range(len(values)))

    while current_weight1 > W1 or current_weight2 > W2:
        # Calculate the direction vector
        direction_vector = (current_weight1 - W1, current_weight2 - W2)

        worst_ratio = float('inf')
        item_to_remove = -1

        for i in selected_items:
            # Project the item's resource vector onto the direction vector
            projection = weights1[i] * direction_vector[0] + weights2[i] * direction_vector[1]
            ratio = values[i] / projection if projection != 0 else float('inf')

            if ratio < worst_ratio:
                worst_ratio = ratio
                item_to_remove = i

        if item_to_remove != -1:
            selected_items.remove(item_to_remove)
            current_weight1 -= weights1[item_to_remove]
            current_weight2 -= weights2[item_to_remove]

    return selected_items


def balanced_knapsack_2d(values, weights1, weights2, W1, W2):
    # Step 1: Insert all items
    selected_items = set(range(len(values)))

    # Step 2: Iteratively remove items until the solution becomes feasible
    # selected_items = remove_items_to_balance_simple(weights1, weights2, W1, W2, values)
    selected_items = remove_items_to_balance_vec(weights1, weights2, W1, W2, values)

    # Calculate the total value of the selected items
    total_value = sum(values[i] for i in selected_items)
    selected_items = list(selected_items)

    return total_value, selected_items


# Example usage
values = generate_random_list(1000, 1, 10)
weights1 = generate_random_list(1000, 1, 10)
weights2 = generate_random_list(1000, 1, 10)
W1 = 200
W2 = 200

total_value, selected_items = balanced_knapsack_2d(values, weights1, weights2, W1, W2)
print(f"Values: {values}")
print(f"Weights1: {weights1}")
print(f"Weights2: {weights2}")
print(f"Total Value: {total_value}")
print(f"Selected Items: {selected_items}")


