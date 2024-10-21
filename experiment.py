import numpy as np
import pulp
import matplotlib.pyplot as plt 

class MultiDimensionalEIP1559:
    def __init__(self, initial_base_fees, max_block_sizes):
        """
        Initialize the multi-dimensional EIP-1559 system.
        
        :param initial_base_fees: A numpy array of initial base fees for each dimension.
        :param max_block_sizes: A numpy array of maximum block sizes for each dimension.
        """
        self.base_fees = np.array(initial_base_fees,dtype=np.float64)
        self.max_block_sizes = np.array(max_block_sizes)
        self.target_block_sizes = self.max_block_sizes // 2
        self.base_fee_max_change_denominator = 16  # Maximum change per block is 1/8th of the base fee

    def update_base_fees(self, txn_values, txn_sizes, allocated_tnxs):
        """
        Update the base fees for all dimensions based on the current block sizes.
        
        :param current_block_sizes: A numpy array of current block sizes for each dimension.
        :return: Updated base fees as a numpy array.
        """

        dim = len(self.max_block_sizes)
        current_block_sizes = np.zeros(dim)

        for d in range(dim):
            for txn in allocated_tnxs:
                current_block_sizes[d] += txn_sizes[txn][d]

        delta = current_block_sizes - self.target_block_sizes

        #print(f"Current Block Size: {current_block_sizes}")
        #print(f"Delta: {delta}")

        # Increase or decrease the base fee based on the block usage
        base_fee_changes = (self.base_fees * delta) / self.target_block_sizes / self.base_fee_max_change_denominator
        self.base_fees += base_fee_changes
        self.base_fees = np.maximum(self.base_fees, [1 for d in range(dim)])

        
        # for d in range(dim):
        #     if current_block_sizes[d] > self.target_block_sizes[d]:
        #         self.base_fees[d] += base_fee_changes[d]
        #     else:
        #         self.base_fees[d] += base_fee_changes[d]

        # self.base_fees += np.where(current_block_sizes > self.target_block_sizes, base_fee_changes, -base_fee_changes)

        #print(f"Base Fee: {self.base_fees}")

        return self.base_fees

    def calculate_fees(self, gas_usages, priority_fees):
        """
        Calculate the total fee for a transaction across all dimensions.
        
        :param gas_usages: A numpy array of gas usage for each dimension.
        :param priority_fees: A numpy array of priority fees for each dimension.
        :return: The total fee for the transaction.
        """
        gas_usages = np.array(gas_usages)
        priority_fees = np.array(priority_fees)
        total_fee = np.sum(gas_usages * (self.base_fees + priority_fees))
        
        return total_fee

    def allocate(self, txn_values, txn_sizes):
        ## In the case that txns above base fee exceed max block capacity, allocates favorably to higher txns 
        ## Thus only lose welfare in the case that the base fee is too high 
        n = len(txn_values)
        base_fee_vector = self.base_fees
        allocated_tnxs = [i for i in range(n) if txn_values[i] >= np.sum(txn_sizes[i] * base_fee_vector)]

        dim = len(self.max_block_sizes)

        selected_txn_bids = [txn_values[i] for i in allocated_tnxs]
        selected_txn_sizes = [txn_sizes[i] for i in allocated_tnxs]


        gas_usage = [0 for d in range(dim)]
        for d in range(dim):
            for txn in allocated_tnxs:
                gas_usage[d] += txn_sizes[txn][d]

        # if(any([gas_usage[d] > self.max_block_sizes[d] for d in range(dim)])):
        #     print("oof")

        new_txns = solve_multidimensional_knapsack(selected_txn_bids, selected_txn_sizes, self.max_block_sizes)

        selected_txns = [allocated_tnxs[i] for i in new_txns]

        #print(f"Gas Used: {gas_usage}")
        #print(max_block_sizes)

        return selected_txns


class FPA:
    def __init__(self, max_block_sizes):
        self.max_block_sizes = np.array(max_block_sizes)

    def allocate(self, txn_bids, txn_sizes):
        ## Run Optimal Multidimensional Knapsack Algorithm Here 
        allocated_txns = solve_multidimensional_knapsack(txn_bids, txn_sizes, self.max_block_sizes)
        return allocated_txns


class Wallet:
    def __init__(self, historical_bids, historical_sizes, max_block_sizes, num_blocks):
        self.max_block_sizes = np.array(max_block_sizes)
        self.historical_bids = np.array(historical_bids)
        self.historical_sizes = np.array(historical_sizes)
        self.num_blocks = num_blocks


    def gen_threshold_bid(self,sample_txn_bids,sample_txn_sizes,txn_size, block_size):
        min_threshold_bid = np.min(sample_txn_bids)
        max_threshold_bid = np.max(sample_txn_bids)

        sample_bids = np.insert(sample_txn_bids, 0, 0)
        sample_sizes = np.insert(sample_txn_sizes, 0, np.array(txn_size))
        # print(txn_size)
        # print(sample_txn_sizes)
        # print(sample_sizes)

        sample_txn_bids[0] = 0
        sample_txn_sizes[0] = txn_size

        while(min_threshold_bid < max_threshold_bid-3):
            cur_bid = round((min_threshold_bid + max_threshold_bid)/2 )
            #print(min_threshold_bid,max_threshold_bid,cur_bid)
            sample_txn_bids[0] = cur_bid 
            included_txns = solve_fractional_multidimensional_knapsack(sample_txn_bids,sample_txn_sizes,block_size)
            if 0 in included_txns:
                max_threshold_bid = cur_bid 
            else:
                min_threshold_bid = cur_bid 

        return max_threshold_bid 


    def get_bid(self, txn_size):
        return  0 
        block_size = np.round(self.max_block_sizes/1.1)
        num_draws = 1
        threshold_bids = []

        n = round(len(self.historical_bids)/self.num_blocks)

        indices = [i for i in range(len(self.historical_bids))]
        #print(indices)

        for i in range(num_draws):

            sample = np.random.choice(indices, size=n, replace=False)

            txn_sample_bids =  [self.historical_bids[i] for i in sample]
            txn_samples_sizes = [self.historical_sizes[i] for i in sample]
            x = self.gen_threshold_bid(txn_sample_bids,txn_samples_sizes,txn_size,block_size)
            threshold_bids.append(x)
            

        return np.median(np.array(threshold_bids))



def solve_multidimensional_knapsack(values, weights, capacities):
    # Example usage:
    # values = [60, 100, 120]
    # weights = [(10, 20), (20, 30), (30, 50)]
    # capacities = [50, 50]
    # selected_items, total_value = solve_multidimensional_knapsack(values, weights, capacities)

    # Number of items
    n_items = len(values)

    # Create the problem
    prob = pulp.LpProblem("MultidimensionalKnapsackProblem", pulp.LpMaximize)

    # Decision variables: x[i] is 1 if item i is included, 0 otherwise
    x = pulp.LpVariable.dicts("x", range(n_items), cat='Binary')

    # Objective function: maximize the total value
    prob += pulp.lpSum(values[i] * x[i] for i in range(n_items))

    # Constraints: ensure the total weight for each dimension is within the capacity
    for j in range(len(capacities)):
        prob += pulp.lpSum(weights[i][j] * x[i] for i in range(n_items)) <= capacities[j]

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract the results
    selected_items = [i for i in range(n_items) if pulp.value(x[i]) == 1]
    #total_value = pulp.value(prob.objective)

    return selected_items


def solve_fractional_multidimensional_knapsack(values,weights,capacities):
       # Number of items
    n_items = len(values)

    # Create the problem
    prob = pulp.LpProblem("MultidimensionalKnapsackProblem", pulp.LpMaximize)

    # Decision variables: x[i] is 1 if item i is included, 0 otherwise
    x = pulp.LpVariable.dicts("x", range(n_items), cat='Continuous')

    # Objective function: maximize the total value
    prob += pulp.lpSum(values[i] * x[i] for i in range(n_items))

    # Constraints: ensure the total weight for each dimension is within the capacity
    for j in range(len(capacities)):
        prob += pulp.lpSum(weights[i][j] * x[i] for i in range(n_items)) <= capacities[j]

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract the results
    selected_items = [i for i in range(n_items) if pulp.value(x[i]) > 0]
    #total_value = pulp.value(prob.objective)

    return selected_items


def gen_slot_data(demand):
    #demand = (num_samples, value, size_1, size_2)
    num_samples = demand[0]
    value = demand[1]
    size_1 = demand[2]
    size_2 = demand[3]

    #Draw num_samples from an exponential distribution with parameter value 
    value_samples = 10*np.round(np.random.poisson(value, size=num_samples))
    #Draw  num_samples from a two-dimensional Poisson distribution
    size_samples = np.array([np.random.poisson(size_1, num_samples), np.random.poisson(size_2, num_samples)]).T
    #size_samples = np.ones((num_samples,2))

    return value_samples, size_samples


def gen_demand_schedule(num_blocks):
    num_samples = 100
    x = 100 
    value = 20
    size_1 = 5
    size_2 = 5

    alpha = 300

    demand_schedule = []  
    for i in range(num_blocks):
        if i==25:
            num_samples = 200 
            value = 35 
        if i==50:
            num_samples = 100 
            value = 20
        #num_samples = round(np.random.poisson(alpha))
        demand_schedule.append([num_samples,value,size_1,size_2])
    return demand_schedule


def sim_slot(txn_values, txn_sizes, base_fees, max_block_sizes, wallet):
    eip1559 = MultiDimensionalEIP1559(base_fees, max_block_sizes)
    first_price = FPA(np.array(max_block_sizes) //2)
    allocated_tnxs_1559 = eip1559.allocate(txn_values,txn_sizes)

    txn_bids = [min(txn_values[i], wallet.get_bid(txn_sizes[i])) for i in range(len(txn_values))]
    allocated_tnxs_FPA = first_price.allocate(txn_values,txn_sizes)

    updated_base_fees = eip1559.update_base_fees(txn_values,txn_sizes,allocated_tnxs_1559)
    return allocated_tnxs_1559, allocated_tnxs_FPA, updated_base_fees, txn_bids 


def sim(demand_schedule):
    base_fees = np.array([50, 50])
    max_block_sizes = np.array([200, 200])

    num_blocks = len(demand_schedule)

    apx = 0 
    cnt = 0 

    # Lists to store welfare values at each iteration
    welfare_1559_list = []
    welfare_FPA_list = []
    bid_history = [] 
    size_history = []

    history_len = 3 

    for i in range(history_len+1):
        txn_values, txn_sizes = gen_slot_data(demand_schedule[0]) 
        bid_history.append(txn_values)
        size_history.append(txn_sizes)



    for i in range(num_blocks):
        print(i)
        txn_values, txn_sizes = gen_slot_data(demand_schedule[i]) 


        #temp_size_history = np.array(size_history[-history_len:]).reshape(history_len * len(txn_values), 2)
        #temp_size_history = [size_history[i][j] for j in range(len(size_history[i])) for i in range(-history_len,0)]
        
        temp_size_history = []
        temp_bid_history = [] 
        for i in range(-history_len,0):
            for x in size_history[i]:
                temp_size_history.append(x)
                #print(i,x)
            for x in bid_history[i]:
                temp_bid_history.append(x)

        cur_wallet = Wallet(temp_bid_history, temp_size_history,max_block_sizes,history_len)
        allocated_tnxs_1559, allocated_tnxs_FPA, base_fees, txn_bids = sim_slot(txn_values, txn_sizes, base_fees, max_block_sizes,cur_wallet)


        allocated_txn_sizes = [txn_sizes[txn] for txn in allocated_tnxs_FPA]
        allocated_txn_bids = [txn_bids[txn] for txn in allocated_tnxs_FPA]
        bid_history.append(allocated_txn_bids)
        size_history.append(allocated_txn_sizes)



        welfare_1559 = 0 
        welfare_FPA = 0 

        for txn in allocated_tnxs_1559:
            welfare_1559 += txn_values[txn]

        for txn in allocated_tnxs_FPA:
            welfare_FPA += txn_values[txn]

        # Store the welfare values
        welfare_1559_list.append(welfare_1559)
        welfare_FPA_list.append(welfare_FPA)
        
        if welfare_FPA != 0 and i > 50:
            cnt += 1 
            apx += welfare_1559/welfare_FPA

    # Plot the welfare values
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_blocks), welfare_1559_list, label='Welfare 1559')
    plt.plot(range(num_blocks), welfare_FPA_list, label='Welfare FPA')
    plt.xlabel('Iteration')
    plt.ylabel('Welfare')
    plt.title('Welfare Comparison Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()



schedule = gen_demand_schedule(75)
sim(schedule)

